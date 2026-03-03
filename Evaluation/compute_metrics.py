"""
For a list of WandB model runs, compute per-patch segmentation metrics and
per-patch Betti-number curves on the test set, then write results to CSV.

Betti numbers are computed on individual 1024×1024 patches

Outputs (written next to this script in Evaluation/):
  metrics_<YYYYMMDD_HHMMSS>.csv
      Per-model aggregate metrics (Dice, IoU, clDice, tprec, tsens – raw and
      post-processed) plus Betti-0 F-score and Betti-1 MAE at threshold 0.5.
      Previously computed models are loaded and merged in so re-runs are cheap.

  persistence_<YYYYMMDD_HHMMSS>.csv
      One row per birth-death pair per patch.
      Columns: model, image, type (predicted/groundtruth), dim (0 or 1),
      birth, death — all in filtration space (filtration = 1 - probability).

Usage: python compute_metrics.py [--device CUDA_IDX] [--force-recompute]
"""

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb

# make the repo root importable
_EVAL_DIR = Path(__file__).resolve().parent

import Segmentation.SAM.sam_lora_util as sam_lora_util
import Segmentation.Util.env_utils as env_utils

from Evaluation.evaluation_util import (
    collect_patch_metrics_and_betti,
    load_latest_metrics_csv,
    load_latest_persistence_csv,
)

MODELS_RUNS = ["SAM_LoRA_Finetuning_20260219_150640"]

PARAMS_ARTIFACT_SUFFIX = "_params_minloss:v0"


def _collect_combined(model, test_img_dir, test_mask_dir, downsample_size, device, batch_size, run_name):
    """Run patch-level inference once per batch, returning segmentation metrics
    and per-patch persistence diagram rows.
    """
    dataset = sam_lora_util.SegmentationDataset(
        test_img_dir, test_mask_dir, downsample_size=downsample_size, return_img_name=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return collect_patch_metrics_and_betti(model, loader, device, run_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for patch-level inference (default: 8)")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-evaluate all models, ignoring cached CSVs")
    parser.add_argument("--dataset", type=str, default=None,
                        help="dataset for evaluation, replaces run dataset if provided")
    args = parser.parse_args()

    env_utils.load_segmentation_env()

    SEED = env_utils.load_as("SEED", int, 42)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DATASETS_DIR = os.getenv("DATASETS_DIR")
    HIGHRES_IMG_PATCHES_DIR_NAME = os.getenv(
        "HIGHRES_IMG_PATCHES_DIR_NAME", "img_patches_1024")
    HIGHRES_MASK_PATCHES_DIR_NAME = os.getenv(
        "HIGHRES_MASK_PATCHES_DIR_NAME", "mask_patches_1024")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ForkSight-SAM")

    if DATASETS_DIR is None:
        raise ValueError("DATASETS_DIR environment variable must be set.")

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_prev_metrics = load_latest_metrics_csv(
        _EVAL_DIR) if not args.force_recompute else pd.DataFrame()
    df_prev_persistence = load_latest_persistence_csv(
        _EVAL_DIR) if not args.force_recompute else pd.DataFrame()
    computed_models = set(
        df_prev_metrics.index) if not df_prev_metrics.empty else set()

    api = wandb.Api()
    all_runs = list(api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}"))
    runs_to_eval = [
        r for r in all_runs
        if sam_lora_util.EVALUATED_TAG in r.tags
        and r.state == "finished"
        and r.name in MODELS_RUNS
        and r.name not in computed_models
    ]

    if not runs_to_eval:
        print("No new models to evaluate, all results already cached.")
    else:
        print(f"Will evaluate {len(runs_to_eval)} model(s): "
              + ", ".join(r.name for r in runs_to_eval))

    metrics_results = {}
    all_persistence_rows = []

    for run in runs_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {run.name}")
        print(f"{'='*60}")

        run_artifacts = [a for a in run.logged_artifacts()
                         if a.type == "model"]
        artifact = next(
            (a for a in run_artifacts if a.name.endswith(PARAMS_ARTIFACT_SUFFIX)),
            None,
        )
        if artifact is None:
            print(
                f"  No artifact ending with '{PARAMS_ARTIFACT_SUFFIX}', skipping")
            continue
        print(f"  Artifact: {artifact.name}")

        params, _ = sam_lora_util.get_params_from_artifact(artifact, device)
        model = sam_lora_util.initialize_sam_lora_with_params(
            run.config, params, device)
        model.eval()

        downsample_size = run.config.get("dataset_downsample_size", None)
        if isinstance(downsample_size, list):
            downsample_size = tuple(downsample_size)
        elif isinstance(downsample_size, int) and downsample_size != 0:
            downsample_size = (downsample_size, downsample_size)
        elif isinstance(downsample_size, int) and downsample_size == 0:
            downsample_size = None
        print(f"  Downsample size: {downsample_size}")

        dataset_name = args.dataset if args.dataset else run.config.get(
            "dataset", "")
        test_img_dir = (Path(DATASETS_DIR) / dataset_name /
                        "test" / HIGHRES_IMG_PATCHES_DIR_NAME)
        test_mask_dir = (Path(DATASETS_DIR) / dataset_name /
                         "test" / HIGHRES_MASK_PATCHES_DIR_NAME)

        print(
            f"  Computing patch-level metrics and persistence diagrams (dataset={dataset_name}, batch_size={args.batch_size})")

        (dice, iou, clDice, tprec, tsens), \
            (pp_dice, pp_iou, pp_clDice, pp_tprec, pp_tsens), \
            persistence_rows = \
            _collect_combined(model, test_img_dir, test_mask_dir,
                              downsample_size, device, args.batch_size, run.name)

        all_persistence_rows.extend(persistence_rows)
        metrics_results[run.name] = {
            "dataset": run.config.get("dataset", ""),
            "Dice": dice if dice else float("nan"),
            "IoU": iou if iou else float("nan"),
            "clDice": clDice if clDice else float("nan"),
            "tprec": tprec if tprec else float("nan"),
            "tsens": tsens if tsens else float("nan"),
            "Dice Postprocessed": pp_dice if pp_dice else float("nan"),
            "IoU Postprocessed": pp_iou if pp_iou else float("nan"),
            "clDice Postprocessed": pp_clDice if pp_clDice else float("nan"),
            "tprec Postprocessed": pp_tprec if pp_tprec else float("nan"),
            "tsens Postprocessed": pp_tsens if pp_tsens else float("nan"),
        }

        del model, params
        torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # metrics CSV
    df_new_metrics = pd.DataFrame(metrics_results).T
    df_new_metrics.index.name = "Model"
    if not df_prev_metrics.empty:
        df_new_metrics = pd.concat([df_prev_metrics, df_new_metrics])
        df_new_metrics = df_new_metrics[~df_new_metrics.index.duplicated(
            keep="first")]
    elif df_new_metrics.empty:
        df_new_metrics = df_prev_metrics

    metrics_path = _EVAL_DIR / f"metrics_{timestamp}.csv"
    df_new_metrics.to_csv(metrics_path)
    print(f"\nSaved metrics to:      {metrics_path}")

    # Persistence diagram CSV
    df_new_persistence = pd.DataFrame(
        all_persistence_rows) if all_persistence_rows else pd.DataFrame()
    if not df_prev_persistence.empty and not df_new_persistence.empty:
        df_new_persistence = pd.concat(
            [df_prev_persistence, df_new_persistence], ignore_index=True)
    elif df_new_persistence.empty:
        df_new_persistence = df_prev_persistence

    if not df_new_persistence.empty:
        persistence_path = _EVAL_DIR / f"persistence_{timestamp}.csv"
        df_new_persistence.to_csv(persistence_path, index=False)
        print(f"Saved persistence diagrams to: {persistence_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
