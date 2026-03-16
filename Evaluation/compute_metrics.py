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


import Segmentation.SAM.sam_lora_util as sam_lora_util
import Environment.env_utils as env_utils

from Evaluation.evaluation_util import (
    collect_patch_metrics_and_betti,
    collect_patch_metrics_and_betti_from_masks,
    load_latest_metrics_csv,
    load_latest_persistence_raw_b0_csv,
    load_latest_persistence_raw_b1_csv,
    load_latest_persistence_sdt_b0_csv,
    load_latest_persistence_sdt_b1_csv,
    load_latest_persistence_distances_csv,
)

# WandB SAM runs to evaluate (run names)
# For each run, evaluates artifact with name ending with _SAM_PARAMS_ARTIFACT_SUFFIX
_SAM_MODELS_RUNS = [
    # "SAM_LoRA_Finetuning_20260219_150640",
    "SAM_LoRA_Finetuning_20260224_154858"]
_SAM_PARAMS_ARTIFACT_SUFFIX = "_params_minloss:v0"

# nnUNet evaluations: list of (dataset_name, trainer_class) tuples
# For each pair, every trainer sub-directory under
# NNUNET_RESULTS_DIR/<dataset>/ whose name starts with trainer_class is evaluated
_NNUNET_EVALUATIONS: list[tuple[str, str]] = [
    ("Dataset001_Segmentation_v1", "nnUNetTrainerWandb__nnUNetPlans__2d"),
    # ("Dataset001_Segmentation_v1", "nnUNetTrainerClDiceLoss"),
]


def _collect_combined(model, test_img_dir, test_mask_dir, downsample_size, device, batch_size, run_name, save_pd_dir=None):
    """Run patch-level inference once per batch, returning segmentation metrics
    and per-patch persistence diagram rows.
    """
    dataset = sam_lora_util.SegmentationDataset(
        test_img_dir, test_mask_dir, downsample_size=downsample_size, return_img_name=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return collect_patch_metrics_and_betti(model, loader, device, run_name, save_pd_dir=save_pd_dir)


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
    parser.add_argument("--no-sam", action="store_true",
                        help="Skip SAM model evaluation")
    parser.add_argument("--no-nnunet", action="store_true",
                        help="Skip nnUNet evaluation")
    args = parser.parse_args()

    env_utils.load_forksight_env()

    SEED = env_utils.load_as("SEED", int, 42)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DATASETS_DIR = os.getenv("DATASETS_DIR")
    EVALUATION_OUTPUT_DIR = os.getenv("EVALUATION_OUTPUT_DIR")
    HIGHRES_IMG_PATCHES_DIR_NAME = os.getenv(
        "HIGHRES_IMG_PATCHES_DIR_NAME", "img_patches_1024")
    HIGHRES_MASK_PATCHES_DIR_NAME = os.getenv(
        "HIGHRES_MASK_PATCHES_DIR_NAME", "mask_patches_1024")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
    WANDB_SAM_PROJECT = os.getenv("WANDB_SAM_PROJECT", "ForkSight-SAM")

    NNUNET_RAW_DIR = os.getenv("NNUNET_RAW_DIR")
    NNUNET_RESULTS_DIR = os.getenv("NNUNET_RESULTS_DIR")
    NNUNET_TEST_MASK_DIR = os.getenv("NNUNET_TEST_MASK_DIR", "labelsTs")
    NNUNET_PRED_DIR = os.getenv(
        "NNUNET_PRED_DIR", "best_configuration_inference_output")

    if DATASETS_DIR is None:
        raise ValueError("DATASETS_DIR environment variable must be set.")
    if EVALUATION_OUTPUT_DIR is None:
        raise ValueError(
            "EVALUATION_OUTPUT_DIR environment variable must be set.")
    if _NNUNET_EVALUATIONS and not args.no_nnunet:
        if NNUNET_RAW_DIR is None:
            raise ValueError(
                "NNUNET_RAW_DIR must be set when NNUNET_EVALUATIONS is non-empty.")
        if NNUNET_RESULTS_DIR is None:
            raise ValueError(
                "NNUNET_RESULTS_DIR must be set when NNUNET_EVALUATIONS is non-empty.")

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _EVAL_DIR = Path(EVALUATION_OUTPUT_DIR) / timestamp
    _CSV_DIR = _EVAL_DIR / "csv"
    _CSV_DIR.mkdir(parents=True, exist_ok=True)

    df_prev_metrics = load_latest_metrics_csv(
        _CSV_DIR) if not args.force_recompute else pd.DataFrame()
    df_prev_raw_b0 = load_latest_persistence_raw_b0_csv(
        _CSV_DIR) if not args.force_recompute else pd.DataFrame()
    df_prev_raw_b1 = load_latest_persistence_raw_b1_csv(
        _CSV_DIR) if not args.force_recompute else pd.DataFrame()
    df_prev_sdt_b0 = load_latest_persistence_sdt_b0_csv(
        _CSV_DIR) if not args.force_recompute else pd.DataFrame()
    df_prev_sdt_b1 = load_latest_persistence_sdt_b1_csv(
        _CSV_DIR) if not args.force_recompute else pd.DataFrame()
    df_prev_dist = load_latest_persistence_distances_csv(
        _CSV_DIR) if not args.force_recompute else pd.DataFrame()
    computed_models = set(
        df_prev_metrics.index) if not df_prev_metrics.empty else set()

    metrics_results = {}
    all_raw_b0_rows, all_raw_b1_rows, all_sdt_b0_rows, all_sdt_b1_rows = [], [], [], []
    all_dist_rows = []

    def _record_results(
        model_key: str,
        dataset: str,
        raw_metrics: tuple,
        pp_metrics: tuple,
        raw_b0, raw_b1, sdt_b0, sdt_b1,
        dist_rows: list,
        pd_distances: tuple,
    ):
        dice, iou, clDice, tprec, tsens = raw_metrics
        pp_dice, pp_iou, pp_clDice, pp_tprec, pp_tsens = pp_metrics
        (raw_b0_wd, raw_b0_bn, raw_b1_wd, raw_b1_bn,
         sdt_b0_wd, sdt_b0_bn, sdt_b1_wd, sdt_b1_bn) = pd_distances
        all_raw_b0_rows.extend(raw_b0)
        all_raw_b1_rows.extend(raw_b1)
        all_sdt_b0_rows.extend(sdt_b0)
        all_sdt_b1_rows.extend(sdt_b1)
        all_dist_rows.extend(dist_rows)
        metrics_results[model_key] = {
            "dataset": dataset,
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
            "Wasserstein B0 Raw": raw_b0_wd,
            "Bottleneck B0 Raw": raw_b0_bn,
            "Wasserstein B1 Raw": raw_b1_wd,
            "Bottleneck B1 Raw": raw_b1_bn,
            "Wasserstein B0 SDT": sdt_b0_wd,
            "Bottleneck B0 SDT": sdt_b0_bn,
            "Wasserstein B1 SDT": sdt_b1_wd,
            "Bottleneck B1 SDT": sdt_b1_bn,
        }

    # SAM model evaluation                                                 #
    if not args.no_sam:
        api = wandb.Api()
        all_runs = list(api.runs(f"{WANDB_ENTITY}/{WANDB_SAM_PROJECT}"))
        runs_to_eval = [
            r for r in all_runs
            if sam_lora_util.EVALUATED_TAG in r.tags
            and r.state == "finished"
            and r.name in _SAM_MODELS_RUNS
            and r.name not in computed_models
        ]

        if not runs_to_eval:
            print("No new SAM models to evaluate, all results already cached.")
        else:
            print(f"Will evaluate {len(runs_to_eval)} SAM model(s): "
                  + ", ".join(r.name for r in runs_to_eval))

        for run in runs_to_eval:
            print(f"\n{'='*60}")
            print(f"Evaluating SAM: {run.name}")
            print(f"{'='*60}")

            run_artifacts = [a for a in run.logged_artifacts()
                             if a.type == "model"]
            artifact = next(
                (a for a in run_artifacts if a.name.endswith(
                    _SAM_PARAMS_ARTIFACT_SUFFIX)),
                None,
            )
            if artifact is None:
                print(
                    f"  No artifact ending with '{_SAM_PARAMS_ARTIFACT_SUFFIX}', skipping")
                continue
            print(f"  Artifact: {artifact.name}")

            params, _ = sam_lora_util.get_params_from_artifact(
                artifact, device)
            model = sam_lora_util.initialize_sam_lora_with_params(
                run.config, params, device)
            model.eval()

            downsample_size = run.config.get("dataset_downsample_size", None)
            if isinstance(downsample_size, list) and len(downsample_size) == 2 and all(isinstance(x, int) and x > 0 for x in downsample_size):
                downsample_size = tuple(downsample_size)
            elif isinstance(downsample_size, int) and downsample_size > 0:
                downsample_size = (downsample_size, downsample_size)
            else:
                downsample_size = None
            print(f"  Downsample size: {downsample_size}")

            dataset_name = args.dataset if args.dataset else run.config.get(
                "dataset", "")
            test_img_dir = (Path(DATASETS_DIR) / dataset_name /
                            "test" / HIGHRES_IMG_PATCHES_DIR_NAME)
            test_mask_dir = (Path(DATASETS_DIR) / dataset_name /
                             "test" / HIGHRES_MASK_PATCHES_DIR_NAME)

            print(
                f"\n  Computing patch-level metrics and persistence diagrams "
                f"(dataset={dataset_name}, batch_size={args.batch_size})")

            raw_metrics, pp_metrics, \
                raw_b0_rows, raw_b1_rows, sdt_b0_rows, sdt_b1_rows, \
                pd_distance_rows, pd_distances = \
                _collect_combined(model, test_img_dir, test_mask_dir,
                                  downsample_size, device, args.batch_size, run.name,
                                  save_pd_dir=_EVAL_DIR / f"persistence_{run.name}")

            _record_results(
                run.name, run.config.get("dataset", ""),
                raw_metrics, pp_metrics,
                raw_b0_rows, raw_b1_rows, sdt_b0_rows, sdt_b1_rows,
                pd_distance_rows, pd_distances,
            )

            del model, params
            torch.cuda.empty_cache()

    # nnUNet evaluation
    if not args.no_nnunet and _NNUNET_EVALUATIONS:
        for dataset_name, trainer_class in _NNUNET_EVALUATIONS:
            pred_dir = Path(NNUNET_RESULTS_DIR) / \
                dataset_name / trainer_class / NNUNET_PRED_DIR
            if not pred_dir:
                print(
                    f"\n[WARN] No predictions directory '{str(pred_dir)}' found, skipping.")
                continue

            gt_mask_dir = Path(NNUNET_RAW_DIR) / \
                dataset_name / NNUNET_TEST_MASK_DIR
            if not gt_mask_dir.is_dir():
                print(
                    f"\n[WARN] nnUNet GT mask dir not found: {gt_mask_dir}, skipping.")
                continue

            model_key = f"nnunet/{dataset_name}/{trainer_class}"
            if model_key in computed_models:
                print(f"\n  Already cached: {model_key}")
                continue

            print(f"\n{'='*60}")
            print(f"Evaluating nnUNet: {model_key}")
            print(f"  GT masks  : {gt_mask_dir}")
            print(f"  Predictions: {pred_dir}")
            print(f"{'='*60}")

            raw_metrics, pp_metrics, \
                raw_b0_rows, raw_b1_rows, sdt_b0_rows, sdt_b1_rows, \
                pd_distance_rows, pd_distances = \
                collect_patch_metrics_and_betti_from_masks(
                    gt_mask_dir, pred_dir, model_key,
                    save_pd_dir=_EVAL_DIR /
                    f"persistence_{dataset_name}_{trainer_class}",
                )

            _record_results(
                model_key, dataset_name,
                raw_metrics, pp_metrics,
                raw_b0_rows, raw_b1_rows, sdt_b0_rows, sdt_b1_rows,
                pd_distance_rows, pd_distances,
            )

    # metrics CSV
    df_new_metrics = pd.DataFrame(metrics_results).T
    df_new_metrics.index.name = "Model"
    if not df_prev_metrics.empty:
        df_new_metrics = pd.concat([df_prev_metrics, df_new_metrics])
        df_new_metrics = df_new_metrics[~df_new_metrics.index.duplicated(
            keep="first")]
    elif df_new_metrics.empty:
        df_new_metrics = df_prev_metrics

    metrics_path = _CSV_DIR / "metrics.csv"
    df_new_metrics.to_csv(metrics_path)
    print(f"\n  Saved metrics to:\t\t\t{metrics_path}")

    # Persistence diagram CSVs (one per Betti number × raw/SDT)
    _persistence_specs = [
        ("persistence_raw_b0", all_raw_b0_rows, df_prev_raw_b0),
        ("persistence_raw_b1", all_raw_b1_rows, df_prev_raw_b1),
        ("persistence_sdt_b0", all_sdt_b0_rows, df_prev_sdt_b0),
        ("persistence_sdt_b1", all_sdt_b1_rows, df_prev_sdt_b1),
        ("persistence_distances", all_dist_rows, df_prev_dist),
    ]
    for stem, new_rows, df_prev in _persistence_specs:
        df_new = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
        if not df_prev.empty and not df_new.empty:
            df_new = pd.concat([df_prev, df_new], ignore_index=True)
        elif df_new.empty:
            df_new = df_prev
        if not df_new.empty:
            path = _CSV_DIR / f"{stem}.csv"
            df_new.to_csv(path, index=False)
            print(f"  Saved persistence data to:\t{path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
