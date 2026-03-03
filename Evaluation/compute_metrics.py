"""
For a list of WandB model runs, compute per-patch segmentation metrics and
full-image Betti-number curves on the test set, then write results to CSV.

Outputs (written next to this script in Evaluation/):
  metrics_<YYYYMMDD_HHMMSS>.csv
      Per-model aggregate metrics (Dice, IoU, clDice, tprec, tsens – raw and
      post-processed) plus Betti-0 F-score and Betti-1 MAE at threshold 0.5.
      Previously computed models are loaded and merged in so re-runs are cheap.

  betti_<YYYYMMDD_HHMMSS>.csv
      One row per (model, image, type) where type is "predicted" or
      "groundtruth".  Columns b0_t<T> / b1_t<T> give the Betti numbers at
      every threshold T in BETTI_THRESHOLDS.

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

import Segmentation.PostProcessing.segmentation_postprocessing as postproc
import Segmentation.SAM.sam_lora_util as sam_lora_util
import Segmentation.Util.env_utils as env_utils
import Segmentation.Util.dataset_util as dataset_util

from Evaluation.evaluation_util import (
    collect_patch_metrics,
    compute_betti_aggregate_metrics,
    get_betti_at_thresholds,
    load_latest_betti_csv,
    load_latest_metrics_csv,
    load_transform_image,
)

MODELS_RUNS = [
    "SAM_LoRA_Finetuning_20260212_155155",
    "SAM_LoRA_Finetuning_20260212_155412",
    "SAM_LoRA_Finetuning_20260212_155506",
    "SAM_LoRA_Finetuning_20260212_155528",
    "SAM_LoRA_Finetuning_20260212_155556",
    "SAM_LoRA_Finetuning_20260212_161635",
    "SAM_LoRA_Finetuning_20260218_112642-clDice",
    "SAM_LoRA_Finetuning_20260218_112642-Skel",
    "SAM_LoRA_Finetuning_20260219_093322",
    "SAM_LoRA_Finetuning_20260219_131434",
    "SAM_LoRA_Finetuning_20260219_132212",
    "SAM_LoRA_Finetuning_20260219_150640",
]

BETTI_THRESHOLDS = np.linspace(0.0, 1.0, 51)

PARAMS_ARTIFACT_SUFFIX = "_params_minloss:v0"
PATCH_SIZE = (1024, 1024)
GRID_SIZE = (4, 4)

LOSS_CONFIG_KEYS = [
    "bce_loss_weight",
    "cl_dice_loss_weight",
    "dice_loss_weight",
    "focal_loss_weight",
    "junction_heatmap_weight_scale",
    "junction_patch_weight",
    "skeleton_recall_loss_weight",
    "topological_loss_weight",
]

METRIC_COLS = [
    "Dice", "IoU", "clDice", "tprec", "tsens",
    "Dice Postprocessed", "IoU Postprocessed",
    "clDice Postprocessed", "tprec Postprocessed", "tsens Postprocessed",
    "Betti0 F-Score@0.5", "Betti1 MAE@0.5",
]


def _betti_col(prefix: str, threshold: float) -> str:
    return f"{prefix}_t{threshold:.2f}"


def _collect_patch_metrics(model, test_img_dir, test_mask_dir, downsample_size, device, batch_size=8):
    """Run patch-level inference in batches using SegmentationDataset + DataLoader.
    """
    dataset = sam_lora_util.SegmentationDataset(
        test_img_dir, test_mask_dir, downsample_size=downsample_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return collect_patch_metrics(model, loader, device)


def _collect_full_image_betti(
    model,
    full_img_names,
    test_img_dir,
    raw_data_dir,
    highres_mask_dir_name,
    downsample_size,
    device,
    run_name,
):
    """Run full-image inference for every base image, compute Betti curves,
    and return:
      - all_b0_curves, all_b1_curves  (predicted, per image)
      - gt_b0_list, gt_b1_list        (groundtruth scalar per image)
      - betti_rows                    (list of dicts ready for DataFrame)
    """
    all_b0_curves, all_b1_curves = [], []
    gt_b0_list, gt_b1_list = [], []
    betti_rows = []

    for img_name in full_img_names:
        patch_paths = sorted(
            p for p in test_img_dir.glob("*.png")
            if p.name.startswith(f"{img_name}_patch")
        )
        gt_full_path = Path(raw_data_dir) / \
            highres_mask_dir_name / f"{img_name}.png"

        if not patch_paths:
            print(f"  [warn] no patches found for {img_name}, skipping Betti.")
            continue
        if not gt_full_path.exists():
            print(
                f"  [warn] full groundtruth mask not found: {gt_full_path}, skipping Betti.")
            continue

        patch_probs = []
        for patch_path in patch_paths:
            img = load_transform_image(
                patch_path, downsample_size=downsample_size)
            input_list = sam_lora_util.get_batched_input_list(
                img.unsqueeze(0).to(device))
            with torch.no_grad():
                out = model(batched_input=input_list, multimask_output=False)

            low_res_logits = out[0]["low_res_logits"]
            resized_logits = model.sam_model.postprocess_masks(
                low_res_logits,
                input_size=(1024, 1024),
                original_size=(1024, 1024),
            )
            patch_probs.append(torch.sigmoid(resized_logits.squeeze(0)))

        patch_probs_t = torch.stack(patch_probs, dim=0).detach().cpu()
        stitched = postproc.stitch_mask_tiles(
            patch_probs_t,
            grid_size=GRID_SIZE,
            original_input_patch_img_size=PATCH_SIZE,
            as_uint=False,
        )
        probs_np = stitched.squeeze(0).detach().cpu().numpy()

        # groundtruth Betti numbers
        gt_mask_full = load_transform_image(
            gt_full_path, is_mask=True, is_full_image=True)
        gt_mask_np = gt_mask_full.squeeze(0).cpu().numpy()
        gt_b0_vals, gt_b1_vals = get_betti_at_thresholds(gt_mask_np, [1.0])
        gt_b0 = gt_b0_vals[0]
        gt_b1 = gt_b1_vals[0]
        gt_b0_list.append(gt_b0)
        gt_b1_list.append(gt_b1)

        # predicted Betti numbers at all thresholds
        pred_b0, pred_b1 = get_betti_at_thresholds(probs_np, BETTI_THRESHOLDS)
        all_b0_curves.append(pred_b0)
        all_b1_curves.append(pred_b1)

        # build CSV rows
        row_pred = {"model": run_name, "image": img_name, "type": "predicted"}
        row_gt = {"model": run_name, "image": img_name, "type": "groundtruth"}
        for ti, t in enumerate(BETTI_THRESHOLDS):
            row_pred[_betti_col("b0", t)] = pred_b0[ti]
            row_pred[_betti_col("b1", t)] = pred_b1[ti]

            # ground truth Betti numbers only at threshold=1.0 since it's binary,
            # but keep same column structure for ease of analysis
            row_gt[_betti_col("b0", t)] = gt_b0
            row_gt[_betti_col("b1", t)] = gt_b1

        betti_rows.append(row_pred)
        betti_rows.append(row_gt)

        t50_idx = int(np.argmin(np.abs(BETTI_THRESHOLDS - 0.5)))
        print(f"    {img_name}: GT B0={gt_b0}, GT B1={gt_b1} | "
              f"pred B0@0.5={pred_b0[t50_idx]}, pred B1@0.5={pred_b1[t50_idx]}")

    return all_b0_curves, all_b1_curves, gt_b0_list, gt_b1_list, betti_rows


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

    RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
    DATASETS_DIR = os.getenv("DATASETS_DIR")
    HIGHRES_IMG_PATCHES_DIR_NAME = os.getenv(
        "HIGHRES_IMG_PATCHES_DIR_NAME", "img_patches_1024")
    HIGHRES_MASK_PATCHES_DIR_NAME = os.getenv(
        "HIGHRES_MASK_PATCHES_DIR_NAME", "mask_patches_1024")
    HIGHRES_MASK_DIR_NAME = os.getenv("HIGHRES_MASK_DIR_NAME", "masks_4096")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ForkSight-SAM")

    if RAW_DATA_DIR is None or DATASETS_DIR is None:
        raise ValueError(
            "RAW_DATA_DIR and DATASETS_DIR environment variables must be set.")

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_prev_metrics = load_latest_metrics_csv(
        _EVAL_DIR) if not args.force_recompute else pd.DataFrame()
    df_prev_betti = load_latest_betti_csv(
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
    all_betti_rows = []

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
        elif isinstance(downsample_size, int) and downsample_size > 0:
            downsample_size = (downsample_size, downsample_size)
        print(f"  Downsample size: {downsample_size}")

        dataset_name = args.dataset if args.dataset else run.config.get(
            "dataset", "")
        test_img_dir = (Path(DATASETS_DIR) / dataset_name /
                        "test" / HIGHRES_IMG_PATCHES_DIR_NAME)
        test_mask_dir = (Path(DATASETS_DIR) / dataset_name /
                         "test" / HIGHRES_MASK_PATCHES_DIR_NAME)

        print(
            f"  Computing patch-level metrics (batch_size={args.batch_size})")
        (dice_s, iou_s, clDice_s, tprec_s, tsens_s), \
            (pp_dice_s, pp_iou_s, pp_clDice_s, pp_tprec_s, pp_tsens_s) = \
            _collect_patch_metrics(model, test_img_dir, test_mask_dir,
                                   downsample_size, device, batch_size=args.batch_size)

        full_img_names = sorted(
            dataset_util.get_base_images(test_img_dir, exclude_soi_images=True)
        )
        print(
            f"  Computing Betti numbers for {len(full_img_names)} full image")
        all_b0_curves, all_b1_curves, gt_b0_list, gt_b1_list, betti_rows = \
            _collect_full_image_betti(
                model, full_img_names, test_img_dir,
                RAW_DATA_DIR, HIGHRES_MASK_DIR_NAME,
                downsample_size, device, run.name,
            )
        all_betti_rows.extend(betti_rows)

        if all_b0_curves:
            fscore_per_t, mae_b1_per_t = compute_betti_aggregate_metrics(
                all_b0_curves, all_b1_curves, gt_b0_list, gt_b1_list, BETTI_THRESHOLDS,
            )
            # Summary scalar: use threshold=0.5 (index 5 with 11 evenly-spaced values)
            t50_idx = int(np.argmin(np.abs(BETTI_THRESHOLDS - 0.5)))
            betti0_fscore_summary = fscore_per_t[t50_idx]
            betti1_mae_summary = mae_b1_per_t[t50_idx]
        else:
            betti0_fscore_summary = float("nan")
            betti1_mae_summary = float("nan")

        result = {
            "dataset": run.config.get("dataset", ""),
            "Dice": dice_s if dice_s else float("nan"),
            "IoU": iou_s if iou_s else float("nan"),
            "clDice": clDice_s if clDice_s else float("nan"),
            "tprec": tprec_s if tprec_s else float("nan"),
            "tsens": tsens_s if tsens_s else float("nan"),
            "Dice Postprocessed": pp_dice_s if pp_dice_s else float("nan"),
            "IoU Postprocessed": pp_iou_s if pp_iou_s else float("nan"),
            "clDice Postprocessed": pp_clDice_s if pp_clDice_s else float("nan"),
            "tprec Postprocessed": pp_tprec_s if pp_tprec_s else float("nan"),
            "tsens Postprocessed": pp_tsens_s if pp_tsens_s else float("nan"),
            "Betti0 F-Score@0.5": betti0_fscore_summary,
            "Betti1 MAE@0.5": betti1_mae_summary,
        }
        for k in LOSS_CONFIG_KEYS:
            result[k] = run.config.get(k, 0.0)

        metrics_results[run.name] = result

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

    # Betti CSV
    df_new_betti = pd.DataFrame(
        all_betti_rows) if all_betti_rows else pd.DataFrame()
    if not df_prev_betti.empty and not df_new_betti.empty:
        df_new_betti = pd.concat(
            [df_prev_betti, df_new_betti], ignore_index=True)
    elif df_new_betti.empty:
        df_new_betti = df_prev_betti

    if not df_new_betti.empty:
        betti_path = _EVAL_DIR / f"betti_{timestamp}.csv"
        df_new_betti.to_csv(betti_path, index=False)
        print(f"Saved Betti numbers to: {betti_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
