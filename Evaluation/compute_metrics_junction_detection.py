"""Evaluate SAM LoRA models on junction detection using full-size images.

For each model run, predicts segmentation masks for full-size (4096×4096)
images by splitting them into 16 1024×1024 patches, stitching predictions,
and running postprocessing + junction detection on the stitched masks.
Detected junctions are matched against ground-truth point annotations and
per-image predictions and aggregate metrics are written to CSV.

Required environment variables (loaded via load_forksight_env):
  JUNCTION_DETECTION_DATASET_DIR   root of the junction detection dataset
  JUNCTION_MATCHING_THRESHOLD      max pixel distance for GT↔pred match
                                   (optional, default 75)

Outputs (written to EVALUATION_OUTPUT_DIR/junction_detection/<timestamp>/):
  predictions_<model>.csv   one row per detected junction per image:
      image, source (raw|pp), x, y, pred_type, matched_gt_x, matched_gt_y,
      matched_gt_type, distance, is_tp, is_fp
  metrics.csv               per-model aggregate metrics

Usage:
  python compute_junction_detection_metrics.py [--device N] [--batch-size N]
                                               [--force-recompute]
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb

import Environment.env_utils as env_utils
from Segmentation.SAM.sam_lora_util import (
    get_params_from_artifact,
    initialize_sam_lora_with_params,
)
from Segmentation.PostProcessing.segmentation_postprocessing import (
    postprocess_segmentation_masks,
)
from JunctionDetection.SkeletonizeDetect.segmentation_junction_detection import (
    detect_junctions_in_segmentation_mask,
)
from Evaluation.pipeline_evaluation_shared import (
    load_full_image_as_patches,
    predict_patches_batched,
    PATCH_SIZE,
    GRID_SIZE,
)
from Evaluation.compute_metrics_config import (
    SAM_MODELS_RUNS,
    SAM_PARAMS_ARTIFACT_SUFFIX,
)

_JUNCTION_TYPE_3_WAY = "3-way"
_JUNCTION_TYPE_4_WAY = "4-way"


def _label_to_junction_type(label: str) -> str | None:
    """Map CSV label to '3-way', '4-way', or None (Negative / no junction)."""
    if label in ("Crossing", "Reversed Fork"):
        return _JUNCTION_TYPE_4_WAY
    elif label == "Normal Fork":
        return _JUNCTION_TYPE_3_WAY
    elif label == "Negative":
        return None
    raise ValueError(f"Unknown label in CSV: '{label}'")


def _load_gt_annotations(csv_path: Path) -> dict[str, list[dict]]:
    """Load ground-truth annotations from relabeling_data.csv.

    Returns
    -------
    dict mapping image stem to list of {x, y, type} dicts.
    Images with only Negative labels are included as empty lists
    """
    df = pd.read_csv(csv_path)
    gt: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        stem = str(row["image"])
        if stem not in gt:
            gt[stem] = []
        jtype = _label_to_junction_type(str(row["label"]))
        if jtype is not None:
            gt[stem].append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "type": jtype,
            })
    return gt


def _match_predictions_to_gt(
    pred_coords: np.ndarray,
    pred_types: list[str],
    gt_annotations: list[dict],
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    """Greedy nearest-neighbour matching of predicted junctions to GT.

    Matching is purely spatial (type-agnostic): a prediction matches a GT
    junction if it is within the distance threshold, regardless of type.
    Each GT can be matched at most once; if multiple predictions are within
    the threshold of the same GT, the closest one wins and the rest become
    false positives.

    Parameters
    ----------
    pred_coords   : (N, 2) array of (x, y) predicted junction coordinates.
    pred_types    : list of N strings, '3-way' or '4-way' corresponding to pred_coords, indicating the predicted junction type.
    gt_annotations: list of {x, y, type} dicts for the GT junctions.
    threshold     : maximum pixel distance for a valid match.

    Returns
    -------
    pred_rows      : list of dicts (one per prediction) with fields:
                     x, y, pred_type, matched_gt_x, matched_gt_y,
                     matched_gt_type, distance, is_tp, is_fp
    fn_annotations : list of GT annotation dicts that were not matched.
    """
    n_pred = len(pred_coords)
    n_gt = len(gt_annotations)

    if n_pred == 0:
        return [], list(gt_annotations)

    gt_coords = np.array([[a["x"], a["y"]] for a in gt_annotations])

    # Compute all pairwise distances between predictions and GT junctions
    diff = pred_coords[:, None, :] - gt_coords[None, :, :]  # (N_pred, N_gt, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)              # (N_pred, N_gt)

    # Collect all candidate (pred, gt) pairs within the threshold,
    # sorted by distance so the greedy loop always assigns the closest first.
    # Matching is type-agnostic; type correctness is evaluated separately.
    candidate_pairs = sorted(
        (dist_matrix[pred_idx, gt_idx], pred_idx, gt_idx)
        for pred_idx in range(n_pred)
        for gt_idx in range(n_gt)
        if dist_matrix[pred_idx, gt_idx] <= threshold
    )

    # Greedy one-to-one assignment: each prediction and each GT can be matched
    # at most once; closer pairs take priority over farther ones
    pred_matched_to_gt: list[int | None] = [None] * n_pred
    gt_matched = [False] * n_gt
    for _, pred_idx, gt_idx in candidate_pairs:
        if pred_matched_to_gt[pred_idx] is None and not gt_matched[gt_idx]:
            pred_matched_to_gt[pred_idx] = gt_idx
            gt_matched[gt_idx] = True

    # Build one output row per prediction
    pred_rows: list[dict] = []
    for pred_idx, (x, y) in enumerate(pred_coords):
        matched_gt_idx = pred_matched_to_gt[pred_idx]
        if matched_gt_idx is not None:
            pred_rows.append({
                "x": float(x),
                "y": float(y),
                "pred_type": pred_types[pred_idx],
                "matched_gt_x": float(gt_coords[matched_gt_idx, 0]),
                "matched_gt_y": float(gt_coords[matched_gt_idx, 1]),
                "matched_gt_type": gt_annotations[matched_gt_idx]["type"],
                "distance": float(dist_matrix[pred_idx, matched_gt_idx]),
                "is_tp": True,
                "is_fp": False,
            })
        else:
            pred_rows.append({
                "x": float(x),
                "y": float(y),
                "pred_type": pred_types[pred_idx],
                "matched_gt_x": None,
                "matched_gt_y": None,
                "matched_gt_type": None,
                "distance": None,
                "is_tp": False,
                "is_fp": True,
            })

    fn_annotations = [a for gt_idx, a in enumerate(
        gt_annotations) if not gt_matched[gt_idx]]
    return pred_rows, fn_annotations


def _compute_metrics(
    all_pred_rows: list[dict],
    all_fn_annotations: list[dict],
) -> dict:
    """Compute aggregate junction detection metrics across all images.

    Matching is type-agnostic (spatial only), so localization and type
    classification are evaluated independently:
      - Localization: TP/FP/FN/precision/recall/F1 based on spatial matching.
      - Type classification: accuracy and per-type confusion counts computed
        only on spatially matched (TP) pairs.

    Parameters
    ----------
    all_pred_rows      : prediction rows from all images for one mask variant.
    all_fn_annotations : unmatched GT annotations (FNs) from all images.

    Returns
    -------
    Flat dict of metric name: numeric value
    """
    def _compute_precision_recall_f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    metrics: dict = {}

    # Localization metrics (type-agnostic)
    tp_loc = sum(1 for r in all_pred_rows if r["is_tp"])
    fp_loc = sum(1 for r in all_pred_rows if r["is_fp"])
    fn_loc = len(all_fn_annotations)
    prec_loc, rec_loc, f1_loc = _compute_precision_recall_f1(
        tp_loc, fp_loc, fn_loc)
    metrics.update({
        "tp_loc": tp_loc, "fp_loc": fp_loc, "fn_loc": fn_loc,
        "precision_loc": prec_loc, "recall_loc": rec_loc, "f1_loc": f1_loc,
    })

    # Type classification metrics (on spatially matched pairs only)
    matched_rows = [r for r in all_pred_rows if r["is_tp"]]
    type_correct = sum(
        1 for r in matched_rows if r["pred_type"] == r["matched_gt_type"])
    type_incorrect = tp_loc - type_correct
    type_accuracy = type_correct / tp_loc if tp_loc > 0 else 0.0

    # 2x2 confusion matrix: rows = GT type, columns = predicted type
    cm_3way_3way = sum(1 for r in matched_rows
                       if r["matched_gt_type"] == _JUNCTION_TYPE_3_WAY
                       and r["pred_type"] == _JUNCTION_TYPE_3_WAY)
    cm_3way_4way = sum(1 for r in matched_rows
                       if r["matched_gt_type"] == _JUNCTION_TYPE_3_WAY
                       and r["pred_type"] == _JUNCTION_TYPE_4_WAY)
    cm_4way_3way = sum(1 for r in matched_rows
                       if r["matched_gt_type"] == _JUNCTION_TYPE_4_WAY
                       and r["pred_type"] == _JUNCTION_TYPE_3_WAY)
    cm_4way_4way = sum(1 for r in matched_rows
                       if r["matched_gt_type"] == _JUNCTION_TYPE_4_WAY
                       and r["pred_type"] == _JUNCTION_TYPE_4_WAY)

    # Per-type precision/recall/f1 derived from confusion matrix
    prec_3way, rec_3way, f1_3way = _compute_precision_recall_f1(
        cm_3way_3way, cm_4way_3way, cm_3way_4way)
    prec_4way, rec_4way, f1_4way = _compute_precision_recall_f1(
        cm_4way_4way, cm_3way_4way, cm_4way_3way)

    metrics.update({
        "type_correct": type_correct,
        "type_incorrect": type_incorrect,
        "type_accuracy": type_accuracy,
        "cm_gt3_pred3": cm_3way_3way,
        "cm_gt3_pred4": cm_3way_4way,
        "cm_gt4_pred3": cm_4way_3way,
        "cm_gt4_pred4": cm_4way_4way,
        "type_precision_3way": prec_3way,
        "type_recall_3way": rec_3way,
        "type_f1_3way": f1_3way,
        "type_precision_4way": prec_4way,
        "type_recall_4way": rec_4way,
        "type_f1_4way": f1_4way,
    })

    return metrics


def _process_image(
    model,
    image_path: Path,
    device: torch.device,
    batch_size: int,
    gt_annotations: list[dict],
    matching_threshold: float,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Run inference + junction detection for one full-size image.

    Returns
    -------
    raw_pred_rows  : prediction rows from the *raw* stitched mask.
    raw_fn_annots  : unmatched GT annotations (FNs) for the raw mask.
    pp_pred_rows   : prediction rows from the *post-processed* stitched mask.
    pp_fn_annots   : unmatched GT annotations (FNs) for the post-processed mask.
    """
    patches, _ = load_full_image_as_patches(image_path)
    pred_mask_patches, _ = predict_patches_batched(
        model, patches, device, batch_size)

    raw_stitched, _ = postprocess_segmentation_masks(
        pred_mask_patches, grid_size=GRID_SIZE,
        original_input_patch_img_size=PATCH_SIZE,
        remove_small_objects=False,
    )
    raw_stitched = raw_stitched.detach().cpu()

    pp_stitched, _ = postprocess_segmentation_masks(
        pred_mask_patches, grid_size=GRID_SIZE,
        original_input_patch_img_size=PATCH_SIZE,
        remove_small_objects=True,
    )
    pp_stitched = pp_stitched.detach().cpu()

    def _detect_and_match(mask: torch.Tensor, source: str):
        coords_3way, coords_4way, _ = detect_junctions_in_segmentation_mask(
            mask)
        if len(coords_3way) > 0 or len(coords_4way) > 0:
            pred_coords = np.concatenate([coords_3way, coords_4way], axis=0)
        else:
            pred_coords = np.empty((0, 2))
        pred_types = [_JUNCTION_TYPE_3_WAY] * len(coords_3way) + \
            [_JUNCTION_TYPE_4_WAY] * len(coords_4way)
        pred_rows, fn_annots = _match_predictions_to_gt(
            pred_coords, pred_types, gt_annotations, matching_threshold,
        )
        for r in pred_rows:
            r["source"] = source
        return pred_rows, fn_annots

    raw_pred_rows, raw_fn_annots = _detect_and_match(raw_stitched, "raw")
    pp_pred_rows, pp_fn_annots = _detect_and_match(pp_stitched, "pp")

    return raw_pred_rows, raw_fn_annots, pp_pred_rows, pp_fn_annots


def _load_previous_metrics(base_dir: Path) -> pd.DataFrame:
    """Return the most recent metrics.csv across all timestamped sub-dirs of
    *base_dir*, or an empty DataFrame if none exists.

    Sub-directories are expected to be named YYYYMMDD_HHMMSS; the one with the
    lexicographically largest name (i.e. the most recent timestamp) is used.
    """
    candidates = [(p.parent.name, p) for p in base_dir.glob("*/metrics.csv")]
    if not candidates:
        return pd.DataFrame()
    _, latest = max(candidates, key=lambda t: t[0])
    df = pd.read_csv(latest, index_col="model")
    print(f"Loaded previous metrics from: {latest} ({len(df)} model(s))")
    return df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Patches per forward pass (default: 4)")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-evaluate all models, ignoring cached results")
    args = parser.parse_args()

    env_utils.load_forksight_env()

    EVALUATION_OUTPUT_DIR = os.getenv("EVALUATION_OUTPUT_DIR")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
    WANDB_SAM_PROJECT = os.getenv("WANDB_SAM_PROJECT", "ForkSight-SAM")
    JUNCTION_DETECTION_DATASET_DIR = os.getenv(
        "JUNCTION_DETECTION_DATASET_DIR")
    JUNCTION_MATCHING_THRESHOLD = env_utils.load_as(
        "JUNCTION_MATCHING_THRESHOLD", float, 75.0)

    if EVALUATION_OUTPUT_DIR is None:
        raise ValueError(
            "EVALUATION_OUTPUT_DIR environment variable must be set.")
    if JUNCTION_DETECTION_DATASET_DIR is None:
        raise ValueError(
            "JUNCTION_DETECTION_DATASET_DIR environment variable must be set.")

    test_dir = Path(JUNCTION_DETECTION_DATASET_DIR) / "test"
    test_images_dir = test_dir / "images"
    test_labels_csv = test_dir / "relabeling_data.csv"

    if not test_images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found: {test_images_dir}")
    if not test_labels_csv.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {test_labels_csv}")

    gt_by_image = _load_gt_annotations(test_labels_csv)
    test_image_paths = sorted(p for p in test_images_dir.glob("*.png"))
    if not test_image_paths:
        raise FileNotFoundError(f"No image files found in {test_images_dir}")

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    print(
        f"Found {len(test_image_paths)} image(s) and {len(gt_by_image)} annotated stem(s).")
    print(f"Matching threshold: {JUNCTION_MATCHING_THRESHOLD} px")

    out_base = Path(EVALUATION_OUTPUT_DIR) / "junction_detection"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load previously computed metrics to skip already-evaluated models
    df_prev = _load_previous_metrics(
        out_base) if not args.force_recompute else pd.DataFrame()
    computed_models = set(df_prev.index) if not df_prev.empty else set()

    api = wandb.Api()
    all_runs = list(api.runs(f"{WANDB_ENTITY}/{WANDB_SAM_PROJECT}"))
    runs_to_eval = [
        r for r in all_runs
        and r.state == "finished"
        and r.name in SAM_MODELS_RUNS
        and r.name not in computed_models
    ]

    if not runs_to_eval:
        print("No new SAM models to evaluate, all results already cached.")
        return

    print(f"Will evaluate {len(runs_to_eval)} SAM model(s): "
          + ", ".join(r.name for r in runs_to_eval))

    new_metrics_rows: list[dict] = []

    for run in runs_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {run.name}")
        print(f"{'='*60}")

        run_artifacts = [a for a in run.logged_artifacts()
                         if a.type == "model"]
        artifact = next(
            (a for a in run_artifacts if a.name.endswith(SAM_PARAMS_ARTIFACT_SUFFIX)),
            None,
        )
        if artifact is None:
            print(
                f"  No artifact ending with '{SAM_PARAMS_ARTIFACT_SUFFIX}', skipping.")
            continue
        print(f"  Artifact: {artifact.name}")

        params, _ = get_params_from_artifact(artifact, device)
        model = initialize_sam_lora_with_params(run.config, params, device)
        model.eval()

        raw_pred_rows_all: list[dict] = []
        raw_fn_all: list[dict] = []
        pp_pred_rows_all: list[dict] = []
        pp_fn_all: list[dict] = []
        pred_csv_rows: list[dict] = []

        for img_path in test_image_paths:
            stem = img_path.stem
            gt_annotations = gt_by_image.get(stem, None)
            if gt_annotations is None:
                raise ValueError(
                    f"No GT annotations found for image stem '{stem}' in CSV.")

            raw_preds, raw_fns, pp_preds, pp_fns = _process_image(
                model, img_path, device, args.batch_size,
                gt_annotations, JUNCTION_MATCHING_THRESHOLD,
            )

            raw_pred_rows_all.extend(raw_preds)
            raw_fn_all.extend(raw_fns)
            pp_pred_rows_all.extend(pp_preds)
            pp_fn_all.extend(pp_fns)

            for r in raw_preds + pp_preds:
                pred_csv_rows.append({"image": stem, **r})

        safe_name = run.name.replace("/", "_")
        pred_df = pd.DataFrame(pred_csv_rows)
        pred_path = out_dir / f"predictions_{safe_name}.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"\n  Saved predictions as {pred_path}")

        raw_metrics = _compute_metrics(raw_pred_rows_all, raw_fn_all)
        pp_metrics = _compute_metrics(pp_pred_rows_all, pp_fn_all)

        row: dict = {"model": run.name,
                     "dataset": run.config.get("dataset", "")}
        for k, v in raw_metrics.items():
            row[f"raw_{k}"] = v
        for k, v in pp_metrics.items():
            row[f"pp_{k}"] = v
        new_metrics_rows.append(row)

        print(f"\n  [raw]  loc P={raw_metrics['precision_loc']:.3f} "
              f"R={raw_metrics['recall_loc']:.3f} F1={raw_metrics['f1_loc']:.3f} "
              f"| type acc={raw_metrics['type_accuracy']:.3f}")
        print(f"  [pp]   loc P={pp_metrics['precision_loc']:.3f} "
              f"R={pp_metrics['recall_loc']:.3f} F1={pp_metrics['f1_loc']:.3f} "
              f"| type acc={pp_metrics['type_accuracy']:.3f}")

        del model, params
        torch.cuda.empty_cache()

    # Merge new results with any previously cached rows and save
    df_new = pd.DataFrame(new_metrics_rows).set_index("model")
    if not df_prev.empty:
        df_new = pd.concat([df_prev, df_new])
    metrics_path = out_dir / "metrics.csv"
    df_new.to_csv(metrics_path)
    print(f"\nSaved metrics as {metrics_path}")


if __name__ == "__main__":
    main()
