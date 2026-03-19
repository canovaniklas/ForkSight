"""Evaluate pre-computed segmentation patch predictions on junction detection.

Reads binary prediction patches saved by the model-specific inference scripts
(infer_patches_junction_sam.py / infer_patches_junction_nnunet.py) and runs
postprocessing + junction detection + GT matching + metric computation on them.

No model or training-framework libraries are required; this script can be run
in any environment that has the ForkSight core dependencies.

Expected layout under JUNCTION_PRED_DIR:
  <JUNCTION_PRED_DIR>/
    <safe_model_key>/
      metadata.json          {"model_key": str, "dataset": str}
      <image_stem>_patch_00.png
      ...
      <image_stem>_patch_15.png

Required environment variables (loaded via load_forksight_env):
  JUNCTION_DETECTION_DATASET_DIR   root of the junction detection dataset
  JUNCTION_PRED_DIR                directory with per-model prediction subdirs
  JUNCTION_MATCHING_THRESHOLD      max pixel distance for GT↔pred match
                                   (optional, default 75)

Outputs (written to EVALUATION_OUTPUT_DIR/junction_detection/<timestamp>/):
  predictions_<model>.csv   one row per detected junction per image
  metrics.csv               per-model aggregate metrics
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image

import Environment.env_utils as env_utils
from Segmentation.PostProcessing.segmentation_postprocessing import (
    postprocess_segmentation_masks,
)
from JunctionDetection.SkeletonizeDetect.segmentation_junction_detection import (
    detect_junctions_in_segmentation_mask,
)
from Evaluation.pipeline_evaluation_shared import (
    load_full_image_as_patches,
    plot_images_masks_junctions,
    PATCH_SIZE,
    GRID_SIZE,
)

_JUNCTION_TYPE_3_WAY = "3-way"
_JUNCTION_TYPE_4_WAY = "4-way"
_N_PATCHES = GRID_SIZE[0] * GRID_SIZE[1]


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
    Images with only Negative labels are included as empty lists.
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
    pred_types    : list of N strings, '3-way' or '4-way'.
    gt_annotations: list of {x, y, type} dicts for the GT junctions.
    threshold     : maximum pixel distance for a valid match.

    Returns
    -------
    pred_rows      : list of dicts (one per prediction)
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

    fn_annotations = [a for gt_idx, a in enumerate(gt_annotations)
                      if not gt_matched[gt_idx]]
    return pred_rows, fn_annotations


def _compute_metrics(
    all_pred_rows: list[dict],
    all_fn_annotations: list[dict],
) -> dict:
    """Compute aggregate junction detection metrics across all images."""
    def _prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    metrics: dict = {}

    tp_loc = sum(1 for r in all_pred_rows if r["is_tp"])
    fp_loc = sum(1 for r in all_pred_rows if r["is_fp"])
    fn_loc = len(all_fn_annotations)
    prec_loc, rec_loc, f1_loc = _prf(tp_loc, fp_loc, fn_loc)
    metrics.update({
        "tp_loc": tp_loc, "fp_loc": fp_loc, "fn_loc": fn_loc,
        "precision_loc": prec_loc, "recall_loc": rec_loc, "f1_loc": f1_loc,
    })

    matched_rows = [r for r in all_pred_rows if r["is_tp"]]
    type_correct = sum(
        1 for r in matched_rows if r["pred_type"] == r["matched_gt_type"])
    type_incorrect = tp_loc - type_correct
    type_accuracy = type_correct / tp_loc if tp_loc > 0 else 0.0

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

    prec_3way, rec_3way, f1_3way = _prf(
        cm_3way_3way, cm_4way_3way, cm_3way_4way)
    prec_4way, rec_4way, f1_4way = _prf(
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


def _load_pred_patches(model_pred_dir: Path, image_stem: str) -> torch.Tensor:
    """Load _N_PATCHES patch PNGs for one full image as a (N, 1, H, W) tensor."""
    patches = []
    for idx in range(_N_PATCHES):
        patch_path = model_pred_dir / f"{image_stem}_patch_{idx:02d}.png"
        arr = np.array(Image.open(patch_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask = torch.from_numpy((arr > 0).astype(np.float32)).unsqueeze(0)
        patches.append(mask)
    return torch.stack(patches)


def _process_image(
    pred_mask_patches: torch.Tensor,
    gt_annotations: list[dict],
    matching_threshold: float,
) -> tuple:
    """Run postprocessing + junction detection + GT matching for one full image.

    Parameters
    ----------
    pred_mask_patches : (N, 1, H, W) binary float32 tensor in row-major patch order.

    Returns
    -------
    raw_pred_rows, raw_fn_annots, pp_pred_rows, pp_fn_annots,
    pp_stitched, pp_coords_3way, pp_coords_4way
    """
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
        pred_types = ([_JUNCTION_TYPE_3_WAY] * len(coords_3way)
                      + [_JUNCTION_TYPE_4_WAY] * len(coords_4way))
        pred_rows, fn_annots = _match_predictions_to_gt(
            pred_coords, pred_types, gt_annotations, matching_threshold,
        )
        for r in pred_rows:
            r["source"] = source
        return pred_rows, fn_annots, coords_3way, coords_4way

    raw_pred_rows, raw_fn_annots, _, _ = _detect_and_match(raw_stitched, "raw")
    pp_pred_rows, pp_fn_annots, pp_coords_3way, pp_coords_4way = _detect_and_match(
        pp_stitched, "pp")

    return (raw_pred_rows, raw_fn_annots, pp_pred_rows, pp_fn_annots,
            pp_stitched, pp_coords_3way, pp_coords_4way)


def _save_junction_detection_plot(
    full_img: torch.Tensor,
    pp_stitched: torch.Tensor,
    coords_3way: np.ndarray,
    coords_4way: np.ndarray,
    gt_annotations: list[dict],
    plot_path: Path,
    title: str = "",
) -> None:
    """Save a full-image plot: stitched image + pp mask overlay + junction markers.

    Predicted 3-way: lime open circles.  Predicted 4-way: orange open circles.
    GT 3-way: lime Xs.  GT 4-way: orange Xs.
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    if title:
        ax.set_title(title, fontsize=10)

    plot_images_masks_junctions(
        full_img,
        predicted_mask=pp_stitched.numpy(),
        groundtruth_mask=None,
        junction_coords_3way=coords_3way if len(coords_3way) > 0 else None,
        junction_coords_4way=coords_4way if len(coords_4way) > 0 else None,
        ax=ax,
        plot_grid=False,
    )

    gt_3way = np.array([[a["x"], a["y"]] for a in gt_annotations
                        if a["type"] == _JUNCTION_TYPE_3_WAY])
    gt_4way = np.array([[a["x"], a["y"]] for a in gt_annotations
                        if a["type"] == _JUNCTION_TYPE_4_WAY])
    if len(gt_3way) > 0:
        ax.plot(gt_3way[:, 0], gt_3way[:, 1], "x",
                color="lime", markersize=8, markeredgewidth=1, label="GT 3-way")
    if len(gt_4way) > 0:
        ax.plot(gt_4way[:, 0], gt_4way[:, 1], "x",
                color="orange", markersize=8, markeredgewidth=1, label="GT 4-way")

    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def _evaluate_model(
    model_key: str,
    model_dataset: str,
    model_pred_dir: Path,
    test_image_paths: list[Path],
    gt_by_image: dict[str, list[dict]],
    matching_threshold: float,
    out_dir: Path,
    is_test: bool = False,
    plot_dir: Path | None = None,
) -> dict:
    """Run the full evaluation loop for one model and return a metrics row dict."""
    raw_pred_rows_all: list[dict] = []
    raw_fn_all: list[dict] = []
    pp_pred_rows_all: list[dict] = []
    pp_fn_all: list[dict] = []
    pred_csv_rows: list[dict] = []

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    for img_path in test_image_paths:
        stem = img_path.stem
        gt_annotations = gt_by_image.get(stem, None)
        if gt_annotations is None:
            raise ValueError(
                f"No GT annotations found for image stem '{stem}' in CSV.")

        pred_mask_patches = _load_pred_patches(model_pred_dir, stem)

        raw_preds, raw_fns, pp_preds, pp_fns, \
            pp_stitched, pp_coords_3way, pp_coords_4way = _process_image(
                pred_mask_patches, gt_annotations, matching_threshold,
            )

        raw_pred_rows_all.extend(raw_preds)
        raw_fn_all.extend(raw_fns)
        pp_pred_rows_all.extend(pp_preds)
        pp_fn_all.extend(pp_fns)

        for r in raw_preds + pp_preds:
            pred_csv_rows.append({"image": stem, **r})

        if plot_dir is not None:
            _, full_img = load_full_image_as_patches(img_path)
            _save_junction_detection_plot(
                full_img, pp_stitched,
                pp_coords_3way, pp_coords_4way,
                gt_annotations,
                plot_dir / f"{stem}.png",
                title=f"{model_key} — {stem}",
            )

        if is_test:
            break

    safe_name = model_key.replace("/", "_")
    pred_df = pd.DataFrame(pred_csv_rows)
    pred_path = out_dir / f"predictions_{safe_name}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Saved predictions as {pred_path}")

    raw_metrics = _compute_metrics(raw_pred_rows_all, raw_fn_all)
    pp_metrics = _compute_metrics(pp_pred_rows_all, pp_fn_all)

    print(f"\n  [raw]  loc P={raw_metrics['precision_loc']:.3f} "
          f"R={raw_metrics['recall_loc']:.3f} F1={raw_metrics['f1_loc']:.3f} "
          f"| type acc={raw_metrics['type_accuracy']:.3f}")
    print(f"  [pp]   loc P={pp_metrics['precision_loc']:.3f} "
          f"R={pp_metrics['recall_loc']:.3f} F1={pp_metrics['f1_loc']:.3f} "
          f"| type acc={pp_metrics['type_accuracy']:.3f}")

    row: dict = {"model": model_key, "dataset": model_dataset}
    for k, v in raw_metrics.items():
        row[f"raw_{k}"] = v
    for k, v in pp_metrics.items():
        row[f"pp_{k}"] = v
    return row


def _load_previous_metrics(base_dir: Path) -> pd.DataFrame:
    """Return the most recent metrics.csv across all timestamped sub-dirs, or
    an empty DataFrame if none exists."""
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
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-evaluate all models, ignoring cached results")
    parser.add_argument("--is-test", action="store_true",
                        help="Break after computing metrics for the first image")
    parser.add_argument("--plot", action="store_true",
                        help="Save stitched-image plots with junction markers per model")
    args = parser.parse_args()

    env_utils.load_forksight_env()

    EVALUATION_OUTPUT_DIR = os.getenv("EVALUATION_OUTPUT_DIR")
    JUNCTION_DETECTION_DATASET_DIR = os.getenv(
        "JUNCTION_DETECTION_DATASET_DIR")
    JUNCTION_PRED_DIR = os.getenv("JUNCTION_PRED_DIR")
    JUNCTION_MATCHING_THRESHOLD = env_utils.load_as(
        "JUNCTION_MATCHING_THRESHOLD", float, 75.0)

    if EVALUATION_OUTPUT_DIR is None:
        raise ValueError(
            "EVALUATION_OUTPUT_DIR environment variable must be set.")
    if JUNCTION_DETECTION_DATASET_DIR is None:
        raise ValueError(
            "JUNCTION_DETECTION_DATASET_DIR environment variable must be set.")
    if JUNCTION_PRED_DIR is None:
        raise ValueError("JUNCTION_PRED_DIR environment variable must be set.")

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

    pred_base = Path(JUNCTION_PRED_DIR)
    model_dirs = sorted(
        d for d in pred_base.iterdir()
        if d.is_dir() and (d / "metadata.json").is_file()
    )
    if not model_dirs:
        raise FileNotFoundError(
            f"No model prediction directories (with metadata.json) found in "
            f"{pred_base}")

    print(f"Found {len(test_image_paths)} test image(s), "
          f"{len(model_dirs)} model prediction dir(s).")
    print(f"Matching threshold: {JUNCTION_MATCHING_THRESHOLD} px")

    out_base = Path(EVALUATION_OUTPUT_DIR) / "junction_detection"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    _PLOT_DIR = out_dir / "plots" if args.plot else None

    df_prev = (_load_previous_metrics(out_base)
               if not args.force_recompute else pd.DataFrame())
    computed_models = set(df_prev.index) if not df_prev.empty else set()

    new_metrics_rows: list[dict] = []

    for model_dir in model_dirs:
        with open(model_dir / "metadata.json") as f:
            meta = json.load(f)
        model_key = meta["model_key"]
        model_dataset = meta.get("dataset", "")

        if model_key in computed_models:
            print(f"\n  Already cached: {model_key}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_key}")
        print(f"  Predictions: {model_dir}")
        print(f"{'='*60}")

        safe_name = model_key.replace("/", "_")
        row = _evaluate_model(
            model_key=model_key,
            model_dataset=model_dataset,
            model_pred_dir=model_dir,
            test_image_paths=test_image_paths,
            gt_by_image=gt_by_image,
            matching_threshold=JUNCTION_MATCHING_THRESHOLD,
            out_dir=out_dir,
            is_test=args.is_test,
            plot_dir=_PLOT_DIR / safe_name if _PLOT_DIR else None,
        )
        new_metrics_rows.append(row)

    if not new_metrics_rows:
        print("\nNo new models evaluated, all results already cached.")
        return

    df_new = pd.DataFrame(new_metrics_rows).set_index("model")
    if not df_prev.empty:
        df_new = pd.concat([df_prev, df_new])
    metrics_path = out_dir / "metrics.csv"
    df_new.to_csv(metrics_path)
    print(f"\nSaved metrics as {metrics_path}")


if __name__ == "__main__":
    main()
