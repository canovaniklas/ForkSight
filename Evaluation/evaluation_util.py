import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance as gudhi_wasserstein_distance
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from Segmentation.PostProcessing.segmentation_postprocessing import remove_small_objects_from_batch
from Segmentation.PreProcessing.General.preprocessing_util import get_base_images


# CSV filename patterns
METRICS_PATTERN = re.compile(r"^metrics_(\d{8}_\d{6})\.csv$")
BETTI_PATTERN = re.compile(r"^betti_(\d{8}_\d{6})\.csv$")
PERSISTENCE_RAW_B0_PATTERN = re.compile(
    r"^persistence_raw_b0_(\d{8}_\d{6})\.csv$")
PERSISTENCE_RAW_B1_PATTERN = re.compile(
    r"^persistence_raw_b1_(\d{8}_\d{6})\.csv$")
PERSISTENCE_SDT_B0_PATTERN = re.compile(
    r"^persistence_sdt_b0_(\d{8}_\d{6})\.csv$")
PERSISTENCE_SDT_B1_PATTERN = re.compile(
    r"^persistence_sdt_b1_(\d{8}_\d{6})\.csv$")
PERSISTENCE_DIST_PATTERN = re.compile(
    r"^persistence_distances_(\d{8}_\d{6})\.csv$")

PERSISTENCE_THRESHOLD_RAW = 0.01
PERSISTENCE_THRESHOLD_SDT = 2.0

# Patch-grouping for stitched segmentation overlay plots
_PLOT_PATCH_RE = re.compile(r'^(.+)_patch_(\d+)$')
_STITCH_GRID = (4, 4)
_N_STITCH_PATCHES = _STITCH_GRID[0] * _STITCH_GRID[1]


def _render_seg_overlay(
    img: torch.Tensor | None,
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    alpha: float = 0.6,
) -> np.ndarray:
    """Composite prediction (cyan) and false-negative pixels (magenta) over the image.

    Parameters
    ----------
    img       : (3, H, W) float tensor in [0, 1], or None for a black background.
    pred_mask : (1, H, W) binary prediction tensor.
    gt_mask   : (1, H, W) binary ground-truth tensor.

    Returns
    -------
    (H, W, 3) uint8 array suitable for PIL / plt.imsave.
    """
    h, w = pred_mask.shape[-2:]
    base = img.permute(1, 2, 0).cpu().numpy() if img is not None \
        else np.zeros((h, w, 3), dtype=np.float32)
    result = base.astype(np.float32).copy()

    pred_np = pred_mask.squeeze(0).cpu().float().numpy()
    fn_np = ((gt_mask == 1) & (pred_mask == 0)
             ).squeeze(0).cpu().float().numpy()

    cyan = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    magenta = np.array([1.0, 0.0, 0.5], dtype=np.float32)

    m_pred = pred_np > 0
    result[m_pred] = result[m_pred] * (1 - alpha) + cyan * alpha
    m_fn = fn_np > 0
    result[m_fn] = result[m_fn] * (1 - alpha) + magenta * alpha

    return (result.clip(0, 1) * 255).astype(np.uint8)


def finish_seg_overlay_plots(plot_dir: Path) -> None:
    """Stitch per-patch overlay PNGs into full-image plots and clean up.

    For full images: finds base image names via ``get_base_images``, collects
    the 16 matching ``{base}_patch_{idx:02d}.png`` files, tiles them into a
    4×4 canvas and deletes the individual patch files.  Incomplete groups
    (e.g. from ``--test`` mode) are left as individual files.
    For SoI images (single-patch): renames ``{base}_patch_00.png`` to 
    ``{base}.png``.
    """

    rows, cols = _STITCH_GRID

    # Full images
    for base_name in get_base_images(plot_dir, exclude_soi_images=True):
        patch_paths = sorted(
            p for p in plot_dir.glob("*.png")
            if p.name.startswith(f"{base_name}_patch")
        )
        if not patch_paths or len(patch_paths) != _N_STITCH_PATCHES:
            continue
        indexed = []
        for p in patch_paths:
            m = _PLOT_PATCH_RE.match(p.stem)
            indexed.append((int(m.group(2)), p))
        indexed.sort(key=lambda t: t[0])
        h, w = np.array(Image.open(indexed[0][1])).shape[:2]
        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for idx, p in indexed:
            arr = np.array(Image.open(p))
            r, c = divmod(idx, cols)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = arr[:, :, :3]
        Image.fromarray(canvas).save(plot_dir / f"{base_name}.png")
        for _, p in indexed:
            p.unlink()

    # SoI images: rename {base_name}_patch_00.png → {base_name}.png
    all_names = get_base_images(plot_dir, exclude_soi_images=False)
    for base_name in (n for n in all_names if "_soi_" in n):
        patch_paths = sorted(
            p for p in plot_dir.glob("*.png")
            if p.name.startswith(f"{base_name}_patch")
        )
        if len(patch_paths) == 1:
            patch_paths[0].rename(plot_dir / f"{base_name}.png")


def format_score(x) -> str:
    """Format a float score to 4 decimal places, or 'N/A' if None."""
    return f"{x:.4f}" if x is not None else "N/A"


def load_transform_image(
    path: Path,
    is_mask: bool = False,
    downsample_size=None,
    is_full_image: bool = False,
) -> torch.Tensor:
    """Load an image or mask from disk and apply standard transforms.

    Parameters
    ----------
    path:
        Path to the image file.
    is_mask:
        If True the image is a binary mask (no channel repeat, NEAREST interp).
    downsample_size:
        Optional (H, W) tuple.  When provided and *not* a mask or full image,
        the image is first downsampled to this size and then upscaled back to
        1024×1024 (simulates lower-resolution training data).
    is_full_image:
        If True no spatial resize is performed (the caller receives the image
        at its native resolution).  Use this when loading 4096×4096 originals.
    """
    transform_steps = []

    if not is_full_image:
        if downsample_size is not None and not is_mask:
            transform_steps.append(
                transforms.Resize(downsample_size,
                                  interpolation=transforms.InterpolationMode.BILINEAR))
        interp = (transforms.InterpolationMode.NEAREST
                  if is_mask else transforms.InterpolationMode.BILINEAR)
        transform_steps.append(transforms.Resize(
            (1024, 1024), interpolation=interp))

    transform_steps += [
        transforms.ToTensor(),
        transforms.Lambda(
            lambda t: t.repeat(
                3, 1, 1) if t.shape[0] == 1 and not is_mask else t
        ),
    ]

    transform = transforms.Compose(transform_steps)
    img = Image.open(path)
    return transform(img)


def get_single_image_input_list(image: torch.Tensor) -> list:
    """Wrap a single image tensor into the list format expected by SAM."""
    return [
        {"image": img, "original_size": (img.shape[1], img.shape[2])}
        for img in image.unsqueeze(0).unbind(0)
    ]


def get_batched_input_list(batched_input: torch.Tensor) -> list:
    """Wrap a batch tensor (B, 3, H, W) into the list format expected by SAM."""
    return [
        {"image": img, "original_size": (img.shape[1], img.shape[2])}
        for img in batched_input.unbind(0)
    ]


def hard_dice_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    return (2 * intersection + 1e-6) / (union + 1e-6)


def iou_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def hard_clDice(mask_predicted, mask_target):
    def cl_score(img, skeleton):
        return (np.sum(img * skeleton) + 1e-6) / (np.sum(skeleton) + 1e-6)

    tprec = cl_score(mask_target, skeletonize(mask_predicted))
    tsens = cl_score(mask_predicted, skeletonize(mask_target))
    cl_dice = (2 * tprec * tsens + 1e-6) / (tprec + tsens + 1e-6)
    return cl_dice, tprec, tsens


def compute_metrics(
    output_mask: torch.Tensor,
    groundtruth_mask: torch.Tensor,
) -> tuple:
    """Compute Dice, IoU, clDice, tprec, and tsens between two binary masks.

    Expects tensors of shape (1, H, W) or (H, W).  The first dimension
    is removed via squeeze(0) before passing to numpy-based functions.

    Returns
    -------
    (dice, iou, clDice, tprec, tsens) – dice and iou are tensors; the rest
    are floats.
    """
    dice = hard_dice_score(output_mask, groundtruth_mask)
    iou = iou_score(output_mask, groundtruth_mask)

    output_mask_np = output_mask.squeeze(0).cpu().numpy()
    gt_mask_np = groundtruth_mask.squeeze(0).cpu().numpy()

    clDice, tprec, tsens = hard_clDice(output_mask_np, gt_mask_np)
    return dice, iou, clDice, tprec, tsens


_B0_COLOR = "tomato"
_B1_COLOR = "steelblue"

_LEGEND_HANDLES = [
    mpatches.Patch(color=_B0_COLOR, label="B0 (components)"),
    mpatches.Patch(color=_B1_COLOR, label="B1 (loops)"),
]


def _dynamic_lim(pairs_lists: list[list[tuple[float, float]]]) -> tuple[float, float]:
    """Compute axis limits from multiple pair lists (finite pairs only)."""
    all_vals = [v for pairs in pairs_lists for b,
                d in pairs if d < np.inf for v in (b, d)]
    if not all_vals:
        return 0.0, 1.0
    lo, hi = min(all_vals), max(all_vals)
    pad = max((hi - lo) * 0.1, 0.05)
    return lo - pad, hi + pad


def _plot_pd_axis(
    ax,
    b0_pairs: list[tuple[float, float]],
    b1_pairs: list[tuple[float, float]],
    lim: tuple[float, float],
) -> None:
    """Persistence diagram for one source (pred or gt) on *ax*."""
    lo, hi = lim
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    for pairs, color in [(b0_pairs, _B0_COLOR), (b1_pairs, _B1_COLOR)]:
        finite = [(b, d) for b, d in pairs if d < np.inf]
        if finite:
            bs, ds = zip(*finite)
            ax.scatter(bs, ds, c=color, s=15, alpha=0.7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Birth (filtration)")
    ax.set_ylabel("Death (filtration)")


import numpy as np


def _plot_barcode_axis(
    ax,
    b0_pairs: list[tuple[float, float]],
    b1_pairs: list[tuple[float, float]],
    xlim: tuple[float, float],
    max_bar_height: float = 0.02,   # maximum bar thickness
    bar_step: float = 0.035,        # preferred spacing between bar centres
    margin: float = 0.04,           # padding from top/bottom
) -> None:
    # finite pairs, sorted by persistence descending
    b0 = sorted([(b, d) for b, d in b0_pairs if np.isfinite(d)],
                key=lambda p: (p[0], -(p[1] - p[0])))
    b1 = sorted([(b, d) for b, d in b1_pairs if np.isfinite(d)],
                key=lambda p: (p[1], -(p[1] - p[0])))

    # normalize vertical axis so groups can "stick" to edges
    ax.set_ylim(0.0, 1.0)

    # each group occupies one half of the axis minus the margin
    available = 0.5 - margin
    fill_ratio = max_bar_height / bar_step  # preserve height-to-step ratio

    def _group_geometry(n: int) -> tuple[float, float]:
        """Return (step, height) for a group of n bars."""
        if n == 0:
            return bar_step, max_bar_height
        step = available / n if n * bar_step > available else bar_step
        return step, min(max_bar_height, step * fill_ratio)

    step1, height1 = _group_geometry(len(b1))
    step0, height0 = _group_geometry(len(b0))

    # β1: bottom-up
    y1_start = margin + height1 / 2.0
    for i, (b, d) in enumerate(b1):
        ax.barh(y1_start + i * step1, d - b, left=b,
                height=height1, color=_B1_COLOR)

    # β0: top-down
    y0_start = 1.0 - margin - height0 / 2.0
    for i, (b, d) in enumerate(b0):
        ax.barh(y0_start - i * step0, d - b, left=b,
                height=height0, color=_B0_COLOR)

    lo, hi = xlim
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Filtration value")
    ax.set_yticks([])


def _add_figure_legend(fig: plt.Figure, is_sdt: bool = False) -> None:
    fig.legend(handles=_LEGEND_HANDLES, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.05 if not is_sdt else 1.05), frameon=True, fontsize=8)


def _save_raw_persistence_plot(
    pred_b0: list, pred_b1: list,
    gt_b0: list, gt_b1: list,
    patch_name: str,
    save_path: Path,
) -> None:
    """RAW: 2×2 — row 0: pred PD | pred barcode; row 1: GT PD | GT barcode.

    Axes fixed at [-0.05, 1.05] (probability-map domain).
    """
    lim = (-0.05, 1.05)
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(patch_name, y=1.02)
    subfigs = fig.subfigures(2, 1, hspace=0.02)

    subfigs[0].suptitle("Prediction", fontweight="bold")
    ax_pred = subfigs[0].subplots(1, 2)
    _plot_pd_axis(ax_pred[0], pred_b0, pred_b1, lim)
    _plot_barcode_axis(ax_pred[1], pred_b0, pred_b1, lim)

    subfigs[1].suptitle("Ground Truth", fontweight="bold")
    ax_gt = subfigs[1].subplots(1, 2)
    _plot_pd_axis(ax_gt[0], gt_b0, gt_b1, lim)
    _plot_barcode_axis(ax_gt[1], gt_b0, gt_b1, lim)

    _add_figure_legend(fig, is_sdt=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _save_sdt_persistence_plot(
    pred_b0: list, pred_b1: list,
    gt_b0: list, gt_b1: list,
    patch_name: str,
    save_path: Path,
) -> None:
    """SDT: 2×2 — row 0: pred PD | pred barcode; row 1: GT PD | GT barcode.

    Axes limits computed dynamically from the data.
    """
    lim = _dynamic_lim([pred_b0, pred_b1, gt_b0, gt_b1])
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(patch_name, y=1.02)
    subfigs = fig.subfigures(2, 1, hspace=0.02)

    subfigs[0].suptitle("Prediction", fontweight="bold")
    ax_pred = subfigs[0].subplots(1, 2)
    _plot_pd_axis(ax_pred[0], pred_b0, pred_b1, lim)
    _plot_barcode_axis(ax_pred[1], pred_b0, pred_b1, lim)

    subfigs[1].suptitle("Ground Truth", fontweight="bold")
    ax_gt = subfigs[1].subplots(1, 2)
    _plot_pd_axis(ax_gt[0], gt_b0, gt_b1, lim)
    _plot_barcode_axis(ax_gt[1], gt_b0, gt_b1, lim)

    _add_figure_legend(fig, is_sdt=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _save_persistence_diagrams(
    raw_b0_pred: list, raw_b1_pred: list,
    raw_b0_gt: list, raw_b1_gt: list,
    sdt_b0_pred: list, sdt_b0_gt: list,
    sdt_b1_pred: list, sdt_b1_gt: list,
    run_dir: Path,
    patch_name: str,
) -> None:
    """Save two persistence-diagram figures into *run_dir*.

    Files: ``{patch_name}_raw.png`` (2×2, probability map, fixed [0,1] limits),
           ``{patch_name}_sdt.png`` (2×2, SDT map, dynamic limits).
    """
    _save_raw_persistence_plot(
        raw_b0_pred, raw_b1_pred, raw_b0_gt, raw_b1_gt,
        patch_name, run_dir / f"{patch_name}_raw.png",
    )
    _save_sdt_persistence_plot(
        sdt_b0_pred, sdt_b1_pred, sdt_b0_gt, sdt_b1_gt,
        patch_name, run_dir / f"{patch_name}_sdt.png",
    )


# Shared helpers for patch-level metric / persistence accumulation            #
def _setup_run_dir(save_pd_dir: Path | None) -> Path | None:
    if save_pd_dir is None:
        print("Not saving persistence diagrams.")
        return None
    save_pd_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save persistence diagrams to: {save_pd_dir}")
    return save_pd_dir


def _make_accumulators() -> dict:
    return {
        "dice_s": [], "iou_s": [], "clDice_s": [], "tprec_s": [], "tsens_s": [],
        "pp_dice_s": [], "pp_iou_s": [], "pp_clDice_s": [], "pp_tprec_s": [], "pp_tsens_s": [],
        "raw_b0_rows": [], "raw_b1_rows": [],
        "sdt_b0_rows": [], "sdt_b1_rows": [],
        "pd_distance_rows": [],
    }


def _process_patch(
    patch_name: str,
    pred: torch.Tensor,
    gt: torch.Tensor,
    pp_pred: torch.Tensor,
    prob_map: torch.Tensor,
    run_name: str,
    accum: dict,
    run_dir: Path | None,
) -> None:
    """Compute metrics and persistence data for a single patch, appending
    results into *accum* in-place.

    Parameters
    ----------
    patch_name : identifier stored in each output row's ``image`` field.
    pred       : (1, H, W) binary prediction.
    gt         : (1, H, W) binary ground-truth.
    pp_pred    : (1, H, W) post-processed prediction (small objects removed).
    prob_map   : (1, H, W) soft probability map used for raw persistence.
    run_name   : identifier stored in each output row's ``model`` field.
    accum      : mutable accumulator dict created by ``_make_accumulators()``.
    run_dir    : directory for persistence-diagram figures, or None.
    """
    # Segmentation metrics
    d, iou, cl, tp, ts = compute_metrics(pred, gt)
    accum["dice_s"].append(d.item())
    accum["iou_s"].append(iou.item())
    accum["clDice_s"].append(cl)
    accum["tprec_s"].append(tp)
    accum["tsens_s"].append(ts)

    pp_d, pp_iou, pp_cl, pp_tp, pp_ts = compute_metrics(pp_pred, gt)
    accum["pp_dice_s"].append(pp_d.item())
    accum["pp_iou_s"].append(pp_iou.item())
    accum["pp_clDice_s"].append(pp_cl)
    accum["pp_tprec_s"].append(pp_tp)
    accum["pp_tsens_s"].append(pp_ts)

    # Raw persistence (on soft probability map)
    pred_b0_pairs, pred_b1_pairs = get_persistence_pairs(prob_map)
    gt_b0_pairs, gt_b1_pairs = get_persistence_pairs(gt)

    for pairs, rows_key, type_ in [
        (pred_b0_pairs, "raw_b0_rows", "predicted"),
        (gt_b0_pairs, "raw_b0_rows", "groundtruth"),
        (pred_b1_pairs, "raw_b1_rows", "predicted"),
        (gt_b1_pairs, "raw_b1_rows", "groundtruth"),
    ]:
        for birth, death in pairs:
            accum[rows_key].append(
                {"model": run_name, "image": patch_name,
                 "type": type_, "birth": birth, "death": death})

    # SDT persistence (on binary prediction / ground-truth masks)
    pred_sdt_b0, pred_sdt_b1 = get_sdt_persistence_pairs(pred)
    gt_sdt_b0, gt_sdt_b1 = get_sdt_persistence_pairs(gt)

    for pairs, rows_key, type_ in [
        (pred_sdt_b0, "sdt_b0_rows", "predicted"),
        (gt_sdt_b0, "sdt_b0_rows", "groundtruth"),
        (pred_sdt_b1, "sdt_b1_rows", "predicted"),
        (gt_sdt_b1, "sdt_b1_rows", "groundtruth"),
    ]:
        for birth, death in pairs:
            accum[rows_key].append(
                {"model": run_name, "image": patch_name,
                 "type": type_, "birth": birth, "death": death})

    # Persistence distances
    raw_b0_wd, raw_b0_bn = compute_persistence_distances(
        pred_b0_pairs, gt_b0_pairs)
    raw_b1_wd, raw_b1_bn = compute_persistence_distances(
        pred_b1_pairs, gt_b1_pairs)
    sdt_b0_wd, sdt_b0_bn = compute_persistence_distances(
        pred_sdt_b0, gt_sdt_b0)
    sdt_b1_wd, sdt_b1_bn = compute_persistence_distances(
        pred_sdt_b1, gt_sdt_b1)
    accum["pd_distance_rows"].append({
        "model": run_name, "image": patch_name,
        "raw_b0_wasserstein": raw_b0_wd, "raw_b0_bottleneck": raw_b0_bn,
        "raw_b1_wasserstein": raw_b1_wd, "raw_b1_bottleneck": raw_b1_bn,
        "sdt_b0_wasserstein": sdt_b0_wd, "sdt_b0_bottleneck": sdt_b0_bn,
        "sdt_b1_wasserstein": sdt_b1_wd, "sdt_b1_bottleneck": sdt_b1_bn,
    })

    if run_dir is not None:
        _save_persistence_diagrams(
            pred_b0_pairs, pred_b1_pairs,
            gt_b0_pairs, gt_b1_pairs,
            pred_sdt_b0, gt_sdt_b0,
            pred_sdt_b1, gt_sdt_b1,
            run_dir, patch_name,
        )


def _aggregate_results(accum: dict) -> tuple:
    """Compute mean metrics and mean persistence distances from *accum*,
    returning the standard 8-element tuple shared by both public functions."""
    def _nanmean(key):
        vals = [r[key] for r in accum["pd_distance_rows"]]
        return float(np.nanmean(vals)) if vals else float("nan")

    mean_metrics_raw = (
        float(np.mean(accum["dice_s"])), float(np.mean(accum["iou_s"])),
        float(np.mean(accum["clDice_s"])), float(np.mean(accum["tprec_s"])),
        float(np.mean(accum["tsens_s"])),
    )
    mean_metrics_pp = (
        float(np.mean(accum["pp_dice_s"])), float(np.mean(accum["pp_iou_s"])),
        float(np.mean(accum["pp_clDice_s"])), float(
            np.mean(accum["pp_tprec_s"])),
        float(np.mean(accum["pp_tsens_s"])),
    )
    mean_pd_distances = (
        _nanmean("raw_b0_wasserstein"), _nanmean("raw_b0_bottleneck"),
        _nanmean("raw_b1_wasserstein"), _nanmean("raw_b1_bottleneck"),
        _nanmean("sdt_b0_wasserstein"), _nanmean("sdt_b0_bottleneck"),
        _nanmean("sdt_b1_wasserstein"), _nanmean("sdt_b1_bottleneck"),
    )
    return (mean_metrics_raw, mean_metrics_pp,
            accum["raw_b0_rows"], accum["raw_b1_rows"],
            accum["sdt_b0_rows"], accum["sdt_b1_rows"],
            accum["pd_distance_rows"], mean_pd_distances)


@torch.no_grad()
def collect_patch_metrics_and_betti(
    model,
    loader,
    device,
    run_name: str,
    save_pd_dir: Path | None = None,
    is_test: bool = False,
    plot_dir: Path | None = None,
) -> tuple:
    """Run batched patch-level inference once, computing both segmentation
    metrics and per-patch persistence diagram data

    Parameters
    ----------
    model:
        Model in eval mode.
    loader:
        DataLoader from a SegmentationDataset with ``return_img_name=True``,
        yielding (images, gt_masks, _, patch_names) tuples with shuffle=False.
    device:
        Torch device
    run_name:
        Model/run identifier
    save_pd_dir:
        When provided, a 2×2 persistence-diagram figure is saved per patch as
        ``<run_name>_<patch_stem>_persistence.png`` in this directory.

    Returns
    -------
    raw_metrics          : 5-tuple of floats  (dice, iou, clDice, tprec, tsens)
    pp_metrics           : 5-tuple of floats  (post-processed versions of the above)
    raw_b0_rows          : list of dicts, one row per filtered Betti-0 pair on raw prob map
    raw_b1_rows          : list of dicts, one row per filtered Betti-1 pair on raw prob map
    sdt_b0_rows          : list of dicts, one row per filtered Betti-0 pair on SDT map
    sdt_b1_rows          : list of dicts, one row per filtered Betti-1 pair on SDT map
    pd_distance_rows     : list of dicts, one row per patch with Wasserstein and bottleneck
                           distances for raw B0/B1 and SDT B0/B1
    mean_pd_distances    : 8-tuple of floats
                           (raw_b0_wd, raw_b0_bn, raw_b1_wd, raw_b1_bn,
                            sdt_b0_wd, sdt_b0_bn, sdt_b1_wd, sdt_b1_bn)
    """
    run_dir = _setup_run_dir(save_pd_dir)
    accum = _make_accumulators()

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    img_count = 0
    for images, gt_masks, _, patch_names in loader:
        batch_size = images.shape[0]
        input_list = get_batched_input_list(images.to(device))
        outputs = model(batched_input=input_list, multimask_output=False)

        output_masks = torch.cat([out["masks"]
                                 for out in outputs]).detach().cpu()
        pp_masks = remove_small_objects_from_batch(output_masks)

        # Probability maps: upsample low-res logits and apply sigmoid
        prob_maps = []
        for out in outputs:
            resized = model.sam_model.postprocess_masks(
                out["low_res_logits"],
                input_size=(1024, 1024),
                original_size=(1024, 1024),
            )
            prob_maps.append(torch.sigmoid(resized).squeeze(0))

        for i in range(batch_size):
            _process_patch(
                patch_names[i],
                output_masks[i], gt_masks[i], pp_masks[i], prob_maps[i],
                run_name, accum, run_dir,
            )
            if plot_dir is not None:
                arr = _render_seg_overlay(
                    images[i], output_masks[i], gt_masks[i])
                Image.fromarray(arr).save(plot_dir / f"{patch_names[i]}.png")

            img_count += 1

        print(f"Processed {img_count} patches of {len(loader.dataset)} total")

        # in test mode, run 16 patches (one full image)
        if is_test and img_count >= 16:
            break

    return _aggregate_results(accum)


def get_persistence_pairs_from_filtration(
    filtration: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    cc = gd.CubicalComplex(
        dimensions=filtration.shape,
        top_dimensional_cells=filtration.flatten(),
    )
    cc.compute_persistence()

    max_val = float(filtration.max())

    b0_pairs = [(float(b), min(float(d), max_val))
                for b, d in cc.persistence_intervals_in_dimension(0)
                if (min(float(d), max_val) - float(b)) >= threshold]
    b1_pairs = [(float(b), min(float(d), max_val))
                for b, d in cc.persistence_intervals_in_dimension(1)
                if (min(float(d), max_val) - float(b)) >= threshold]
    return b0_pairs, b1_pairs


def get_persistence_pairs(
    prob_map: torch.Tensor,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Compute cubical persistence pairs for a probability map.

    The filtration is inverted (``filtration = 1 - prob_map``) so that
    high-probability regions appear first.  Birth and death values are in
    filtration space; convert back via ``probability = 1 - filtration_value``.

    Parameters
    ----------
    prob_map:
        tensor of shape (1, H, W) or (H, W) containing predicted probabilities in [0, 1].

    Returns
    -------
    b0_pairs : list of (birth, death) tuples for Betti-0
    b1_pairs : list of (birth, death) tuples for Betti-1
    """
    prob_map = prob_map.squeeze(0).cpu().numpy()
    padded = np.pad(prob_map, pad_width=1, mode="constant", constant_values=0)
    filtration = 1.0 - padded
    return get_persistence_pairs_from_filtration(filtration, threshold=PERSISTENCE_THRESHOLD_RAW)


def get_sdt_persistence_pairs(mask: torch.Tensor) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Compute cubical persistence pairs on the signed distance transform (SDT)
    of a binary mask.

    The SDT is positive inside the mask and negative outside.  The filtration
    is ``-SDT`` so that interior (positive) regions appear first in the
    sublevel-set filtration, matching the convention used for probability maps.

    Parameters
    ----------
    mask:
        mask tensor of shape (1, H, W) or (H, W)

    Returns
    -------
    b0_pairs : list of (birth, death) tuples for Betti-0
    b1_pairs : list of (birth, death) tuples for Betti-1
    """
    mask = mask.squeeze(0).cpu().numpy().astype(bool)
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    sdt = inside - outside
    filtration = -sdt
    return get_persistence_pairs_from_filtration(filtration, threshold=PERSISTENCE_THRESHOLD_SDT)


def compute_persistence_distances(
    pred_pairs: list[tuple[float, float]],
    gt_pairs: list[tuple[float, float]],
) -> tuple[float, float]:
    """Compute Wasserstein-1 and bottleneck distance between two persistence diagrams.

    Only finite-death pairs are used to avoid numerical issues with the single
    essential (infinite-death) component present in every cubical complex.

    Returns
    -------
    (wasserstein_dist, bottleneck_dist)
    """
    pred_finite = [(b, d) for b, d in pred_pairs]
    gt_finite = [(b, d) for b, d in gt_pairs]
    pred_arr = np.array(pred_finite, dtype=float)
    gt_arr = np.array(gt_finite, dtype=float)
    try:
        wd = float(gudhi_wasserstein_distance(
            pred_arr, gt_arr, internal_p=2.0))
    except Exception:
        wd = float("nan")
    try:
        bn = float(gd.bottleneck_distance(pred_arr, gt_arr))
    except Exception:
        bn = float("nan")
    return wd, bn


def get_betti_at_thresholds(
    prob_map: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[list, list]:
    """Compute Betti-0 (connected components) and Betti-1 (loops) at multiple
    probability thresholds using cubical persistent homology (gudhi).

    The *prob_map* is treated as a probability map in [0, 1]; the filtration
    direction is inverted so that high-probability regions appear first.

    Parameters
    ----------
    prob_map:
        2-D float array of predicted probabilities (or a binary mask).
    thresholds:
        1-D array of thresholds at which to evaluate the Betti numbers.

    Returns
    -------
    b0, b1 – lists of integers, one per threshold value.
    """
    padded = np.pad(prob_map, pad_width=1, mode="constant", constant_values=0)
    filtration = 1.0 - padded

    cc = gd.CubicalComplex(
        dimensions=filtration.shape,
        top_dimensional_cells=filtration.flatten(),
    )
    cc.compute_persistence()

    pairs = {0: [], 1: []}
    for dim in [0, 1]:
        for birth, death in cc.persistence_intervals_in_dimension(dim):
            pairs[dim].append((birth, death))

    thresholds = np.asarray(thresholds)
    inv_t = 1.0 - thresholds

    b0, b1 = [], []
    for t in inv_t:
        b0.append(sum(1 for b, d in pairs[0] if b <= t and d > t))
        b1.append(sum(1 for b, d in pairs[1] if b <= t and d > t))

    return b0, b1


def plot_betti_curve(
    thresholds: np.ndarray,
    pred_b0,
    pred_b1,
    groundtruth_b0: float,
    groundtruth_b1: float,
    figsize: tuple = (6, 4),
):
    """Plot predicted Betti curves vs ground-truth horizontal reference lines."""
    plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.plot(thresholds, pred_b0, label="Betti-0 (Components)", color="blue")
    ax.plot(thresholds, pred_b1, label="Betti-1 (Loops)", color="orange")
    ax.axhline(y=groundtruth_b0, label="Groundtruth Betti-0",
               color="blue", linestyle="--")
    ax.axhline(y=groundtruth_b1, label="Groundtruth Betti-1",
               color="orange", linestyle="--")

    ax.set_yscale("log")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Betti Number")
    ax.legend()
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()
    plt.close()


def _load_binary_mask_tensor(path: Path) -> torch.Tensor:
    """Load a binary mask image as a (1, H, W) float32 tensor (values 0.0/1.0).

    Handles both 0/255 (standard PNG) and 0/1 (nnUNet convention) encodings.
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy((arr > 0).astype(np.float32)).unsqueeze(0)


@torch.no_grad()
def collect_patch_metrics_and_betti_from_masks(
    gt_mask_dir: Path,
    pred_mask_dir: Path,
    run_name: str,
    save_pd_dir: Path | None = None,
    is_test: bool = False,
    plot_dir: Path | None = None,
    plot_case_mapping: dict | None = None,
    original_img_patches_dir: Path | None = None,
) -> tuple:
    """Compute patch-level segmentation metrics and persistence diagrams by
    loading pre-computed binary prediction masks from a directory.

    Replaces model inference in ``collect_patch_metrics_and_betti`` with direct
    mask loading from disk.  Intended for evaluating nnUNet predictions.

    Prediction files are matched to ground-truth files by stem (filename
    without extension).  Files present in *pred_mask_dir* that have no
    matching entry in *gt_mask_dir* are skipped with a warning.

    The binary prediction is used as its own "probability map" for the raw
    persistence-diagram computation (filtration values are 0 or 1, so the
    curves are degenerate step functions — still a valid topological summary).

    Parameters
    ----------
    gt_mask_dir   : directory containing ground-truth mask files named
                    ``{case}.*``.
    pred_mask_dir : directory containing predicted mask files named
                    ``{case}.*``.
    run_name      : identifier stored in each output row's ``model`` field.
    save_pd_dir   : when given, persistence-diagram figures are saved here.

    Returns
    -------
    Same 8-element tuple as ``collect_patch_metrics_and_betti``:
    ``(mean_metrics_raw, mean_metrics_pp,
      raw_b0_rows, raw_b1_rows, sdt_b0_rows, sdt_b1_rows,
      pd_distance_rows, mean_pd_distances)``
    """
    run_dir = _setup_run_dir(save_pd_dir)
    accum = _make_accumulators()

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    gt_index = {p.stem: p for p in gt_mask_dir.iterdir() if p.is_file()
                and p.suffix == ".png"}
    pred_files = sorted(
        p for p in pred_mask_dir.iterdir()
        if p.is_file() and p.suffix == ".png"
    )

    for idx, pred_path in enumerate(pred_files):
        print(
            f"Processing patch {idx + 1}/{len(pred_files)}: {pred_path.name}", flush=True)
        case = pred_path.stem
        gt_path = gt_index.get(case)
        if gt_path is None:
            print(
                f"  [WARN] No ground-truth found for prediction '{case}', skipping.")
            continue

        pred = _load_binary_mask_tensor(pred_path)  # (1, H, W)
        gt = _load_binary_mask_tensor(gt_path)      # (1, H, W)
        pp_pred = remove_small_objects_from_batch(
            pred.unsqueeze(0)).squeeze(0)           # (1, H, W)

        npz_path = pred_path.parent / f"{case}.npz"
        npz_data = np.load(npz_path)
        key = "probabilities" if "probabilities" in npz_data else npz_data.files[0]
        probs_np = npz_data[key]  # (num_classes, 1, H, W)
        fg_np = probs_np[1] if probs_np.shape[0] > 1 else probs_np[0]
        prob_map = torch.from_numpy(fg_np.astype(np.float32))

        _process_patch(case, pred, gt, pp_pred, prob_map,
                       run_name, accum, run_dir)

        if plot_dir is not None:
            img_tensor = None
            out_name = case
            if plot_case_mapping is not None and case in plot_case_mapping:
                orig_filename, patch_idx = plot_case_mapping[case]
                orig_filename_stem = orig_filename.replace(".png", "")
                if isinstance(patch_idx, int) and not orig_filename_stem.endswith("_soi"):
                    out_name = f"{orig_filename_stem}_patch_{patch_idx:02d}.png"
                elif isinstance(patch_idx, str) and patch_idx == "" and orig_filename_stem.endswith("_soi"):
                    out_name = f"{orig_filename_stem}.png"
                if original_img_patches_dir is not None:
                    img_path = original_img_patches_dir / out_name
                    if img_path.is_file():
                        img_tensor = load_transform_image(img_path)
            arr = _render_seg_overlay(img_tensor, pred, gt)
            Image.fromarray(arr).save(plot_dir / out_name)

        print(
            f"Processed {idx + 1} (pre-predicted) patches of {len(pred_files)} total", flush=True)

        # in test mode, run 16 patches (one full image)
        if is_test and idx >= 15:
            break

    return _aggregate_results(accum)


def _load_latest_csv(metrics_dir: Path, pattern: re.Pattern) -> tuple[pd.DataFrame, Path | None]:
    """Return the most recently modified CSV matching *pattern*, or empty DF."""
    candidates = [p for p in metrics_dir.glob(
        "*.csv") if pattern.match(p.name)]
    if not candidates:
        return pd.DataFrame(), None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest)
    print(f"Loaded previous data from: {latest} (rows={len(df)})")
    return df, latest


def load_latest_metrics_csv(metrics_dir: Path) -> pd.DataFrame:
    """Load the most recent ``metrics_YYYYMMDD_HHMMSS.csv`` in *metrics_dir*.

    Returns an empty DataFrame if none exists.  The ``Model`` column is used
    as the index.
    """
    df, _ = _load_latest_csv(metrics_dir, METRICS_PATTERN)
    if not df.empty and "Model" in df.columns:
        df = df.set_index("Model")
        df.index = df.index.astype(str)
    return df


def load_latest_betti_csv(metrics_dir: Path) -> pd.DataFrame:
    """Load the most recent ``betti_YYYYMMDD_HHMMSS.csv`` in *metrics_dir*.

    Returns an empty DataFrame if none exists.
    """
    df, _ = _load_latest_csv(metrics_dir, BETTI_PATTERN)
    return df


def load_latest_persistence_raw_b0_csv(metrics_dir: Path) -> pd.DataFrame:
    """Load the most recent ``persistence_raw_b0_*.csv``. Columns: model, image, type, birth, death."""
    df, _ = _load_latest_csv(metrics_dir, PERSISTENCE_RAW_B0_PATTERN)
    return df


def load_latest_persistence_raw_b1_csv(metrics_dir: Path) -> pd.DataFrame:
    """Load the most recent ``persistence_raw_b1_*.csv``. Columns: model, image, type, birth, death."""
    df, _ = _load_latest_csv(metrics_dir, PERSISTENCE_RAW_B1_PATTERN)
    return df


def load_latest_persistence_sdt_b0_csv(metrics_dir: Path) -> pd.DataFrame:
    """Load the most recent ``persistence_sdt_b0_*.csv``. Columns: model, image, type, birth, death."""
    df, _ = _load_latest_csv(metrics_dir, PERSISTENCE_SDT_B0_PATTERN)
    return df


def load_latest_persistence_sdt_b1_csv(metrics_dir: Path) -> pd.DataFrame:
    """Load the most recent ``persistence_sdt_b1_*.csv``. Columns: model, image, type, birth, death."""
    df, _ = _load_latest_csv(metrics_dir, PERSISTENCE_SDT_B1_PATTERN)
    return df


def load_latest_persistence_distances_csv(metrics_dir: Path) -> pd.DataFrame:
    """Load the most recent ``persistence_distances_*.csv``.

    Columns: model, image, raw_b0_wasserstein, raw_b0_bottleneck,
             raw_b1_wasserstein, raw_b1_bottleneck,
             sdt_b0_wasserstein, sdt_b0_bottleneck,
             sdt_b1_wasserstein, sdt_b1_bottleneck.
    """
    df, _ = _load_latest_csv(metrics_dir, PERSISTENCE_DIST_PATTERN)
    return df
