import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import gudhi as gd
from skimage.morphology import skeletonize

from Segmentation.PostProcessing.segmentation_postprocessing import remove_small_objects_from_batch

# CSV filename patterns
METRICS_PATTERN = re.compile(r"^metrics_(\d{8}_\d{6})\.csv$")
BETTI_PATTERN = re.compile(r"^betti_(\d{8}_\d{6})\.csv$")


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


@torch.no_grad()
def collect_patch_metrics(model, loader, device) -> tuple:
    """Run batched patch-level inference and return averaged metrics,
    computing metrics on both raw model outputs and post-processed masks

    Parameters
    ----------
    model:
        model in eval mode.
    loader:
        DataLoader yielding (images, gt_masks, _) tuples.
        images shape: (B, 3, H, W); gt_masks shape: (B, 1, H, W).
    device:
        Torch device for the forward pass.

    Returns
    -------
    (raw_metrics, pp_metrics) where each is a 5-tuple of floats: (dice_s, iou_s, clDice_s, tprec_s, tsens_s)
    """
    dice_s, iou_s, clDice_s, tprec_s, tsens_s = [], [], [], [], []
    pp_dice_s, pp_iou_s, pp_clDice_s, pp_tprec_s, pp_tsens_s = [], [], [], [], []

    for images, gt_masks, _ in loader:
        input_list = get_batched_input_list(images.to(device))
        outputs = model(batched_input=input_list, multimask_output=False)
        output_masks = torch.stack([out["masks"]
                                   for out in outputs]).detach().cpu()
        pp_masks = remove_small_objects_from_batch(output_masks)

        print(f"Processing batch of {output_masks.shape[0]} images")
        print(f"    Output masks shape: {output_masks.shape}")
        print(f"    Postprocessed output masks shape: {pp_masks.shape}")
        print(f"    Ground truth masks shape: {gt_masks.shape}")

        for i in range(output_masks.shape[0]):
            gt = gt_masks[i:i + 1]
            pred = output_masks[i:i + 1]
            pp_pred = pp_masks[i:i + 1]

            print(f"\n    processing image {i + 1}")
            print(
                f"    pred shape: {pred.shape}, pp_pred shape: {pp_pred.shape}, gt shape: {gt.shape}")

            dice, iou, clDice, tprec, tsens = compute_metrics(pred, gt)
            dice_s.append(dice.item())
            iou_s.append(iou.item())
            clDice_s.append(clDice)
            tprec_s.append(tprec)
            tsens_s.append(tsens)

            pp_dice, pp_iou, pp_clDice, pp_tprec, pp_tsens = compute_metrics(
                pp_pred, gt)
            pp_dice_s.append(pp_dice.item())
            pp_iou_s.append(pp_iou.item())
            pp_clDice_s.append(pp_clDice)
            pp_tprec_s.append(pp_tprec)
            pp_tsens_s.append(pp_tsens)

    return (
        (float(np.mean(dice_s)), float(np.mean(iou_s)), float(
            np.mean(clDice_s)), float(np.mean(tprec_s)), float(np.mean(tsens_s))),
        (float(np.mean(pp_dice_s)), float(np.mean(pp_iou_s)), float(
            np.mean(pp_clDice_s)), float(np.mean(pp_tprec_s)), float(np.mean(pp_tsens_s))),
    )


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


def topological_fscore_single(pred_b0: float, gt_b0: float) -> float:
    """Topological F-score for a single image based on Betti-0.

    Computes F1 between predicted and ground-truth number of connected
    components using a capped precision/recall formulation.
    """
    if pred_b0 == 0 and gt_b0 == 0:
        return 1.0
    if pred_b0 == 0 or gt_b0 == 0:
        return 0.0
    rec = min(1.0, pred_b0 / gt_b0)
    pre = min(1.0, gt_b0 / pred_b0)
    return 2.0 * (pre * rec) / (pre + rec)


def compute_betti_aggregate_metrics(
    all_b0_curves: list,
    all_b1_curves: list,
    gt_b0_list: list,
    gt_b1_list: list,
    thresholds: np.ndarray,
) -> tuple[list, list]:
    """Compute per-threshold topological F-score (B0) and MAE (B1) over images.

    Parameters
    ----------
    all_b0_curves:
        List of length n_images; each element is a list of length n_thresholds
        containing the predicted Betti-0 values.
    all_b1_curves:
        Same structure for Betti-1.
    gt_b0_list:
        Ground-truth Betti-0 (scalar) per image.
    gt_b1_list:
        Ground-truth Betti-1 (scalar) per image.
    thresholds:
        Array of threshold values used when computing the curves.

    Returns
    -------
    fscore_per_threshold:
        Mean topological F-score (Betti-0) at each threshold.
    mae_b1_per_threshold:
        Mean absolute error of Betti-1 at each threshold.
    """
    all_b0 = np.array(all_b0_curves)   # [n_images, n_thresholds]
    all_b1 = np.array(all_b1_curves)
    gt_b0 = np.array(gt_b0_list)
    gt_b1 = np.array(gt_b1_list)

    fscore_per_threshold, mae_b1_per_threshold = [], []
    for i in range(len(thresholds)):
        f_scores = [topological_fscore_single(p, g)
                    for p, g in zip(all_b0[:, i], gt_b0)]
        fscore_per_threshold.append(float(np.mean(f_scores)))
        mae_b1_per_threshold.append(
            float(np.mean(np.abs(all_b1[:, i] - gt_b1))))

    return fscore_per_threshold, mae_b1_per_threshold


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


def plot_betti_aggregate(
    thresholds: np.ndarray,
    fscore_per_threshold: list,
    mae_b1_per_threshold: list,
    label: str = "",
    ax_f=None,
    ax_e=None,
    figsize: tuple = (10, 5),
):
    """Plot per-threshold Betti-0 F-score and Betti-1 MAE on twin axes.

    If *ax_f* and *ax_e* are provided they are reused (for multi-model
    overlays); otherwise a fresh figure is created.
    """
    created_fig = ax_f is None
    if created_fig:
        _, ax_f = plt.subplots(figsize=figsize)
        ax_e = ax_f.twinx()

    ax_f.plot(thresholds, fscore_per_threshold,
              linewidth=2, label=f"B0 F-Score {label}")
    ax_e.plot(thresholds, mae_b1_per_threshold,
              linewidth=2, linestyle="--", label=f"B1 MAE {label}")

    ax_f.set_xlabel("Probability Threshold")
    ax_f.set_ylabel("Avg. Betti-0 F-Score")
    ax_e.set_ylabel("Avg. Betti-1 MAE")
    ax_f.invert_xaxis()

    if created_fig:
        ax_f.legend(loc="upper left")
        ax_e.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        plt.close()

    return ax_f, ax_e


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
