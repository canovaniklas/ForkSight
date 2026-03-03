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
PERSISTENCE_PATTERN = re.compile(r"^persistence_(\d{8}_\d{6})\.csv$")


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
def collect_patch_metrics_and_betti(
    model,
    loader,
    device,
    run_name: str,
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

    Returns
    -------
    raw_metrics     : 5-tuple of floats  (dice, iou, clDice, tprec, tsens)
    pp_metrics      : 5-tuple of floats  (post-processed versions of the above)
    persistence_rows: list of dicts, one row per birth-death pair per patch,
                      with keys (model, image, type, dim, birth, death)
    """
    dice_s, iou_s, clDice_s, tprec_s, tsens_s = [], [], [], [], []
    pp_dice_s, pp_iou_s, pp_clDice_s, pp_tprec_s, pp_tsens_s = [], [], [], [], []
    persistence_rows = []

    for images, gt_masks, _, patch_names in loader:
        batch_size = images.shape[0]
        input_list = get_batched_input_list(images.to(device))
        outputs = model(batched_input=input_list, multimask_output=False)

        output_masks = torch.stack([out["masks"]
                                   for out in outputs]).squeeze(0).detach().cpu()
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
            patch_name = patch_names[i]
            gt = gt_masks[i]
            pred = output_masks[i]
            pp_pred = pp_masks[i]

            print(
                f"gt mask dim: {gt.shape}, pred mask dim: {pred.shape}, pp_pred mask dim: {pp_pred.shape}")

            # Segmentation metrics
            d, iou, cl, tp, ts = compute_metrics(pred, gt)
            dice_s.append(d.item())
            iou_s.append(iou.item())
            clDice_s.append(cl)
            tprec_s.append(tp)
            tsens_s.append(ts)

            pp_d, pp_iou, pp_cl, pp_tp, pp_ts = compute_metrics(pp_pred, gt)
            pp_dice_s.append(pp_d.item())
            pp_iou_s.append(pp_iou.item())
            pp_clDice_s.append(pp_cl)
            pp_tprec_s.append(pp_tp)
            pp_tsens_s.append(pp_ts)

            # Persistence diagrams
            prob_np = prob_maps[i].squeeze(0).cpu().numpy()
            gt_np = gt.squeeze(0).cpu().numpy()
            print(
                f"numpy probability map dims: {prob_np.shape}, numpy gt mask dims: {gt_np.shape}")
            pred_b0_pairs, pred_b1_pairs = get_persistence_pairs(prob_np)
            gt_b0_pairs, gt_b1_pairs = get_persistence_pairs(gt_np)

            for dim, pred_pairs, gt_pairs in [
                (0, pred_b0_pairs, gt_b0_pairs),
                (1, pred_b1_pairs, gt_b1_pairs),
            ]:
                for birth, death in pred_pairs:
                    persistence_rows.append({
                        "model": run_name, "image": patch_name,
                        "type": "predicted", "dim": dim,
                        "birth": birth, "death": death,
                    })
                for birth, death in gt_pairs:
                    persistence_rows.append({
                        "model": run_name, "image": patch_name,
                        "type": "groundtruth", "dim": dim,
                        "birth": birth, "death": death,
                    })

    mean_metrics_raw = (
        float(np.mean(dice_s)), float(
            np.mean(iou_s)), float(np.mean(clDice_s)),
        float(np.mean(tprec_s)), float(np.mean(tsens_s)),
    )
    mean_metrics_pp = (
        float(np.mean(pp_dice_s)), float(
            np.mean(pp_iou_s)), float(np.mean(pp_clDice_s)),
        float(np.mean(pp_tprec_s)), float(np.mean(pp_tsens_s)),
    )
    return mean_metrics_raw, mean_metrics_pp, persistence_rows


def get_persistence_pairs(
    prob_map: np.ndarray,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Compute cubical persistence pairs for a probability map.

    The filtration is inverted (``filtration = 1 - prob_map``) so that
    high-probability regions appear first.  Birth and death values are in
    filtration space; convert back via ``probability = 1 - filtration_value``.

    Parameters
    ----------
    prob_map:
        2-D float array of predicted probabilities or a binary mask.

    Returns
    -------
    b0_pairs : list of (birth, death) tuples for Betti-0
    b1_pairs : list of (birth, death) tuples for Betti-1
    """
    padded = np.pad(prob_map, pad_width=1, mode="constant", constant_values=0)
    filtration = 1.0 - padded
    cc = gd.CubicalComplex(
        dimensions=filtration.shape,
        top_dimensional_cells=filtration.flatten(),
    )
    cc.compute_persistence()
    b0_pairs = [(float(b), float(d))
                for b, d in cc.persistence_intervals_in_dimension(0)]
    b1_pairs = [(float(b), float(d))
                for b, d in cc.persistence_intervals_in_dimension(1)]
    return b0_pairs, b1_pairs


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


def load_latest_persistence_csv(metrics_dir: Path) -> pd.DataFrame:
    """Load the most recent ``persistence_YYYYMMDD_HHMMSS.csv`` in *metrics_dir*.

    Returns an empty DataFrame if none exists.
    Columns: model, image, type, dim, birth, death.
    """
    df, _ = _load_latest_csv(metrics_dir, PERSISTENCE_PATTERN)
    return df
