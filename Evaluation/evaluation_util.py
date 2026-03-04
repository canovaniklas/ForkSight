import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance as gudhi_wasserstein_distance
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from Segmentation.PostProcessing.segmentation_postprocessing import remove_small_objects_from_batch


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
    dice_s, iou_s, clDice_s, tprec_s, tsens_s = [], [], [], [], []
    pp_dice_s, pp_iou_s, pp_clDice_s, pp_tprec_s, pp_tsens_s = [], [], [], [], []
    raw_b0_rows, raw_b1_rows, sdt_b0_rows, sdt_b1_rows = [], [], [], []
    pd_distance_rows = []

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

            # persistence diagrams on raw probability maps
            pred_b0_pairs, pred_b1_pairs = get_persistence_pairs(prob_maps[i])
            gt_b0_pairs, gt_b1_pairs = get_persistence_pairs(gt)

            for birth, death in pred_b0_pairs:
                raw_b0_rows.append({"model": run_name, "image": patch_name,
                                    "type": "predicted", "birth": birth, "death": death})
            for birth, death in gt_b0_pairs:
                raw_b0_rows.append({"model": run_name, "image": patch_name,
                                    "type": "groundtruth", "birth": birth, "death": death})
            for birth, death in pred_b1_pairs:
                raw_b1_rows.append({"model": run_name, "image": patch_name,
                                    "type": "predicted", "birth": birth, "death": death})
            for birth, death in gt_b1_pairs:
                raw_b1_rows.append({"model": run_name, "image": patch_name,
                                    "type": "groundtruth", "birth": birth, "death": death})

            # persistence diagrams on signed distance transform (SDT) maps calculated from binary masks
            pred_sdt_b0, pred_sdt_b1 = get_sdt_persistence_pairs(pred)
            gt_sdt_b0, gt_sdt_b1 = get_sdt_persistence_pairs(gt)

            for birth, death in pred_sdt_b0:
                sdt_b0_rows.append({"model": run_name, "image": patch_name,
                                    "type": "predicted", "birth": birth, "death": death})
            for birth, death in gt_sdt_b0:
                sdt_b0_rows.append({"model": run_name, "image": patch_name,
                                    "type": "groundtruth", "birth": birth, "death": death})
            for birth, death in pred_sdt_b1:
                sdt_b1_rows.append({"model": run_name, "image": patch_name,
                                    "type": "predicted", "birth": birth, "death": death})
            for birth, death in gt_sdt_b1:
                sdt_b1_rows.append({"model": run_name, "image": patch_name,
                                    "type": "groundtruth", "birth": birth, "death": death})

            # Persistence diagram distances (Wasserstein-1 and bottleneck)
            raw_b0_wd, raw_b0_bn = compute_persistence_distances(
                pred_b0_pairs, gt_b0_pairs)
            raw_b1_wd, raw_b1_bn = compute_persistence_distances(
                pred_b1_pairs, gt_b1_pairs)
            sdt_b0_wd, sdt_b0_bn = compute_persistence_distances(
                pred_sdt_b0, gt_sdt_b0)
            sdt_b1_wd, sdt_b1_bn = compute_persistence_distances(
                pred_sdt_b1, gt_sdt_b1)
            pd_distance_rows.append({
                "model": run_name,
                "image": patch_name,
                "raw_b0_wasserstein": raw_b0_wd,
                "raw_b0_bottleneck": raw_b0_bn,
                "raw_b1_wasserstein": raw_b1_wd,
                "raw_b1_bottleneck": raw_b1_bn,
                "sdt_b0_wasserstein": sdt_b0_wd,
                "sdt_b0_bottleneck": sdt_b0_bn,
                "sdt_b1_wasserstein": sdt_b1_wd,
                "sdt_b1_bottleneck": sdt_b1_bn,
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

    def _nanmean_col(key):
        vals = [r[key] for r in pd_distance_rows]
        return float(np.nanmean(vals)) if vals else float("nan")

    mean_pd_distances = (
        _nanmean_col("raw_b0_wasserstein"),
        _nanmean_col("raw_b0_bottleneck"),
        _nanmean_col("raw_b1_wasserstein"),
        _nanmean_col("raw_b1_bottleneck"),
        _nanmean_col("sdt_b0_wasserstein"),
        _nanmean_col("sdt_b0_bottleneck"),
        _nanmean_col("sdt_b1_wasserstein"),
        _nanmean_col("sdt_b1_bottleneck"),
    )
    return (mean_metrics_raw, mean_metrics_pp,
            raw_b0_rows, raw_b1_rows, sdt_b0_rows, sdt_b1_rows,
            pd_distance_rows, mean_pd_distances)


def get_persistence_pairs_from_filtration(
    filtration: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Compute cubical persistence pairs for a given filtration array.

    Pairs with persistence (death - birth) below *threshold* are discarded.
    """
    cc = gd.CubicalComplex(
        dimensions=filtration.shape,
        top_dimensional_cells=filtration.flatten(),
    )
    cc.compute_persistence()
    b0_pairs = [(float(b), float(d))
                for b, d in cc.persistence_intervals_in_dimension(0)
                if (float(d) - float(b)) >= threshold]
    b1_pairs = [(float(b), float(d))
                for b, d in cc.persistence_intervals_in_dimension(1)
                if (float(d) - float(b)) >= threshold]
    return b0_pairs, b1_pairs


def get_persistence_pairs(
    prob_map: torch,
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
    binary_mask:
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
    pred_finite = [(b, d) for b, d in pred_pairs if d < np.inf]
    gt_finite = [(b, d) for b, d in gt_pairs if d < np.inf]
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
