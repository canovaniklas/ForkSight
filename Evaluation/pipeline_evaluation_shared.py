import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import Segmentation.PostProcessing.segmentation_postprocessing as _postproc
import JunctionDetection.SkeletonizeDetect.segmentation_junction_detection as _jd
from evaluation_util import format_score, hard_dice_score, iou_score, hard_clDice, get_betti_at_thresholds, plot_betti_curve


PATCH_SIZE = (1024, 1024)
GRID_SIZE = (4, 4)


def show_mask(mask, ax, color: np.ndarray = None):
    if color is None:
        color = np.array([0.0, 1.0, 1.0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_images_masks_junctions(
    image: torch.Tensor,
    predicted_mask: np.ndarray,
    groundtruth_mask: np.ndarray,
    comparison_mask: np.ndarray = None,
    junction_coords: np.ndarray = None,
    skeleton: np.ndarray = None,
    ax=None,
    figsize=(10, 10),
    predicted_mask_color=None,
    groundtruth_mask_color=None,
    img_alpha=1.0,
    plot_grid: bool = True,
    prob_map: np.ndarray = None,
):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if image is not None:
        _, h, w = image.shape
        ax.imshow(image.permute(1, 2, 0).cpu().numpy(), alpha=img_alpha)
        if plot_grid:
            for i in range(1, 4):
                ax.axvline(x=i * (w / 4), color='red',
                           linestyle='-', linewidth=0.5)
                ax.axhline(y=i * (h / 4), color='red',
                           linestyle='-', linewidth=0.5)

    if predicted_mask is not None:
        show_mask(predicted_mask, ax, color=predicted_mask_color)
    if comparison_mask is not None:
        show_mask(comparison_mask, ax, color=np.array([1.0, 1.0, 0.0, 0.75]))
    if groundtruth_mask is not None:
        show_mask(
            groundtruth_mask, ax,
            color=np.array([1.0, 0.0, 0.5, 0.5]) if groundtruth_mask_color is None
            else groundtruth_mask_color,
        )
    if prob_map is not None:
        im = ax.imshow(prob_map, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Probability')

    if junction_coords is not None:
        ax.plot(junction_coords[:, 0], junction_coords[:, 1], 'ro',
                markersize=20, markerfacecolor='none', markeredgewidth=2)

    if skeleton is not None:
        skel_overlay = np.zeros((*skeleton.shape, 4), dtype=np.uint8)
        skel_overlay[skeleton > 0] = [255, 0, 0, 200]
        height, width = skeleton.shape
        ax.imshow(skel_overlay, extent=[0, width, height, 0], zorder=2)

    ax.axis('off')


def load_binary_mask_as_tensor(path: Path, size: tuple = None) -> torch.Tensor:
    """Load a binary mask image as a (1, H, W) float tensor with values 0.0/1.0.

    Handles masks stored as 0/255 (standard PNG) or 0/1 (nnUNet convention).
    Optionally resizes to *size* = (H, W) using nearest-neighbour interpolation.
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = (arr > 0).astype(np.float32)
    t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    if size is not None:
        t = F.interpolate(t, size=size, mode='nearest')
    return t.squeeze(0)  # (1, H, W)


def stitch_tiles(tiles: torch.Tensor, grid_size: tuple) -> torch.Tensor:
    """Stitch (B, C, H, W) patch tiles in row-major order into (C, rows*H, cols*W).

    Unlike ``stitch_mask_tiles`` in the postprocessing module, this function
    performs no interpolation and supports any number of channels.  It is
    intended for stitching pre-loaded image / mask tensors that are already at
    their target resolution.
    """
    B, C, H, W = tiles.shape
    rows, cols = grid_size
    assert B == rows * cols, f"Expected {rows * cols} tiles, got {B}"
    t = tiles.view(rows, cols, C, H, W)
    t = t.permute(2, 0, 3, 1, 4)   # (C, rows, H, cols, W)
    return t.reshape(C, rows * H, cols * W)


def evaluate_stitched_mask_and_plot(
    pred_stitched: torch.Tensor,
    groundtruth_mask: torch.Tensor,
    original_img: torch.Tensor | None,
    did_remove_small_objects: bool,
    ax=None,
    comparison_mask: torch.Tensor | None = None,
) -> tuple:
    """Compute metrics and plot for an already-stitched prediction mask.

    Parameters
    ----------
    pred_stitched : (1, H, W) binary prediction, already stitched and
                    post-processed (small-object removal applied upstream if
                    desired).
    groundtruth_mask : (1, H, W) binary ground-truth mask.
    original_img : (3, H, W) tensor for visualization, or None.
    did_remove_small_objects : whether small objects were removed upstream.
                               Controls junction detection and the print label.
    ax : matplotlib axes; creates a new figure if None.
    comparison_mask : (1, H, W) tensor whose foreground pixels that are absent
                      from *pred_stitched* are shown as a yellow overlay
                      (typically the small-object-removed prediction, so the
                      overlay highlights what was removed).

    Returns
    -------
    (dice, iou, clDice, tprec, tsens)
    """
    bboxes = _postproc.extract_mask_elements_bboxes(pred_stitched)

    pred_junction_coords, pred_skeleton = None, None
    if did_remove_small_objects:
        pred_junction_coords, pred_skeleton = \
            _jd.detect_junctions_in_segmentation_mask(pred_stitched)

    comparison_overlay = None
    if comparison_mask is not None:
        comparison_overlay = (comparison_mask == 1) & (pred_stitched == 0)

    missed_gt = (groundtruth_mask == 1) & (pred_stitched == 0)

    dice = hard_dice_score(pred_stitched, groundtruth_mask)
    iou = iou_score(pred_stitched, groundtruth_mask)
    pred_np = pred_stitched.squeeze(0).cpu().numpy()
    gt_np = groundtruth_mask.squeeze(0).cpu().numpy()
    clDice, tprec, tsens = hard_clDice(pred_np, gt_np)

    label = 'small objects removed' if did_remove_small_objects else 'small objects NOT removed'
    print(f"\nEvaluation Results ({label}):\n{'=' * 40}")
    print(f"Number of Objects: {len(bboxes)}")
    print(f"Missed ground-truth pixels: {missed_gt.sum().item()}")
    print(f"Hard Dice Score: {format_score(dice)}")
    print(f"IoU Score:       {format_score(iou)}")
    print(
        f"clDice / tprec / tsens: {format_score(clDice)}, {format_score(tprec)}, {format_score(tsens)}")

    if ax is not None:
        plot_images_masks_junctions(
            original_img,
            pred_stitched.numpy(),
            missed_gt.numpy(),
            comparison_mask=comparison_overlay.numpy(
            ) if comparison_overlay is not None else None,
            junction_coords=pred_junction_coords,
            skeleton=pred_skeleton,
            ax=ax,
            figsize=(10, 10),
        )

    return (dice, iou, clDice, tprec, tsens)


def evaluate_full_image_patches(
    patch_pred_masks: torch.Tensor,
    groundtruth_mask: torch.Tensor,
    original_img: torch.Tensor | None,
    patch_size: tuple = PATCH_SIZE,
    grid_size: tuple = GRID_SIZE,
    output_probs: torch.Tensor | None = None,
) -> tuple:
    """Evaluate a full image reconstructed from prediction patches.

    Stitches patches with and without small-object removal, computes metrics
    for both variants, and plots them side-by-side.  Betti curves are plotted
    using *output_probs* when provided; when None, the stitched binary
    prediction is used as a degenerate probability map (all values 0 or 1).

    Parameters
    ----------
    patch_pred_masks : (B, 1, H, W) binary prediction patches in row-major
                       patch order.
    groundtruth_mask : (1, full_H, full_W) full-resolution binary GT mask.
    original_img     : (3, full_H, full_W) full-resolution image for
                       visualization, or None.
    patch_size       : native (H, W) of each patch (used for stitching).
    grid_size        : (rows, cols) of the patch grid.
    output_probs     : (B, 1, H, W) soft probability maps from the model.
                       When None the stitched binary prediction is used for
                       the Betti curve.

    Returns
    -------
    (result_with_removal, result_without_removal)
    Each result is a (dice, iou, clDice, tprec, tsens) tuple.
    """
    pred_cleaned, _ = _postproc.postprocess_segmentation_masks(
        patch_pred_masks, grid_size=grid_size,
        original_input_patch_img_size=patch_size, remove_small_objects=True)
    pred_raw, _ = _postproc.postprocess_segmentation_masks(
        patch_pred_masks, grid_size=grid_size,
        original_input_patch_img_size=patch_size, remove_small_objects=False)

    pred_cleaned = pred_cleaned.detach().cpu()
    pred_raw = pred_raw.detach().cpu()

    _, axes = plt.subplots(1, 2, figsize=(14, 7))

    result_1 = evaluate_stitched_mask_and_plot(
        pred_cleaned, groundtruth_mask, original_img,
        did_remove_small_objects=True, ax=axes[0])

    result_2 = evaluate_stitched_mask_and_plot(
        pred_raw, groundtruth_mask, original_img,
        did_remove_small_objects=False, ax=axes[1], comparison_mask=pred_cleaned)

    plt.tight_layout()
    plt.show()
    plt.close()

    # Betti curves
    if output_probs is not None:
        probs_stitched = _postproc.stitch_mask_tiles(
            output_probs, grid_size=grid_size,
            original_input_patch_img_size=patch_size, as_uint=False)
        probs_np = probs_stitched.squeeze(0).detach().cpu().numpy()
    else:
        # Use the binary stitched prediction; Betti numbers are constant for
        # all thresholds in (0, 1] — still a valid (hard) topological summary.
        probs_np = pred_cleaned.squeeze(0).cpu().numpy().astype(float)

    gt_np = groundtruth_mask.squeeze(0).cpu().numpy()
    gt_b0, gt_b1 = get_betti_at_thresholds(gt_np, [1.0])
    gt_b0, gt_b1 = gt_b0[0], gt_b1[0]
    print(f"Groundtruth Betti numbers: B0={gt_b0}, B1={gt_b1}")

    thresholds = np.linspace(0.0, 1.0, 11)
    pred_b0, pred_b1 = get_betti_at_thresholds(probs_np, thresholds)
    plot_betti_curve(thresholds, pred_b0, pred_b1, gt_b0, gt_b1)

    return result_1, result_2


def evaluate_soi_patch(
    pred_mask: torch.Tensor,
    img_tensor: torch.Tensor,
    groundtruth_mask: torch.Tensor,
) -> None:
    """Evaluate a single SOI (sub-image of interest) patch.

    Applies small-object removal, detects junctions, plots the result.

    Parameters
    ----------
    pred_mask        : (1, H, W) binary prediction mask.
    img_tensor       : (3, H, W) input image for visualization.
    groundtruth_mask : (1, H, W) binary ground-truth mask.
    """
    cleaned = _postproc.remove_small_objects_from_batch(
        pred_mask.unsqueeze(0)).squeeze(0).detach().cpu()  # (1, H, W)

    pred_junction_coords, pred_skeleton = \
        _jd.detect_junctions_in_segmentation_mask(cleaned)

    comparison_mask = (pred_mask == 1) & (cleaned == 0)
    missed_gt = (groundtruth_mask == 1) & (cleaned == 0)

    plot_images_masks_junctions(
        img_tensor, cleaned.numpy(), missed_gt.numpy(),
        comparison_mask=comparison_mask.numpy(),
        junction_coords=pred_junction_coords,
        skeleton=pred_skeleton,
        figsize=(6, 6), plot_grid=False,
    )
