import os
from pathlib import Path
import re
from typing import Any, Mapping
from segment_anything import sam_model_registry
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import wandb
from skimage.morphology import skeletonize, dilation, disk
import numpy as np

from Segmentation.SAM.sam_lora import SamLoRA
from Segmentation.Util.env_utils import load_segmentation_env

load_segmentation_env()

MODEL_CHECKPOINTS_DIR = os.getenv("MODEL_CHECKPOINTS_DIR")

EVALUATED_TAG = "test-evaluated"


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, heatmaps_dir: Path | None = None):
        self.image_paths = list(images_dir.glob("*.png"))
        self.masks_dir = masks_dir
        self.heatmaps_dir = heatmaps_dir

    def _load_image(self, path: Path, is_mask: bool = False) -> torch.Tensor:
        transform = transforms.Compose([
            # resize to 1024x1024 size because a) there are random crops of different sizes, b) SAM model was trained on 1024x1024 images and performs best at that size
            # using nearest neighbor interpolation for masks to preserve label values (no interpolation)
            transforms.Resize((1024, 1024), interpolation=(
                transforms.InterpolationMode.NEAREST if is_mask else transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()
        ])

        if not is_mask:
            transform.transforms.append(transforms.Lambda(
                lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t))

        img = Image.open(path)
        return transform(img)

    def _load_heatmap(self, image_path: Path) -> torch.Tensor:
        heatmap_path = self.heatmaps_dir / f"{image_path.stem}.npy"
        heatmap = torch.from_numpy(
            np.load(heatmap_path).astype(np.float32)).unsqueeze(0)
        return F.interpolate(heatmap.unsqueeze(0), size=(1024, 1024),
                             mode='bilinear', align_corners=False).squeeze(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.masks_dir / image_path.name

        image = self._load_image(image_path)
        mask = self._load_image(mask_path, is_mask=True)
        heatmap = self._load_heatmap(
            image_path) if self.heatmaps_dir else torch.zeros_like(mask)

        return image, mask, heatmap


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, heatmap_weight_scale: float = 1.0, junction_boost: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.heatmap_weight_scale = heatmap_weight_scale
        self.junction_boost = junction_boost

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                heatmap_weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) - predicted logits
            targets: (B, 1, H, W) - ground truth masks
            heatmap_weights: (B, 1, H, W) - junction importance weights [0, 1], or None

        Returns:
            Scalar loss value
        """
        pixel_bce = self.bce(logits, targets)
        pixel_bce_flat = pixel_bce.view(pixel_bce.shape[0], -1)

        if heatmap_weights is not None:
            heatmap_flat = heatmap_weights.view(heatmap_weights.shape[0], -1)

            # Pixel-level weighting: junction pixels get higher weight
            w = (1.0 + self.heatmap_weight_scale * heatmap_flat)
            sample_bce = (pixel_bce_flat * w).mean(dim=1)

            # Sample-level boost: images with more junctions get higher loss
            #junction_density = heatmap_flat.mean(dim=1)
            #junction_boost_factor = 1.0 + self.junction_boost * junction_density
            #sample_bce = weighted_bce * junction_boost_factor
        else:
            sample_bce = pixel_bce_flat.mean(dim=1)

        return sample_bce.mean()


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        masks_prob = torch.sigmoid(logits)

        masks_flat = masks_prob.view(masks_prob.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        intersection = (masks_flat * targets_flat).sum(dim=1)
        cardinality = masks_flat.sum(dim=1) + targets_flat.sum(dim=1)

        return (1 - (2 * intersection + 1e-6) / (cardinality + 1e-6)).mean()


class SoftSkeletonize(torch.nn.Module):
    def __init__(self, num_iter: int):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):
        # input img shape: (B, C, H, W), where C = 1 for binary masks (foreground only)
        return self.soft_skel(img)


class SoftClDiceLoss(nn.Module):
    def __init__(self, skeletonize_iter: int, smooth: float = 1e-6):
        super(SoftClDiceLoss, self).__init__()
        self.iter = skeletonize_iter
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=skeletonize_iter)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        mask_prob = torch.sigmoid(logits)

        skel_pred = self.soft_skeletonize(mask_prob)
        skel_true = self.soft_skeletonize(targets)

        tprec = (torch.sum(skel_pred * targets) +
                 self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * mask_prob) +
                 self.smooth) / (torch.sum(skel_true) + self.smooth)

        return 1. - 2.0 * (tprec * tsens) / (tprec + tsens)


class JunctionPatchLoss(nn.Module):
    def __init__(self, patch_size: int = 64, loss_type: str = "cldice", skeletonize_iter: int = 15):
        """
        Loss computed on patches around junction centers.

        Args:
            patch_size: Size of square patch around each junction (default 64)
            loss_type: Type of loss to compute on patches: "cldice", "dice", or "focal" (default "cldice")
            skeletonize_iter: Number of iterations for soft skeletonization if using cldice
        """
        super(JunctionPatchLoss, self).__init__()
        self.patch_size = patch_size
        self.loss_type = loss_type

        if loss_type == "cldice":
            self.loss_fn = SoftClDiceLoss(skeletonize_iter)
        elif loss_type == "dice":
            self.loss_fn = SoftDiceLoss()
        elif loss_type == "focal":
            # Focal loss with alpha=0.25, gamma=2.0
            self.focal_alpha = 0.25
            self.focal_gamma = 2.0
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def _extract_patch(self, tensor: torch.Tensor, center_y: int, center_x: int, h: int, w: int) -> torch.Tensor:
        """Extract a patch around a center point, handling boundary cases."""
        half_patch = self.patch_size // 2

        y_start = max(0, center_y - half_patch)
        y_end = min(h, center_y + half_patch)
        x_start = max(0, center_x - half_patch)
        x_end = min(w, center_x + half_patch)

        return tensor[:, :, y_start:y_end, x_start:x_end]

    def _compute_focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for a patch."""
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.focal_gamma

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss.mean()

    def _find_junction_centers(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Find junction centers by identifying pixels with maximum heatmap value.
        If heatmap is all zeros (no junctions), returns empty tensor.

        Args:
            heatmap: (H, W) - 2D heatmap

        Returns:
            Tensor of (N, 2) coordinates (y, x) of junction centers
        """
        max_val = heatmap.max()

        # If heatmap is all zeros, no junctions present
        if max_val == 0:
            return torch.empty((0, 2), dtype=torch.long, device=heatmap.device)

        # Find all pixels at maximum value (the Gaussian peaks)
        is_peak = heatmap == max_val
        junction_coords = torch.nonzero(is_peak, as_tuple=False)  # (N, 2) of (y, x)
        return junction_coords

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, heatmap_weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) - predicted logits
            targets: (B, 1, H, W) - ground truth masks
            heatmap_weights: (B, 1, H, W) - junction heatmaps, or None

        Returns:
            Mean loss across all detected junction patches, or 0 if no junctions found
        """
        if heatmap_weights is None:
            return torch.tensor(0.0, device=logits.device)

        batch_size, _, h, w = logits.shape
        all_patch_losses = []

        for b in range(batch_size):
            heatmap = heatmap_weights[b, 0]  # (H, W)

            # Find junction centers (pixels at maximum heatmap value)
            junction_coords = self._find_junction_centers(heatmap)

            if junction_coords.shape[0] == 0:
                continue  # No junctions in this sample

            # Extract patches around each junction center and compute loss
            for coord in junction_coords:
                center_y, center_x = coord[0].item(), coord[1].item()

                logits_patch = self._extract_patch(logits[b:b+1], center_y, center_x, h, w)
                targets_patch = self._extract_patch(targets[b:b+1], center_y, center_x, h, w)

                # Skip if patch is too small (near boundaries)
                if logits_patch.shape[2] < 8 or logits_patch.shape[3] < 8:
                    continue

                if self.loss_type == "focal":
                    patch_loss = self._compute_focal_loss(logits_patch, targets_patch)
                else:
                    patch_loss = self.loss_fn(logits_patch, targets_patch)

                all_patch_losses.append(patch_loss)

        if len(all_patch_losses) == 0:
            return torch.tensor(0.0, device=logits.device)

        return torch.stack(all_patch_losses).mean()


class ClDiceDiceBCELoss(nn.Module):
    def __init__(self, skeletonize_iter: int, cl_dice_weight: float, dice_weight: float,
                 upsample_lowres_logits: tuple[int, int] | None = None,
                 heatmap_weight_scale: float = 1.0, junction_boost: float = 0.5,
                 use_junction_patch_loss: bool = False, junction_patch_weight: float = 0.0,
                 junction_patch_size: int = 64,
                 junction_loss_type: str = "cldice"):
        """
        Combined centerline Dice + Dice + BCE loss with optional junction patch loss.

        Args:
            skeletonize_iter: Number of iterations for soft skeletonization
            cl_dice_weight: Weight for centerline Dice loss
            dice_weight: Weight for standard Dice loss
            upsample_lowres_logits: Optional (H, W) to upsample logits before loss computation
            heatmap_weight_scale: Pixel-level junction weighting (default 1.0)
            junction_boost: Sample-level junction boost (default 0.5)
            use_junction_patch_loss: Whether to add junction patch loss (default False)
            junction_patch_weight: Weight for junction patch loss term (default 0.0)
            junction_patch_size: Size of patches around junctions (default 64)
            junction_loss_type: Loss type for patches: "cldice", "dice", or "focal" (default "cldice")
        """
        super(ClDiceDiceBCELoss, self).__init__()

        self.cl_dice_loss = SoftClDiceLoss(skeletonize_iter)
        self.dice_loss = SoftDiceLoss()
        self.bce_loss = BCEWithLogitsLoss(
            heatmap_weight_scale=heatmap_weight_scale,
            junction_boost=junction_boost
        )

        self.use_junction_patch_loss = use_junction_patch_loss
        self.junction_patch_weight = junction_patch_weight
        if use_junction_patch_loss:
            self.junction_patch_loss = JunctionPatchLoss(
                patch_size=junction_patch_size,
                loss_type=junction_loss_type,
                skeletonize_iter=skeletonize_iter
            )

        assert cl_dice_weight + dice_weight <= 1.0, \
            "Sum of cl_dice_weight and dice_weight must be less than or equal to 1.0"
        assert cl_dice_weight >= 0.0 and dice_weight >= 0.0, "Weights must be non-negative"

        self.cl_dice_weight = cl_dice_weight
        self.dice_weight = dice_weight
        self.upsample_lowres_logits = upsample_lowres_logits

    def forward(self, low_res_logits: torch.Tensor, targets: torch.Tensor,
                heatmap_weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            low_res_logits: predicted low-res mask logits, shape (B, 1, H_lowres, W_lowres)
            targets: binary 0/1, shape (B, 1, H_original, W_original)
            heatmap_weights: junction Gaussian weights, shape (B, 1, H_original, W_original), or None

        Returns:
            Combined loss value
        """
        if self.upsample_lowres_logits is not None:
            low_res_logits = F.interpolate(
                low_res_logits, size=self.upsample_lowres_logits,
                mode='bilinear', align_corners=False)

        logit_spatial = (low_res_logits.shape[-2], low_res_logits.shape[-1])

        targets = targets.to(device=low_res_logits.device, dtype=torch.float32)
        targets_resized = F.interpolate(
            targets, size=logit_spatial, mode='nearest')

        heatmap_weights_resized = None
        if heatmap_weights is not None:
            heatmap_weights_resized = F.interpolate(
                heatmap_weights.to(
                    device=low_res_logits.device, dtype=torch.float32),
                size=logit_spatial, mode='bilinear', align_corners=False)

        cl_dice_loss_value = self.cl_dice_loss(low_res_logits, targets_resized)
        dice_loss_value = self.dice_loss(low_res_logits, targets_resized)
        bce_loss_value = self.bce_loss(
            low_res_logits, targets_resized, heatmap_weights_resized)

        base_bce_weight = 1 - self.cl_dice_weight - self.dice_weight
        total_loss = (self.cl_dice_weight * cl_dice_loss_value) + \
            (self.dice_weight * dice_loss_value) + \
            (base_bce_weight * bce_loss_value)

        # Add junction patch loss if enabled
        if self.use_junction_patch_loss and heatmap_weights_resized is not None:
            junction_patch_loss_value = self.junction_patch_loss(
                low_res_logits, targets_resized, heatmap_weights_resized)
            total_loss = total_loss + self.junction_patch_weight * junction_patch_loss_value

        return total_loss


class SkeletonRecallLoss2D(nn.Module):
    def __init__(self):
        super(SkeletonRecallLoss2D, self).__init__()

    def _skeletonize_sample(self, mask: torch.Tensor) -> torch.Tensor:
        device = mask.device

        mask_np = mask.detach().cpu().squeeze().numpy().astype(np.uint8)
        binary_mask = (mask_np > 0).astype(np.uint8)
        skeleton = skeletonize(binary_mask).astype(np.uint8)
        tubed_skeleton = dilation(skeleton, footprint=disk(2))

        # refine with original mask: ensure the tubed skeleton doesn't 'leak' outside the actual structure
        final_skeleton = tubed_skeleton.astype(
            np.float32) * mask_np.astype(np.float32)

        return torch.from_numpy(final_skeleton).unsqueeze(0).to(device)

    def forward(self, logits: torch.Tensor, ground_truth_masks: torch.Tensor):
        """
        logits: (B, C, H, W) - segmentation network output logits
        ground_truth_masks: (B, C, H, W) - ground truth segmentation mask (0 or 1)
        C = 1 for binary segmentation in our case
        """
        probs = torch.sigmoid(logits)

        skeleton_list = [self._skeletonize_sample(ground_truth_masks[i])
                         for i in range(ground_truth_masks.shape[0])]
        skeletons = torch.stack(skeleton_list)

        intersection = (probs * skeletons).sum(dim=(2, 3))
        ground_truth_sum = skeletons.sum(dim=(2, 3))

        recall = (intersection + 1e-6) / (ground_truth_sum + 1e-6)
        loss = 1.0 - recall

        return loss.mean()


class SkeletonRecallDiceBCELoss(nn.Module):
    def __init__(self, skeleton_recall_weight: float, dice_weight: float,
                 upsample_lowres_logits: tuple[int, int] | None = None,
                 heatmap_weight_scale: float = 1.0, junction_boost: float = 0.5,
                 use_junction_patch_loss: bool = False, junction_patch_weight: float = 0.0,
                 junction_patch_size: int = 64,
                 junction_loss_type: str = "cldice", skeletonize_iter: int = 15):
        """
        Combined skeleton recall + Dice + BCE loss with optional junction patch loss.

        Args:
            skeleton_recall_weight: Weight for skeleton recall loss
            dice_weight: Weight for standard Dice loss
            upsample_lowres_logits: Optional (H, W) to upsample logits before loss computation
            heatmap_weight_scale: Pixel-level junction weighting (default 1.0)
            junction_boost: Sample-level junction boost (default 0.5)
            use_junction_patch_loss: Whether to add junction patch loss (default False)
            junction_patch_weight: Weight for junction patch loss term (default 0.0)
            junction_patch_size: Size of patches around junctions (default 64)
            junction_loss_type: Loss type for patches: "cldice", "dice", or "focal" (default "cldice")
            skeletonize_iter: Number of iterations for soft skeletonization if using cldice
        """
        super(SkeletonRecallDiceBCELoss, self).__init__()

        self.skeleton_recall_loss = SkeletonRecallLoss2D()
        self.dice_loss = SoftDiceLoss()
        self.bce_loss = BCEWithLogitsLoss(
            heatmap_weight_scale=heatmap_weight_scale,
            junction_boost=junction_boost
        )

        self.use_junction_patch_loss = use_junction_patch_loss
        self.junction_patch_weight = junction_patch_weight
        if use_junction_patch_loss:
            self.junction_patch_loss = JunctionPatchLoss(
                patch_size=junction_patch_size,
                loss_type=junction_loss_type,
                skeletonize_iter=skeletonize_iter
            )

        assert skeleton_recall_weight + dice_weight <= 1.0, \
            "Sum of skeleton_recall_weight and dice_weight must be less than or equal to 1.0"
        assert skeleton_recall_weight >= 0.0 and dice_weight >= 0.0, \
            "Weights must be non-negative"

        self.skeleton_recall_weight = skeleton_recall_weight
        self.dice_weight = dice_weight
        self.upsample_lowres_logits = upsample_lowres_logits

    def forward(self, low_res_logits: torch.Tensor, ground_truth_masks: torch.Tensor,
                heatmap_weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            low_res_logits: predicted low-res mask logits, shape (B, 1, H_lowres, W_lowres)
            ground_truth_masks: binary 0/1, shape (B, 1, H_original, W_original)
            heatmap_weights: junction Gaussian weights, shape (B, 1, H_original, W_original), or None

        Returns:
            Combined loss value
        """
        if self.upsample_lowres_logits is not None:
            low_res_logits = F.interpolate(
                low_res_logits, size=self.upsample_lowres_logits,
                mode='bilinear', align_corners=False)

        logit_spatial = (low_res_logits.shape[-2], low_res_logits.shape[-1])

        ground_truth_masks = ground_truth_masks.to(
            device=low_res_logits.device, dtype=torch.float32)
        ground_truth_masks_resized = F.interpolate(
            ground_truth_masks, size=logit_spatial, mode='nearest')

        # bilinear for heatmap: continuous field, not discrete labels
        heatmap_weights_resized = None
        if heatmap_weights is not None:
            heatmap_weights_resized = F.interpolate(
                heatmap_weights.to(
                    device=low_res_logits.device, dtype=torch.float32),
                size=logit_spatial, mode='bilinear', align_corners=False)

        skeleton_recall_loss_value = self.skeleton_recall_loss(
            low_res_logits, ground_truth_masks_resized)
        dice_loss_value = self.dice_loss(
            low_res_logits, ground_truth_masks_resized)
        bce_loss_value = self.bce_loss(
            low_res_logits, ground_truth_masks_resized, heatmap_weights_resized)

        base_bce_weight = 1 - self.skeleton_recall_weight - self.dice_weight
        total_loss = (self.skeleton_recall_weight * skeleton_recall_loss_value) + \
            (self.dice_weight * dice_loss_value) + \
            (base_bce_weight * bce_loss_value)

        # Add junction patch loss if enabled
        if self.use_junction_patch_loss and heatmap_weights_resized is not None:
            junction_patch_loss_value = self.junction_patch_loss(
                low_res_logits, ground_truth_masks_resized, heatmap_weights_resized)
            total_loss = total_loss + self.junction_patch_weight * junction_patch_loss_value

        return total_loss


def hard_dice_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    return (2 * intersection + 1e-6) / (union + 1e-6)


def hard_clDice(mask_predicted, mask_target):
    def cl_score(img, skeleton):
        return (np.sum(img * skeleton) + 1e-6) / (np.sum(skeleton) + 1e-6)

    tprec = cl_score(mask_target, skeletonize(mask_predicted))
    tsens = cl_score(mask_predicted, skeletonize(mask_target))
    cl_dice = (2 * tprec * tsens + 1e-6) / (tprec + tsens + 1e-6)

    return cl_dice, tprec, tsens


def iou_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def get_batched_input_list(batched_input: torch.Tensor):
    return [{
        "image": img,
        "original_size": (img.shape[1], img.shape[2])
    } for img in batched_input.unbind(0)]


@torch.no_grad()
def evaluate_model(model: SamLoRA, test_imgs_dir: Path, test_masks_dir: Path, device: torch.device, model_params_name: str,
                   cl_dice_skeletonize_iter: int, cl_dice_weight: float, dice_weight: float):
    model.eval()
    model.to(device)

    dataset = SegmentationDataset(test_imgs_dir, test_masks_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,)

    bce_with_logits_dice_loss = ClDiceDiceBCELoss(
        skeletonize_iter=cl_dice_skeletonize_iter, cl_dice_weight=cl_dice_weight, dice_weight=dice_weight)

    tprec_scores = []
    tsens_scores = []
    hard_clDice_scores = []
    hard_dice_scores = []
    iou_scores = []
    losses = []

    for image, mask, _ in dataloader:
        image = image.to(device)
        mask = mask.to(device)

        batched_input = get_batched_input_list(image)
        outputs = model(batched_input, multimask_output=False)

        output_logits = torch.cat([d["low_res_logits"]
                                   for d in outputs], dim=0)
        output_mask = outputs[0]['masks'].squeeze(0)

        losses.append(bce_with_logits_dice_loss(output_logits, mask).item())
        hard_dice_scores.append(hard_dice_score(output_mask, mask).item())
        iou_scores.append(iou_score(output_mask, mask).item())

        output_mask_np = output_mask.squeeze(0).cpu().numpy()
        mask_np = mask.squeeze(0).cpu().numpy()

        cl_dice_score, tprec, tsens = hard_clDice(output_mask_np, mask_np)
        hard_clDice_scores.append(cl_dice_score)
        tprec_scores.append(tprec)
        tsens_scores.append(tsens)

    metrics = {
        f"test/{model_params_name}/mean_bce_dice_loss": sum(losses) / len(losses),
        f"test/{model_params_name}/mean_dice_score": sum(hard_dice_scores) / len(hard_dice_scores),
        f"test/{model_params_name}/mean_iou_score": sum(iou_scores) / len(iou_scores),
        f"test/{model_params_name}/mean_clDice_score": sum(hard_clDice_scores) / len(hard_clDice_scores),
        f"test/{model_params_name}/mean_tprec": sum(tprec_scores) / len(tprec_scores),
        f"test/{model_params_name}/mean_tsens": sum(tsens_scores) / len(tsens_scores),
    }

    return (metrics)


def initialize_sam_lora_with_params(wandb_run_config: dict[str, Any], params: Mapping[str, Any], device: torch.device) -> SamLoRA:
    sam_checkpoint = str(Path(MODEL_CHECKPOINTS_DIR) /
                         f"{wandb_run_config['SAM_checkpoint']}.pth")
    sam = sam_model_registry[wandb_run_config["SAM_model_type"]](
        checkpoint=sam_checkpoint)

    sam.to(device)

    finetune_img_encoder = "image_encoder" in wandb_run_config["finetuned_modules"]
    finetune_mask_decoder = "mask_decoder" in wandb_run_config["finetuned_modules"]
    finetune_prompt_encoder = "prompt_encoder" in wandb_run_config["finetuned_modules"]

    sam_lora = SamLoRA(sam, r=wandb_run_config["LoRA_rank"], finetune_img_encoder=finetune_img_encoder,
                       finetune_mask_decoder=finetune_mask_decoder, finetune_prompt_encoder=finetune_prompt_encoder)
    sam_lora.to(device)

    sam_lora.load_state_dict(params, strict=False)

    return sam_lora


def get_params_from_artifact(artifact: wandb.Artifact, device: torch.device):
    pattern = re.compile(r"(params.*)")
    match = pattern.search(artifact.name)
    if match is None:
        return None, None

    artifact_dir = artifact.download()
    ckpt_path = next(Path(artifact_dir).glob("*.pt"))
    params = torch.load(ckpt_path, map_location=device)

    return params, match.group(1).replace(":v0", "")
