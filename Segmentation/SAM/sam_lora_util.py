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
    def __init__(self, heatmap_weight_scale: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.heatmap_weight_scale = heatmap_weight_scale

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                heatmap_weights: torch.Tensor | None = None,
                pixel_mask: torch.Tensor | None = None) -> torch.Tensor:
        pixel_bce = self.bce(logits, targets)

        base_pixel_bce_flat = pixel_bce.view(pixel_bce.shape[0], -1)
        total_pixel_bce_flat = base_pixel_bce_flat

        heatmap_weighted_pixel_bce_flat = torch.zeros_like(base_pixel_bce_flat)
        if heatmap_weights is not None:
            heatmap_flat = heatmap_weights.view(heatmap_weights.shape[0], -1)
            heatmap_weighted_pixel_bce_flat = base_pixel_bce_flat * \
                (self.heatmap_weight_scale * heatmap_flat)
            total_pixel_bce_flat = base_pixel_bce_flat + heatmap_weighted_pixel_bce_flat

        if pixel_mask is not None:
            mask_flat = pixel_mask.view(pixel_mask.shape[0], -1)
            divider = mask_flat.sum(dim=1).clamp(min=1)
            sample_bce = (total_pixel_bce_flat *
                          mask_flat).sum(dim=1) / divider
            sample_base_bce = (base_pixel_bce_flat *
                               mask_flat).sum(dim=1) / divider
            sample_heatmap_bce = (
                heatmap_weighted_pixel_bce_flat * mask_flat).sum(dim=1) / divider
        else:
            sample_bce = total_pixel_bce_flat.mean(dim=1)
            sample_base_bce = base_pixel_bce_flat.mean(dim=1)
            sample_heatmap_bce = heatmap_weighted_pixel_bce_flat.mean(dim=1)

        return sample_bce.mean(), sample_base_bce.mean(), sample_heatmap_bce.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 heatmap_weight_scale: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.heatmap_weight_scale = heatmap_weight_scale
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                heatmap_weights: torch.Tensor | None = None,
                pixel_mask: torch.Tensor | None = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        pixel_bce = self.bce(logits, targets)

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        pixel_focal = alpha_t * focal_weight * pixel_bce

        base_pixel_focalloss_flat = pixel_focal.view(pixel_focal.shape[0], -1)
        total_pixel_focalloss_flat = base_pixel_focalloss_flat

        heatmap_weighted_pixel_focalloss_flat = torch.zeros_like(
            base_pixel_focalloss_flat)
        if heatmap_weights is not None:
            heatmap_flat = heatmap_weights.view(heatmap_weights.shape[0], -1)
            heatmap_weighted_pixel_focalloss_flat = base_pixel_focalloss_flat * \
                (self.heatmap_weight_scale * heatmap_flat)
            total_pixel_focalloss_flat = base_pixel_focalloss_flat + \
                heatmap_weighted_pixel_focalloss_flat

        if pixel_mask is not None:
            mask_flat = pixel_mask.view(pixel_mask.shape[0], -1)
            divider = mask_flat.sum(dim=1).clamp(min=1)
            sample_focalloss = (total_pixel_focalloss_flat *
                                mask_flat).sum(dim=1) / divider
            sample_base_focalloss = (
                base_pixel_focalloss_flat * mask_flat).sum(dim=1) / divider
            sample_heatmap_focalloss = (
                heatmap_weighted_pixel_focalloss_flat * mask_flat).sum(dim=1) / divider
        else:
            sample_focalloss = total_pixel_focalloss_flat.mean(dim=1)
            sample_base_focalloss = base_pixel_focalloss_flat.mean(dim=1)
            sample_heatmap_focalloss = heatmap_weighted_pixel_focalloss_flat.mean(
                dim=1)

        return sample_focalloss.mean(), sample_base_focalloss.mean(), sample_heatmap_focalloss.mean()


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


class JunctionRegionLoss(nn.Module):
    def __init__(self, loss_type: str = "skeleton_recall", skeletonize_iter: int = 15,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """
        Loss computed only on junction regions (non-zero heatmap pixels).

        Args:
            loss_type: "cldice", "dice", "focal", or "skeleton_recall"
            skeletonize_iter: Iterations for soft skeletonization (cldice)
            focal_alpha: Alpha for focal loss
            focal_gamma: Gamma for focal loss
        """
        super(JunctionRegionLoss, self).__init__()
        self.loss_type = loss_type

        if loss_type == "cldice":
            self.loss_fn = SoftClDiceLoss(skeletonize_iter)
        elif loss_type == "dice":
            self.loss_fn = SoftDiceLoss()
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_type == "skeleton_recall":
            self.loss_fn = SkeletonRecallLoss2D()
        elif loss_type == "bce":
            self.loss_fn = BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                heatmap_weights: torch.Tensor | None = None) -> torch.Tensor:
        if heatmap_weights is None:
            return torch.tensor(0.0, device=logits.device)

        junction_mask = (heatmap_weights > 0).float()
        if junction_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        if self.loss_type == "focal" or self.loss_type == "bce":
            # Pixel-level loss: average only over junction pixels via pixel_mask
            total_loss, base_loss, heatmap_loss = self.loss_fn(
                logits, targets, pixel_mask=junction_mask)
            return total_loss
        else:
            # Ratio-based losses (dice, cldice, skeleton_recall):
            # mask logits to large negative outside junctions (sigmoid → 0)
            masked_logits = logits * junction_mask + \
                (-1e4) * (1 - junction_mask)
            masked_targets = targets * junction_mask
            return self.loss_fn(masked_logits, masked_targets)


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


class CombinedLoss(nn.Module):
    def __init__(self,
                 bce_weight: float = 0.0,
                 focal_weight: float = 0.0,
                 dice_weight: float = 0.0,
                 cl_dice_weight: float = 0.0,
                 skeleton_recall_weight: float = 0.0,
                 heatmap_weight_scale: float = 1.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 skeletonize_iter: int = 15,
                 junction_patch_weight: float = 0.0,
                 junction_loss_type: str = "skeleton_recall"):
        super(CombinedLoss, self).__init__()

        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.cl_dice_weight = cl_dice_weight
        self.skeleton_recall_weight = skeleton_recall_weight

        if bce_weight > 0:
            self.bce_loss = BCEWithLogitsLoss(
                heatmap_weight_scale=heatmap_weight_scale)
        if focal_weight > 0:
            self.focal_loss = FocalLoss(
                alpha=focal_alpha, gamma=focal_gamma,
                heatmap_weight_scale=heatmap_weight_scale)
        if dice_weight > 0:
            self.dice_loss = SoftDiceLoss()
        if cl_dice_weight > 0:
            self.cl_dice_loss = SoftClDiceLoss(skeletonize_iter)
        if skeleton_recall_weight > 0:
            self.skeleton_recall_loss = SkeletonRecallLoss2D()

        self.junction_patch_weight = junction_patch_weight
        self.junction_region_loss = None
        if junction_loss_type is not None and junction_patch_weight > 0:
            self.junction_region_loss = JunctionRegionLoss(
                loss_type=junction_loss_type,
                skeletonize_iter=skeletonize_iter,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma)

    def forward(self, low_res_logits: torch.Tensor, targets: torch.Tensor,
                heatmap_weights: torch.Tensor | None = None) -> torch.Tensor:
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

        total_loss = torch.tensor(0.0, device=low_res_logits.device)
        bce_total, bce_base, bce_heatmap_weighted = torch.tensor(0.0, device=low_res_logits.device), torch.tensor(
            0.0, device=low_res_logits.device), torch.tensor(0.0, device=low_res_logits.device)
        focal_loss_total, focal_loss_base, focal_loss_heatmap_weighted = torch.tensor(0.0, device=low_res_logits.device), torch.tensor(
            0.0, device=low_res_logits.device), torch.tensor(0.0, device=low_res_logits.device)
        dice_loss = torch.tensor(0.0, device=low_res_logits.device)
        cl_dice_loss = torch.tensor(0.0, device=low_res_logits.device)
        skeleton_recall_loss = torch.tensor(0.0, device=low_res_logits.device)
        junction_loss = torch.tensor(0.0, device=low_res_logits.device)

        if self.bce_weight > 0:
            bce_total, bce_base, bce_heatmap_weighted = self.bce_loss(
                low_res_logits, targets_resized, heatmap_weights_resized)
            bce_total = self.bce_weight * bce_total
            bce_base = self.bce_weight * bce_base
            bce_heatmap_weighted = self.bce_weight * bce_heatmap_weighted
            total_loss = total_loss + bce_total
        if self.focal_weight > 0:
            focal_loss_total, focal_loss_base, focal_loss_heatmap_weighted = self.focal_loss(
                low_res_logits, targets_resized, heatmap_weights_resized)
            focal_loss_total = self.focal_weight * focal_loss_total
            focal_loss_base = self.focal_weight * focal_loss_base
            focal_loss_heatmap_weighted = self.focal_weight * focal_loss_heatmap_weighted
            total_loss = total_loss + focal_loss_total
        if self.dice_weight > 0:
            dice_loss = self.dice_weight * self.dice_loss(
                low_res_logits, targets_resized)
            total_loss = total_loss + dice_loss
        if self.cl_dice_weight > 0:
            cl_dice_loss = self.cl_dice_weight * self.cl_dice_loss(
                low_res_logits, targets_resized)
            total_loss = total_loss + cl_dice_loss
        if self.skeleton_recall_weight > 0:
            skeleton_recall_loss = self.skeleton_recall_weight * self.skeleton_recall_loss(
                low_res_logits, targets_resized)
            total_loss = total_loss + skeleton_recall_loss
        if self.junction_region_loss is not None and heatmap_weights_resized is not None:
            junction_loss = self.junction_patch_weight * self.junction_region_loss(
                low_res_logits, targets_resized, heatmap_weights_resized)
            total_loss = total_loss + junction_loss

        return total_loss, bce_total, bce_base, bce_heatmap_weighted, focal_loss_total, focal_loss_base, focal_loss_heatmap_weighted, \
            dice_loss, cl_dice_loss, skeleton_recall_loss, junction_loss


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
                   bce_weight: float = 0.0, focal_weight: float = 0.0, dice_weight: float = 0.0,
                   cl_dice_weight: float = 0.0, skeleton_recall_weight: float = 0.0,
                   focal_alpha: float = 0.25, focal_gamma: float = 2.0, skeletonize_iter: int = 15):
    model.eval()
    model.to(device)

    dataset = SegmentationDataset(test_imgs_dir, test_masks_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,)

    eval_loss_fn = CombinedLoss(
        bce_weight=bce_weight, focal_weight=focal_weight, dice_weight=dice_weight,
        cl_dice_weight=cl_dice_weight, skeleton_recall_weight=skeleton_recall_weight,
        focal_alpha=focal_alpha, focal_gamma=focal_gamma, skeletonize_iter=skeletonize_iter)

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

        total_loss, _, _, _, _, _, _, _, _, _, _ = eval_loss_fn(
            output_logits, mask)
        losses.append(total_loss.item())
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
