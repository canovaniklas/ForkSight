from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

from Segmentation.SAM.sam_lora import SamLoRA


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path):
        self.image_paths = list(images_dir.glob("*.png"))
        self.masks_dir = masks_dir

    def _load_image(self, path: Path, is_mask: bool = False) -> torch.Tensor:
        transform = transforms.Compose([
            # using nearest neighbor interpolation for masks to preserve label values (no interpolation)
            # transforms.Resize((1024, 1024), interpolation=(
            #    transforms.InterpolationMode.NEAREST if is_mask else transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()
        ])

        if not is_mask:
            transform.transforms.append(transforms.Lambda(
                lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t))

        img = Image.open(path)
        return transform(img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.masks_dir / image_path.name

        image = self._load_image(image_path)
        mask = self._load_image(mask_path, is_mask=True)

        return image, mask


class BCEWithLogitsDiceLoss(nn.Module):
    """
    Combined BCEWithLogits + Soft Dice loss for binary segmentation.

    Args:
      bce_weight: weight for the BCE loss term.
      dice_weight: weight for the Dice loss term.
      smooth: smoothing constant to avoid division by zero.
      reduction: 'mean' or 'sum' for reduction over batch for final loss.
    """

    def __init__(self, dice_weight: float = 0.8, upsample_lowres_logits=None):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_weight = dice_weight
        self.upsample_lowres_logits = upsample_lowres_logits

    def forward(self, low_res_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        masks: predicted masks logits, shape (B, 1, H_original, W_original)
        low_res_logits: predicted low-res mask logits, shape (B, 1, H_lowres, W_lowres)
        targets: binary 0/1, shape (B, 1, H_original, W_original)
        """
        if self.upsample_lowres_logits is not None:
            low_res_logits = F.interpolate(
                low_res_logits, size=self.upsample_lowres_logits, mode='bilinear', align_corners=False)

        targets = targets.to(device=low_res_logits.device, dtype=torch.float32)
        targets_resized = F.interpolate(targets, size=(
            low_res_logits.shape[-2], low_res_logits.shape[-1]), mode='nearest')

        # --- cross entropy term ---
        pixel_bce = self.bce(low_res_logits, targets_resized)
        sample_bce = pixel_bce.view(pixel_bce.shape[0], -1).mean(dim=1)
        bce_loss = sample_bce.mean()

        # --- dice term ---
        masks_prob = torch.sigmoid(low_res_logits)
        masks_flat = masks_prob.view(masks_prob.shape[0], -1)
        targets_flat = targets_resized.view(targets_resized.shape[0], -1)

        intersection = (masks_flat * targets_flat).sum(dim=1)
        cardinality = masks_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_loss = (1 - (2 * intersection + 1e-6) /
                     (cardinality + 1e-6)).mean()

        # --- combined loss ---
        return (1.0 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


def hard_dice_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    return (2 * intersection + 1e-6) / (union + 1e-6)


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
def evaluate_model(model: SamLoRA, test_imgs_dir: Path, test_masks_dir: Path, device: torch.device, model_params_name: str):
    model.eval()
    model.to(device)

    dataset = SegmentationDataset(test_imgs_dir, test_masks_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,)

    bce_with_logits_dice_loss = BCEWithLogitsDiceLoss()

    hard_dice_scores = []
    iou_scores = []
    losses = []

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        batched_input = get_batched_input_list(images)
        outputs = model(batched_input, multimask_output=False)

        output_logits = torch.cat([d["low_res_logits"]
                                   for d in outputs], dim=0)
        output_mask = outputs[0]['masks'].squeeze(0)

        losses.append(bce_with_logits_dice_loss(output_logits, masks).item())
        hard_dice_scores.append(hard_dice_score(output_mask, masks).item())
        iou_scores.append(iou_score(output_mask, masks).item())

    metrics = {
        f"test/{model_params_name}/mean_bce_dice_loss": sum(losses) / len(losses),
        f"test/{model_params_name}/mean_dice_score": sum(hard_dice_scores) / len(hard_dice_scores),
        f"test/{model_params_name}/mean_iou_score": sum(iou_scores) / len(iou_scores),
    }

    return (metrics)
