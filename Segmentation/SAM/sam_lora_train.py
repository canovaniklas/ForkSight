from datetime import datetime
import os
import random
import sys
from zoneinfo import ZoneInfo
import re

from sam_lora import SamLoRA
from segment_anything import sam_model_registry
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import wandb

USE_WANDB = True

FINETUNE_IMAGE_ENCODER = False
FINETUNE_MASK_DECODER = True
FINETUNE_PROMPT_ENCODER = True

MODEL_CHECKPOINTS_DIR = "/home/jhehli/data/model_checkpoints/"

# TRAIN_IMAGES_DIR = Path(
#    "/home/jhehli/data/datasets/SAM_LoRA_Augmented/train/images/")
# TRAIN_MASKS_DIR = Path("/home/jhehli/data/datasets/SAM_LoRA_Augmented/train/masks/")
# TEST_IMAGES_DIR = Path("/home/jhehli/data/datasets/SAM_LoRA_Augmented/test/images/")
# TEST_MASKS_DIR = Path("/home/jhehli/data/datasets/SAM_LoRA_Augmented/test/masks/")

TRAIN_IMAGES_DIR = Path(
    "/home/jhehli/data/datasets/SAM_LoRA_Augmented/train/images_256/")
TRAIN_MASKS_DIR = Path(
    "/home/jhehli/data/datasets/SAM_LoRA_Augmented/train/masks_256/")
TEST_IMAGES_DIR = Path(
    "/home/jhehli/data/datasets/SAM_LoRA_Augmented/test/images_256/")
TEST_MASKS_DIR = Path(
    "/home/jhehli/data/datasets/SAM_LoRA_Augmented/test/masks_256/")
LR = 0.001
NUM_CLASSES = 1  # binary segmentation (DNA vs background)
BATCH_SIZE = 6
MAX_EPOCHS = 200

UPSAMPLE_LOWRES_LOGITS = None  # set to None to disable upsampling


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path):
        self.image_paths = list(images_dir.glob("*.png"))
        self.masks_dir = masks_dir

    def _load_image(self, path: Path, is_mask: bool = False) -> torch.Tensor:
        transform = transforms.Compose([transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
                                        transforms.ToTensor()])

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

    def __init__(self, dice_weight: float = 0.8):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_weight = dice_weight
        self.upsample_lowres_logits = UPSAMPLE_LOWRES_LOGITS

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
            low_res_logits.shape[-2], low_res_logits.shape[-1]), mode='bilinear', align_corners=False)

        # --- cross entropy term ---
        pixel_bce = self.bce(low_res_logits, targets_resized)
        sample_bce = pixel_bce.view(pixel_bce.shape[0], -1).mean(dim=1)
        bce_loss = sample_bce.mean()

        # --- dice term ---
        # masks_flat = masks.view(masks.shape[0], -1).float()
        # targets_flat = targets.view(targets.shape[0], -1).float()
        # intersection = (masks_flat * targets_flat).sum(dim=1)
        # cardinality = masks_flat.sum(dim=1) + targets_flat.sum(dim=1)
        # dice_score = (2.0 * intersection + self.smooth) / \
        #    (cardinality + self.smooth)
        # dice_loss = (1.0 - dice_score).mean()
        masks_prob = torch.sigmoid(low_res_logits)
        masks_flat = masks_prob.view(masks_prob.shape[0], -1)
        targets_flat = targets_resized.view(targets_resized.shape[0], -1)

        intersection = (masks_flat * targets_flat).sum(dim=1)
        cardinality = masks_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_loss = (1 - (2 * intersection + 1e-6) /
                     (cardinality + 1e-6)).mean()

        # --- combined loss ---
        return (1.0 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_base_training_images():
    ''' returns a list of names of unique base images in the training dataset, before augmentation. '''
    train_image_filenames = [p.name for p in TRAIN_IMAGES_DIR.glob("*.png")]
    pattern = re.compile(r'^[0-9]{8}.*_soi_[0-9]+_[^_]+\.png$')
    return [f for f in train_image_filenames if pattern.match(f)]


def init_wandb_run(trainset_len: int, valset_len: int, trainable_params_count: int):
    curr_datetime = datetime.now(
        ZoneInfo("Europe/Zurich")).strftime("%Y%m%d_%H%M%S")

    architecture_suffixes = []
    if FINETUNE_IMAGE_ENCODER:
        architecture_suffixes.append("image_encoder")
    if FINETUNE_MASK_DECODER:
        architecture_suffixes.append("mask_decoder")
    if FINETUNE_PROMPT_ENCODER:
        architecture_suffixes.append("prompt_encoder")

    base_training_images = get_base_training_images()

    return wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="EM_IMCR_BIOVSION",
        # Set the wandb project where this run will be logged.
        project="ForkSight-SAM",
        name=f"SAM_LoRA_Finetuning_{curr_datetime}",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": LR,
            "architecture": f"SAM LoRA ({', '.join(architecture_suffixes)})",
            "dataset": f"lowres custom segmentation dataset (training set size: {trainset_len}, validation set size: {valset_len})",
            "num_base_training_images": len(base_training_images),
            "base_training_images": str(base_training_images),
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_classes": NUM_CLASSES,
            "trainable_parameters": trainable_params_count,
            "upsample_lowres_logits": str(UPSAMPLE_LOWRES_LOGITS),
        },
    )


def init_model(device: torch.device) -> SamLoRA:
    print(
        f"using python: {sys.executable}, {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")

    sam_checkpoint = str(Path(MODEL_CHECKPOINTS_DIR) / "sam_vit_b_01ec64.pth")
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    print(
        f"SAM model loaded on {sam.device}, with {sum(p.numel() for p in sam.parameters() if p.requires_grad)} trainable parameters")

    sam_lora = SamLoRA(sam, r=4, finetune_img_encoder=FINETUNE_IMAGE_ENCODER,
                       finetune_mask_decoder=FINETUNE_MASK_DECODER, finetune_prompt_encoder=FINETUNE_PROMPT_ENCODER)
    sam_lora.to(device)
    print(
        f"SAM model with LoRA fine-tuning initialized, on {sam_lora.device}, with {sum(p.numel() for p in sam_lora.parameters() if p.requires_grad)} trainable parameters")
    # print(sam_lora)

    return sam_lora


def save_mask(img, mask, idx=0):
    mask = mask.cpu().numpy()
    img = img.cpu().numpy().transpose(1, 2, 0)

    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)

    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')

    plt.imshow(img)

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.gca().imshow(mask_image)

    plt.tight_layout()
    fig.tight_layout()
    fig.savefig(f"sam_lora_output_{idx}.png")


def get_batched_input_list(batched_input: torch.Tensor):
    return [{
        "image": img,
        "original_size": (img.shape[1], img.shape[2])
    } for img in batched_input.unbind(0)]


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_lora = init_model(device)

    trainset = SegmentationDataset(
        images_dir=TRAIN_IMAGES_DIR, masks_dir=TRAIN_MASKS_DIR)
    validationset = SegmentationDataset(
        images_dir=TEST_IMAGES_DIR, masks_dir=TEST_MASKS_DIR)
    print("\nNumber of training samples:", len(trainset))
    print("Number of validation samples:", len(validationset), "\n")

    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True)
    validationloader = DataLoader(
        validationset, batch_size=BATCH_SIZE, shuffle=False)

    loss_fn = BCEWithLogitsDiceLoss()

    trainable_params = [
        (name, p) for name, p in sam_lora.named_parameters()
        if p.requires_grad
    ]

    for name, p in trainable_params:
        print(f"Training: {name} with {p.numel()} parameters")
    print()

    optimizer = torch.optim.AdamW(
        params=[p for _, p in trainable_params],
        lr=LR
    )

    if USE_WANDB:
        wandb.login(key="aa4147e2bb5f0315cc9f35b6475561223915112c")
        wandb_run = init_wandb_run(len(trainset), len(
            validationset), sum(p.numel() for _, p in trainable_params))

    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")

        # training
        sam_lora.train()
        total_training_loss = 0.0

        for batched_input, target_masks in trainloader:
            batched_input = batched_input.to(device)
            target_masks = target_masks.to(device)
            batched_input = get_batched_input_list(batched_input)

            optimizer.zero_grad()
            outputs = sam_lora(batched_input=batched_input,
                               multimask_output=NUM_CLASSES > 1)
            output_logits = torch.cat([d["low_res_logits"]
                                      for d in outputs], dim=0)

            loss = loss_fn(output_logits, target_masks)
            total_training_loss += loss.item() * len(batched_input)

            loss.backward()
            optimizer.step()

        # validation
        sam_lora.eval()
        total_validation_loss = 0.0

        with torch.no_grad():
            for batched_input, target_masks in validationloader:
                batched_input = batched_input.to(device)
                target_masks = target_masks.to(device)
                batched_input = get_batched_input_list(batched_input)

                outputs = sam_lora(batched_input=batched_input,
                                   multimask_output=NUM_CLASSES > 1)
                output_logits = torch.cat([d["low_res_logits"]
                                          for d in outputs], dim=0)

                loss = loss_fn(output_logits, target_masks)
                total_validation_loss += loss.item() * len(batched_input)

        mean_training_loss = total_training_loss / len(trainset)
        mean_validation_loss = total_validation_loss / len(validationset)

        print(f"    Train Loss: {mean_training_loss:.4f}")
        print(f"    Validation Loss: {mean_validation_loss:.4f}")

        if USE_WANDB and wandb_run is not None:
            wandb_run.log({
                "train/loss": mean_training_loss,
                "validation/loss": mean_validation_loss,
            })

    # save the fine-tuned model parameters
    trainable_params = {name: p.detach().cpu() for name,
                        p in sam_lora.named_parameters() if p.requires_grad}

    curr_datetime = datetime.now(
        ZoneInfo("Europe/Zurich")).strftime("%Y%m%d_%H%M%S")
    torch.save(trainable_params,
               f"SavedModels/sam_lora_finetuned_params_{curr_datetime}.pt")

    if USE_WANDB and wandb_run is not None:
        artifact = wandb.Artifact(
            name=f"sam_lora_finetuned_params_{curr_datetime}", type="model")
        artifact.add_file(
            f"SavedModels/sam_lora_finetuned_params_{curr_datetime}.pt")
        wandb_run.log_artifact(artifact)

        wandb_run.finish()


if __name__ == "__main__":
    seed_everything(42)
    train()
