from datetime import datetime
import os
import random
import sys
from zoneinfo import ZoneInfo
import re

from segment_anything import sam_model_registry
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import wandb

from Segmentation.SAM.sam_lora import SamLoRA
from Segmentation.Util.env_utils import load_as, load_as_bool, load_segmentation_env

load_segmentation_env()

SEED = load_as("SEED", int, 42)

MODEL_CHECKPOINTS_DIR = os.getenv("MODEL_CHECKPOINTS_DIR")
MODEL_OUT_DIR = os.getenv("MODEL_OUT_DIR")
DATASETS_DIR = os.getenv("DATASETS_DIR")

DATASET_NAME = os.getenv("DATASET_NAME", "SAM_LoRA_Augmented")

LOWRES_IMG_DIR_NAME = os.getenv("LOWRES_IMG_DIR_NAME", "images_1024")
LOWRES_MASK_DIR_NAME = os.getenv("LOWRES_MASK_DIR_NAME", "masks_1024")
CROPPED_AUG_IMG_DIR_NAME = os.getenv("CROPPED_AUG_IMG_DIR_NAME", "images_256")
CROPPED_AUG_MASK_DIR_NAME = os.getenv("CROPPED_AUG_MASK_DIR_NAME", "masks_256")

USE_WANDB = load_as_bool("USE_WANDB", True)
WANDB_TEST = load_as_bool("WANDB_TEST", False)
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ForkSight-SAM")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

SAM_LORA_USE_CROPPED_IMAGES = load_as_bool(
    "SAM_LORA_USE_CROPPED_IMAGES", True)
SAM_LORA_FINETUNE_IMAGE_ENCODER = load_as_bool(
    "SAM_LORA_FINETUNE_IMAGE_ENCODER", False)
SAM_LORA_FINETUNE_MASK_DECODER = load_as_bool(
    "SAM_LORA_FINETUNE_MASK_DECODER", True)
SAM_LORA_FINETUNE_PROMPT_ENCODER = load_as_bool(
    "SAM_LORA_FINETUNE_PROMPT_ENCODER", True)

SAM_LORA_LR = load_as("SAM_LORA_LR", float, 1e-3)
SAM_LORA_NUM_CLASSES = load_as("SAM_LORA_NUM_CLASSES", int, 1)
SAM_LORA_BATCH_SIZE = load_as("SAM_LORA_BATCH_SIZE", int, 2)
SAM_LORA_MAX_EPOCHS = load_as("SAM_LORA_MAX_EPOCHS", int, 150)
# set to None to disable upsampling
SAM_LORA_UPSAMPLE_LOWRES_LOGITS = os.getenv(
    "SAM_LORA_UPSAMPLE_LOWRES_LOGITS") or None
SAM_LORA_MODEL_TYPE = os.getenv("SAM_LORA_MODEL_TYPE", "vit_b")
SAM_LORA_MODEL_CHECKPOINT = os.getenv(
    "SAM_LORA_MODEL_CHECKPOINT", "sam_vit_b_01ec64")
SAM_LORA_RANK = load_as("SAM_LORA_RANK", int, 4)

if MODEL_CHECKPOINTS_DIR is None or DATASETS_DIR is None or MODEL_OUT_DIR is None:
    raise ValueError(
        "MODEL_CHECKPOINTS_DIR, DATASETS_DIR, and MODEL_OUT_DIR environment variables must be set.")
if not Path(MODEL_CHECKPOINTS_DIR).is_dir():
    raise ValueError(
        f"MODEL_CHECKPOINTS_DIR '{MODEL_CHECKPOINTS_DIR}' is not a valid directory.")
if not Path(DATASETS_DIR).is_dir():
    raise ValueError(
        f"DATASETS_DIR '{DATASETS_DIR}' is not a valid directory.")
if not Path(MODEL_OUT_DIR).is_dir():
    raise ValueError(
        f"MODEL_OUT_DIR '{MODEL_OUT_DIR}' is not a valid directory.")

TRAIN_IMAGES_DIR = Path(DATASETS_DIR) / DATASET_NAME / "train" / (
    CROPPED_AUG_IMG_DIR_NAME if SAM_LORA_USE_CROPPED_IMAGES else LOWRES_IMG_DIR_NAME)
TRAIN_MASKS_DIR = Path(DATASETS_DIR) / DATASET_NAME / "train" / (
    CROPPED_AUG_MASK_DIR_NAME if SAM_LORA_USE_CROPPED_IMAGES else LOWRES_MASK_DIR_NAME)


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path):
        self.image_paths = list(images_dir.glob("*.png"))
        self.masks_dir = masks_dir

    def _load_image(self, path: Path, is_mask: bool = False) -> torch.Tensor:
        # using nearest neighbor interpolation for masks to preserve label values (no interpolation)
        transform = transforms.Compose([transforms.Resize((1024, 1024), interpolation=(transforms.InterpolationMode.NEAREST if is_mask else transforms.InterpolationMode.BILINEAR)),
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
        self.upsample_lowres_logits = SAM_LORA_UPSAMPLE_LOWRES_LOGITS

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
            low_res_logits.shape[-2], low_res_logits.shape[-1]), mode='nearest', align_corners=False)

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

    finetuned_modules = []
    if SAM_LORA_FINETUNE_IMAGE_ENCODER:
        finetuned_modules.append("image_encoder")
    if SAM_LORA_FINETUNE_MASK_DECODER:
        finetuned_modules.append("mask_decoder")
    if SAM_LORA_FINETUNE_PROMPT_ENCODER:
        finetuned_modules.append("prompt_encoder")

    base_training_images = get_base_training_images()

    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"SAM_LoRA_Finetuning_{curr_datetime}",
        config={
            "learning_rate": SAM_LORA_LR,
            "SAM_checkpoint": SAM_LORA_MODEL_CHECKPOINT,
            "LoRA_rank": SAM_LORA_RANK,
            "finetuned_modules": str(finetuned_modules),
            "dataset": f"{DATASET_NAME}",
            "use_cropped_images": SAM_LORA_USE_CROPPED_IMAGES,
            "train_set_size": trainset_len,
            "val_set_size": valset_len,
            "num_base_training_images": len(base_training_images),
            "base_training_images": str(base_training_images),
            "epochs": SAM_LORA_MAX_EPOCHS,
            "batch_size": SAM_LORA_BATCH_SIZE,
            "num_classes": SAM_LORA_NUM_CLASSES,
            "trainable_parameters": trainable_params_count,
            "upsample_lowres_logits": str(SAM_LORA_UPSAMPLE_LOWRES_LOGITS),
        },
    )


def init_model(device: torch.device) -> SamLoRA:
    print(
        f"using python: {sys.executable}, {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")

    sam_checkpoint = str(Path(MODEL_CHECKPOINTS_DIR) /
                         f"{SAM_LORA_MODEL_CHECKPOINT}.pth")
    model_type = SAM_LORA_MODEL_TYPE
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    print(
        f"SAM model loaded on {sam.device}, with {sum(p.numel() for p in sam.parameters() if p.requires_grad)} trainable parameters")

    sam_lora = SamLoRA(sam, r=SAM_LORA_RANK, finetune_img_encoder=SAM_LORA_FINETUNE_IMAGE_ENCODER,
                       finetune_mask_decoder=SAM_LORA_FINETUNE_MASK_DECODER, finetune_prompt_encoder=SAM_LORA_FINETUNE_PROMPT_ENCODER)
    sam_lora.to(device)

    print(
        f"SAM model with LoRA fine-tuning initialized, on {sam_lora.device}, with {sum(p.numel() for p in sam_lora.parameters() if p.requires_grad)} trainable parameters")

    return sam_lora


def get_batched_input_list(batched_input: torch.Tensor):
    return [{
        "image": img,
        "original_size": (img.shape[1], img.shape[2])
    } for img in batched_input.unbind(0)]


def save_params(params: dict[str, torch.Tensor], wandb_run, filename=None):
    if filename is None:
        curr_datetime = datetime.now(
            ZoneInfo("Europe/Zurich")).strftime("%Y%m%d_%H%M%S")
        filename = f"sam_lora_finetuned_params_{curr_datetime}.pt"

    model_out_path = Path(MODEL_OUT_DIR) / filename
    torch.save(params, str(model_out_path))

    if USE_WANDB and wandb_run is not None:
        artifact = wandb.Artifact(name=model_out_path.stem, type="model")
        artifact.add_file(str(model_out_path))
        print(
            f"Saving fine-tuned model parameters {str(model_out_path)}to wandb artifact '{model_out_path.stem}'")
        print(artifact)
        wandb_run.log_artifact(artifact)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_lora = init_model(device)

    dataset = SegmentationDataset(
        images_dir=TRAIN_IMAGES_DIR, masks_dir=TRAIN_MASKS_DIR)

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices, val_indices = indices[:split], indices[split:]

    print("\nNumber of training samples:", len(train_indices))
    print("Number of validation samples:", len(val_indices), "\n")

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    trainloader = DataLoader(
        dataset, batch_size=SAM_LORA_BATCH_SIZE, sampler=train_sampler)
    validationloader = DataLoader(
        dataset, batch_size=SAM_LORA_BATCH_SIZE, sampler=val_sampler)

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
        lr=SAM_LORA_LR,
    )

    wandb_run = None
    if USE_WANDB:
        wandb.login(key=WANDB_API_KEY)
        wandb_run = init_wandb_run(len(train_indices), len(
            val_indices), sum(p.numel() for _, p in trainable_params))

    if USE_WANDB and WANDB_TEST and wandb_run is not None:
        test_params = {"test_param": torch.randn(2, 2)}
        save_params(test_params, wandb_run, filename="wandb_test_artifact.pt")
        print("wandb test artifact created, exiting.")

        wandb_run.finish()
        return

    for epoch in range(SAM_LORA_MAX_EPOCHS):
        print(f"\nEpoch {epoch+1}/{SAM_LORA_MAX_EPOCHS}")

        # training
        sam_lora.train()
        total_training_loss = 0.0

        for batched_input, target_masks in trainloader:
            batched_input = batched_input.to(device)
            target_masks = target_masks.to(device)
            batched_input = get_batched_input_list(batched_input)

            optimizer.zero_grad()
            outputs = sam_lora(batched_input=batched_input,
                               multimask_output=SAM_LORA_NUM_CLASSES > 1)
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
                                   multimask_output=SAM_LORA_NUM_CLASSES > 1)
                output_logits = torch.cat([d["low_res_logits"]
                                          for d in outputs], dim=0)

                loss = loss_fn(output_logits, target_masks)
                total_validation_loss += loss.item() * len(batched_input)

        mean_training_loss = total_training_loss / len(train_indices)
        mean_validation_loss = total_validation_loss / len(val_indices)

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
    save_params(trainable_params, wandb_run)

    if USE_WANDB and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    seed_everything(SEED)
    train()
