from datetime import datetime
import os
import random
import sys
from zoneinfo import ZoneInfo

from segment_anything import sam_model_registry
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from pathlib import Path
import wandb

from Segmentation.SAM.sam_lora import SamLoRA
from Segmentation.SAM.sam_lora_util import ClDiceDiceBCELoss, SegmentationDataset, evaluate_model, get_batched_input_list, EVALUATED_TAG
from Segmentation.Util.env_utils import load_as, load_as_bool, load_as_tuple, load_segmentation_env
from Segmentation.Util.dataset_util import get_base_images

load_segmentation_env()

SEED = load_as("SEED", int, 42)

MODEL_CHECKPOINTS_DIR = os.getenv("MODEL_CHECKPOINTS_DIR")
MODEL_OUT_DIR = os.getenv("MODEL_OUT_DIR")
DATASETS_DIR = os.getenv("DATASETS_DIR")

DATASET_NAME = os.getenv("DATASET_NAME", "SAM_LoRA_Augmented")

LOWRES_IMG_PATCHES_DIR_NAME = os.getenv(
    "LOWRES_IMG_PATCHES_DIR_NAME", "img_patches_256")
LOWRES_MASK_PATCHES_DIR_NAME = os.getenv(
    "LOWRES_MASK_PATCHES_DIR_NAME", "mask_patches_256")
HIGHRES_IMG_PATCHES_DIR_NAME = os.getenv(
    "HIGHRES_IMG_PATCHES_DIR_NAME", "img_patches_1024")
HIGHRES_MASK_PATCHES_DIR_NAME = os.getenv(
    "HIGHRES_MASK_PATCHES_DIR_NAME", "mask_patches_1024")

USE_WANDB = load_as_bool("USE_WANDB", True)
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ForkSight-SAM")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

SAM_LORA_INPUT_IMG_TYPE = os.getenv(
    "SAM_LORA_INPUT_IMG_TYPE", "patches_highres")
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
SAM_LORA_UPSAMPLE_LOWRES_LOGITS = load_as_tuple(
    "SAM_LORA_UPSAMPLE_LOWRES_LOGITS", default=None, dtype=int)
SAM_LORA_MODEL_TYPE = os.getenv("SAM_LORA_MODEL_TYPE", "vit_b")
SAM_LORA_MODEL_CHECKPOINT = os.getenv(
    "SAM_LORA_MODEL_CHECKPOINT", "sam_vit_b_01ec64")
SAM_LORA_RANK = load_as("SAM_LORA_RANK", int, 4)
SAM_LORA_SCHEDULER_TYPE = os.getenv("SAM_LORA_SCHEDULER_TYPE", "OneCycleLR")
SAM_LORA_CL_DICE_LOSS_WEIGHT = load_as(
    "SAM_LORA_CL_DICE_LOSS_WEIGHT", float, 0.45)
SAM_LORA_DICE_LOSS_WEIGHT = load_as("SAM_LORA_DICE_LOSS_WEIGHT", float, 0.45)
SAM_LORA_CL_DICE_SKELETONIZE_ITERATIONS = load_as(
    "SAM_LORA_CL_DICE_SKELETONIZE_ITERATIONS", int, 15)

EARLY_STOPPING_PATIENCE = load_as("EARLY_STOPPING_PATIENCE", int, 15)
EARLY_STOPPING_DELTA = load_as("EARLY_STOPPING_DELTA", float, 0.005)
EARLY_STOPPING_MIN_EPOCHS = load_as("EARLY_STOPPING_MIN_EPOCHS", int, 50)

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

train_dir = Path(DATASETS_DIR) / DATASET_NAME / "train"
test_dir = Path(DATASETS_DIR) / DATASET_NAME / "test"
if SAM_LORA_INPUT_IMG_TYPE == "patches_lowres":
    TRAIN_IMAGES_DIR = train_dir / LOWRES_IMG_PATCHES_DIR_NAME
    TRAIN_MASKS_DIR = train_dir / LOWRES_MASK_PATCHES_DIR_NAME
    TEST_IMAGES_DIR = test_dir / LOWRES_IMG_PATCHES_DIR_NAME
    TEST_MASKS_DIR = test_dir / LOWRES_MASK_PATCHES_DIR_NAME
elif SAM_LORA_INPUT_IMG_TYPE == "patches_highres":
    TRAIN_IMAGES_DIR = train_dir / HIGHRES_IMG_PATCHES_DIR_NAME
    TRAIN_MASKS_DIR = train_dir / HIGHRES_MASK_PATCHES_DIR_NAME
    TEST_IMAGES_DIR = test_dir / HIGHRES_IMG_PATCHES_DIR_NAME
    TEST_MASKS_DIR = test_dir / HIGHRES_MASK_PATCHES_DIR_NAME


RUN_DATETIME_STR = datetime.now(
    ZoneInfo("Europe/Zurich")).strftime("%Y%m%d_%H%M%S")


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, min_epochs=0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs

        self.lowest_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss: float, epoch: int) -> bool:
        if epoch < self.min_epochs:
            return False

        if self.lowest_loss is None:
            self.lowest_loss = current_loss
            return False

        if (self.lowest_loss - current_loss) > self.min_delta:
            self.lowest_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered (lowest loss: {:.4f})".format(
                    self.lowest_loss))
        return self.early_stop


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_init_run_out_dir(wandb_run):
    run_dirname = wandb_run.name.lower(
    ) if wandb_run is not None else f"sam_lora_finetuning_{RUN_DATETIME_STR}"

    run_out_dir = Path(MODEL_OUT_DIR) / run_dirname
    run_out_dir.mkdir(parents=True, exist_ok=True)

    return run_out_dir


def init_wandb_run(trainset_len: int, valset_len: int, trainable_params_count: int):
    finetuned_modules = []
    if SAM_LORA_FINETUNE_IMAGE_ENCODER:
        finetuned_modules.append("image_encoder")
    if SAM_LORA_FINETUNE_MASK_DECODER:
        finetuned_modules.append("mask_decoder")
    if SAM_LORA_FINETUNE_PROMPT_ENCODER:
        finetuned_modules.append("prompt_encoder")

    base_training_images = get_base_images(imgs_dir=TRAIN_IMAGES_DIR)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"SAM_LoRA_Finetuning_{RUN_DATETIME_STR}",
        config={
            "learning_rate": SAM_LORA_LR,
            "learning_rate_scheduler": SAM_LORA_SCHEDULER_TYPE,
            "SAM_model_type": SAM_LORA_MODEL_TYPE,
            "SAM_checkpoint": SAM_LORA_MODEL_CHECKPOINT,
            "LoRA_rank": SAM_LORA_RANK,
            "finetuned_modules": str(finetuned_modules),
            "dataset": f"{DATASET_NAME}",
            "input_img_type": SAM_LORA_INPUT_IMG_TYPE,
            "train_set_size": trainset_len,
            "val_set_size": valset_len,
            "num_base_training_images": len(base_training_images),
            "base_training_images": str(base_training_images),
            "epochs": SAM_LORA_MAX_EPOCHS,
            "batch_size": SAM_LORA_BATCH_SIZE,
            "num_classes": SAM_LORA_NUM_CLASSES,
            "trainable_parameters": trainable_params_count,
            "upsample_lowres_logits": str(SAM_LORA_UPSAMPLE_LOWRES_LOGITS),
            "cl_dice_loss_weight": SAM_LORA_CL_DICE_LOSS_WEIGHT,
            "dice_loss_weight": SAM_LORA_DICE_LOSS_WEIGHT,
            "cl_dice_skeletonize_iterations": SAM_LORA_CL_DICE_SKELETONIZE_ITERATIONS,
        },
    )

    run_out_dir = get_init_run_out_dir(run)
    with open(str(run_out_dir / "wandb_run_id.txt"), "w") as f:
        f.write(run.id)

    return run


def init_model(device: torch.device, verbose: bool = True) -> SamLoRA:
    if verbose:
        print(
            f"using python: {sys.executable}, {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")

    sam_checkpoint = str(Path(MODEL_CHECKPOINTS_DIR) /
                         f"{SAM_LORA_MODEL_CHECKPOINT}.pth")
    model_type = SAM_LORA_MODEL_TYPE
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    if verbose:
        print(
            f"SAM model loaded on {sam.device}, with {sum(p.numel() for p in sam.parameters() if p.requires_grad)} trainable parameters")

    sam_lora = SamLoRA(sam, r=SAM_LORA_RANK, finetune_img_encoder=SAM_LORA_FINETUNE_IMAGE_ENCODER,
                       finetune_mask_decoder=SAM_LORA_FINETUNE_MASK_DECODER, finetune_prompt_encoder=SAM_LORA_FINETUNE_PROMPT_ENCODER)
    sam_lora.to(device)

    if verbose:
        print(
            f"SAM model with LoRA fine-tuning initialized, on {sam_lora.device}, with {sum(p.numel() for p in sam_lora.parameters() if p.requires_grad)} trainable parameters")

    return sam_lora


def save_params(sam_lora: SamLoRA, wandb_run, suffix: str = None):
    params = {name: p.detach().cpu() for name,
              p in sam_lora.named_parameters() if p.requires_grad}

    suffix = f"_{suffix}" if suffix else ""
    filename = f"params{suffix}.pt"
    run_out_dir = get_init_run_out_dir(wandb_run)
    filepath = str(run_out_dir / filename)

    torch.save(params, filepath)


def init_data_loaders():
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
        dataset, batch_size=SAM_LORA_BATCH_SIZE, sampler=train_sampler, drop_last=True)
    validationloader = DataLoader(
        dataset, batch_size=SAM_LORA_BATCH_SIZE, sampler=val_sampler)

    return trainloader, validationloader, len(train_indices), len(val_indices)


def get_trainable_params(sam_lora: SamLoRA):
    return [
        (name, p) for name, p in sam_lora.named_parameters()
        if p.requires_grad
    ]


def train(sam_lora: SamLoRA, wandb_run: wandb.Run, trainloader: DataLoader, validationloader: DataLoader, device: torch.device):
    loss_fn = ClDiceDiceBCELoss(skeletonize_iter=SAM_LORA_CL_DICE_SKELETONIZE_ITERATIONS,
                                cl_dice_weight=SAM_LORA_CL_DICE_LOSS_WEIGHT,
                                dice_weight=SAM_LORA_DICE_LOSS_WEIGHT,
                                upsample_lowres_logits=SAM_LORA_UPSAMPLE_LOWRES_LOGITS)

    trainable_params = get_trainable_params(sam_lora)
    for name, p in trainable_params:
        print(f"Training: {name} with {p.numel()} parameters")
    print()

    optimizer = torch.optim.AdamW(
        params=[p for _, p in trainable_params],
        lr=SAM_LORA_LR,
    )

    steps_per_epoch = len(trainloader)
    total_steps = steps_per_epoch * SAM_LORA_MAX_EPOCHS
    scheduler = None
    if SAM_LORA_SCHEDULER_TYPE == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=SAM_LORA_LR, total_steps=total_steps, pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=10.0)
    elif SAM_LORA_SCHEDULER_TYPE == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=0.0001)

    min_validation_loss = float('inf')
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_DELTA, min_epochs=EARLY_STOPPING_MIN_EPOCHS)

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

            torch.nn.utils.clip_grad_norm_(sam_lora.parameters(), max_norm=1.0)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

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

        # epoch metrics
        num_training_samples = len(trainloader) * trainloader.batch_size
        mean_training_loss = total_training_loss / num_training_samples
        num_validation_samples = len(
            validationloader) * validationloader.batch_size
        mean_validation_loss = total_validation_loss / num_validation_samples

        print(f"    Train Loss: {mean_training_loss:.4f}")
        print(f"    Validation Loss: {mean_validation_loss:.4f}")
        print(f"    Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        if mean_validation_loss < min_validation_loss:
            min_validation_loss = mean_validation_loss
            print("    New minimum validation loss achieved, saving model parameters")
            save_params(sam_lora, wandb_run, "minloss")

        if USE_WANDB and wandb_run is not None:
            wandb_run.log({
                "train/loss": mean_training_loss,
                "validation/loss": mean_validation_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            })

        # early stopping
        if early_stopping(mean_validation_loss, epoch):
            break

    save_params(sam_lora, wandb_run, "final")


def evaluate_checkpoints(wandb_run: wandb.Run, device: torch.device):
    run_out_dir = get_init_run_out_dir(wandb_run)

    for param_file in run_out_dir.glob("*.pt"):
        print(
            f"\nEvaluating model parameters from file: {str(param_file)}")

        sam_lora = init_model(device, verbose=False)
        params = torch.load(param_file, map_location=device)
        sam_lora.load_state_dict(params, strict=False)

        metrics = evaluate_model(model=sam_lora, test_imgs_dir=TEST_IMAGES_DIR, test_masks_dir=TEST_MASKS_DIR,
                                 device=device, model_params_name=param_file.stem,
                                 cl_dice_skeletonize_iter=SAM_LORA_CL_DICE_SKELETONIZE_ITERATIONS, cl_dice_weight=SAM_LORA_CL_DICE_LOSS_WEIGHT,
                                 dice_weight=SAM_LORA_DICE_LOSS_WEIGHT)
        for metric_name, metric_value in metrics.items():
            print(f"        {metric_name}: {metric_value:.4f}")
            wandb_run.summary[metric_name] = metric_value

    wandb_run.tags = list(set(wandb_run.tags) | {EVALUATED_TAG})


def train_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_lora = init_model(device)
    trainloader, validationloader, train_size, val_size = init_data_loaders()

    wandb_run = None
    if USE_WANDB:
        wandb.login(key=WANDB_API_KEY)
        wandb_run = init_wandb_run(train_size, val_size, sum(
            p.numel() for _, p in get_trainable_params(sam_lora)))

    train(sam_lora, wandb_run, trainloader, validationloader, device)

    if USE_WANDB and wandb_run is not None:
        evaluate_checkpoints(wandb_run, device)
        wandb_run.finish()


if __name__ == "__main__":
    seed_everything(SEED)
    train_evaluate()
