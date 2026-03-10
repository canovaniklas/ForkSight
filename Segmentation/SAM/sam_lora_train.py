import os
from datetime import datetime
import random
import sys
from zoneinfo import ZoneInfo
from Evaluation.evaluation_util import compute_metrics
from segment_anything import sam_model_registry
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
import ast

from Segmentation.SAM.sam_lora import SamLoRA
from Segmentation.SAM.sam_lora_util import EVALUATED_TAG, CombinedLoss, SegmentationDataset, evaluate_model, get_batched_input_list
from Environment.env_utils import load_as, load_as_bool, load_segmentation_env
from Segmentation.PreProcessing.dataset_util import get_base_images

load_segmentation_env()

SEED = load_as("SEED", int, 42)

MODEL_CHECKPOINTS_DIR = os.getenv("MODEL_CHECKPOINTS_DIR")
MODEL_OUT_DIR = os.getenv("MODEL_OUT_DIR")
DATASETS_DIR = os.getenv("DATASETS_DIR")

DATASET_NAME = os.getenv("DATASET_NAME", "Segmentation_v1")

HIGHRES_IMG_PATCHES_DIR_NAME = os.getenv(
    "HIGHRES_IMG_PATCHES_DIR_NAME", "img_patches_1024")
HIGHRES_MASK_PATCHES_DIR_NAME = os.getenv(
    "HIGHRES_MASK_PATCHES_DIR_NAME", "mask_patches_1024")
HIGHRES_HEATMAP_PATCHES_DIR_NAME = os.getenv(
    "HIGHRES_HEATMAP_PATCHES_DIR_NAME", "heatmap_patches_1024")

USE_WANDB = load_as_bool("USE_WANDB", True)
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ForkSight-SAM")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

SAM_LORA_FINETUNE_IMAGE_ENCODER = load_as_bool(
    "SAM_LORA_FINETUNE_IMAGE_ENCODER", False)
SAM_LORA_FINETUNE_IMAGE_ENCODER_N_BLOCKS = load_as(
    "SAM_LORA_FINETUNE_IMAGE_ENCODER_N_BLOCKS", int, 0)
SAM_LORA_FINETUNE_MASK_DECODER = load_as_bool(
    "SAM_LORA_FINETUNE_MASK_DECODER", True)
SAM_LORA_FINETUNE_PROMPT_ENCODER = load_as_bool(
    "SAM_LORA_FINETUNE_PROMPT_ENCODER", True)

SAM_LORA_LR = load_as("SAM_LORA_LR", float, 1e-3)
SAM_LORA_IMAGE_ENCODER_LR = load_as("SAM_LORA_IMAGE_ENCODER_LR", float, None)
SAM_LORA_BATCH_SIZE = load_as("SAM_LORA_BATCH_SIZE", int, 20)
SAM_LORA_MAX_EPOCHS = load_as("SAM_LORA_MAX_EPOCHS", int, 100)
SAM_LORA_MODEL_TYPE = os.getenv("SAM_LORA_MODEL_TYPE", "vit_b")
SAM_LORA_MODEL_CHECKPOINT = os.getenv(
    "SAM_LORA_MODEL_CHECKPOINT", "sam_vit_b_01ec64")
SAM_LORA_RANK = load_as("SAM_LORA_RANK", int, 4)
SAM_LORA_SCHEDULER_TYPE = os.getenv("SAM_LORA_SCHEDULER_TYPE", "OneCycleLR")

SAM_LORA_BCE_LOSS_WEIGHT = load_as("SAM_LORA_BCE_LOSS_WEIGHT", float, 0.0)
SAM_LORA_FOCAL_LOSS_WEIGHT = load_as("SAM_LORA_FOCAL_LOSS_WEIGHT", float, 0.0)
SAM_LORA_FOCAL_ALPHA = load_as("SAM_LORA_FOCAL_ALPHA", float, 0.25)
SAM_LORA_FOCAL_GAMMA = load_as("SAM_LORA_FOCAL_GAMMA", float, 2.0)
SAM_LORA_DICE_LOSS_WEIGHT = load_as("SAM_LORA_DICE_LOSS_WEIGHT", float, 0.0)
SAM_LORA_CL_DICE_LOSS_WEIGHT = load_as(
    "SAM_LORA_CL_DICE_LOSS_WEIGHT", float, 0.0)
SAM_LORA_CL_DICE_SKELETONIZE_ITERATIONS = load_as(
    "SAM_LORA_CL_DICE_SKELETONIZE_ITERATIONS", int, 15)
SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT = load_as(
    "SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT", float, 0.0)
# set to 0.0 to disable junction heatmap weighting
SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE = load_as(
    "SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE", float, 0.0)
# set None to disable junction patch loss
SAM_LORA_JUNCTION_LOSS_TYPE = os.getenv("SAM_LORA_JUNCTION_LOSS_TYPE", None)
SAM_LORA_JUNCTION_PATCH_WEIGHT = load_as(
    "SAM_LORA_JUNCTION_PATCH_WEIGHT", float, 0.0)
SAM_LORA_TOPOLOGICAL_LOSS_WEIGHT = load_as(
    "SAM_LORA_TOPOLOGICAL_LOSS_WEIGHT", float, 0.0)

_raw_downsample = load_as("DATASET_DOWNSAMPLE_SIZE", int, None)
DATASET_DOWNSAMPLE_SIZE = (_raw_downsample, _raw_downsample) if (
    _raw_downsample is not None and _raw_downsample != 0) else None

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
val_dir = Path(DATASETS_DIR) / DATASET_NAME / "validation"
test_dir = Path(DATASETS_DIR) / DATASET_NAME / "test"
TRAIN_IMAGES_DIR = train_dir / HIGHRES_IMG_PATCHES_DIR_NAME
TRAIN_MASKS_DIR = train_dir / HIGHRES_MASK_PATCHES_DIR_NAME
VAL_IMAGES_DIR = val_dir / HIGHRES_IMG_PATCHES_DIR_NAME
VAL_MASKS_DIR = val_dir / HIGHRES_MASK_PATCHES_DIR_NAME
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

    def reset(self):
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
    run_dirname = f"{wandb_run.name.lower()}_{wandb_run.id}" \
        if wandb_run is not None else f"sam_lora_finetuning_{RUN_DATETIME_STR}"

    run_out_dir = Path(MODEL_OUT_DIR) / run_dirname
    run_out_dir.mkdir(parents=True, exist_ok=True)

    return run_out_dir


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

    sam_lora = SamLoRA(sam, r=SAM_LORA_RANK, finetune_img_encoder_lora=SAM_LORA_FINETUNE_IMAGE_ENCODER,
                       finetune_img_encoder_n_blocks=SAM_LORA_FINETUNE_IMAGE_ENCODER_N_BLOCKS,
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
    train_heatmaps_dir = train_dir / HIGHRES_HEATMAP_PATCHES_DIR_NAME \
        if SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE > 0.0 else None
    val_heatmaps_dir = val_dir / HIGHRES_HEATMAP_PATCHES_DIR_NAME \
        if SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE > 0.0 else None

    train_dataset = SegmentationDataset(
        images_dir=TRAIN_IMAGES_DIR,
        masks_dir=TRAIN_MASKS_DIR,
        heatmaps_dir=train_heatmaps_dir,
        downsample_size=DATASET_DOWNSAMPLE_SIZE)

    val_dataset = SegmentationDataset(
        images_dir=VAL_IMAGES_DIR,
        masks_dir=VAL_MASKS_DIR,
        heatmaps_dir=val_heatmaps_dir,
        downsample_size=DATASET_DOWNSAMPLE_SIZE)

    print("\nNumber of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset), "\n")

    trainloader = DataLoader(
        train_dataset, batch_size=SAM_LORA_BATCH_SIZE, shuffle=True, drop_last=True)
    validationloader = DataLoader(
        val_dataset, batch_size=SAM_LORA_BATCH_SIZE, shuffle=False)

    return trainloader, validationloader, len(train_dataset), len(val_dataset)


def get_trainable_params(sam_lora: SamLoRA):
    return [
        (name, p) for name, p in sam_lora.named_parameters()
        if p.requires_grad
    ]


def get_cfg_string_from_finetuned_components(finetune_image_encoder, finetune_mask_decoder, finetune_prompt_encoder, finetune_image_encoder_n_blocks):
    modules = []
    if finetune_image_encoder_n_blocks > 0:
        modules.append(f"image_encoder_last_N_blocks_full")
    elif finetune_image_encoder:
        modules.append("image_encoder_lora")
    if finetune_mask_decoder:
        modules.append("mask_decoder")
    if finetune_prompt_encoder:
        modules.append("prompt_encoder")

    return modules


def get_finetuned_components_from_cfg(finetuned_modules):
    finetune_img_encoder = "image_encoder_lora" in finetuned_modules
    finetune_mask_decoder = "mask_decoder" in finetuned_modules
    finetune_prompt_encoder = "prompt_encoder" in finetuned_modules
    finetune_img_encoder_blocks = "image_encoder_last_N_blocks_full" in finetuned_modules

    return finetune_img_encoder, finetune_mask_decoder, finetune_prompt_encoder, finetune_img_encoder_blocks


def init_scheduler(optimizer, max_epochs, steps_per_epoch, max_lrs):
    total_steps = max_epochs * steps_per_epoch
    scheduler = None

    if SAM_LORA_SCHEDULER_TYPE == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lrs, total_steps=total_steps, pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=10.0)
    elif SAM_LORA_SCHEDULER_TYPE == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=0.0001)

    return scheduler


def train(sam_lora: SamLoRA, wandb_run: wandb.Run, trainloader: DataLoader, validationloader: DataLoader, device: torch.device):
    loss_fn = CombinedLoss(bce_weight=SAM_LORA_BCE_LOSS_WEIGHT,
                           focal_weight=SAM_LORA_FOCAL_LOSS_WEIGHT,
                           dice_weight=SAM_LORA_DICE_LOSS_WEIGHT,
                           cl_dice_weight=SAM_LORA_CL_DICE_LOSS_WEIGHT,
                           skeleton_recall_weight=SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT,
                           heatmap_weight_scale=SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE,
                           focal_alpha=SAM_LORA_FOCAL_ALPHA,
                           focal_gamma=SAM_LORA_FOCAL_GAMMA,
                           skeletonize_iter=SAM_LORA_CL_DICE_SKELETONIZE_ITERATIONS,
                           junction_patch_weight=SAM_LORA_JUNCTION_PATCH_WEIGHT,
                           junction_loss_type=SAM_LORA_JUNCTION_LOSS_TYPE,
                           topo_weight=SAM_LORA_TOPOLOGICAL_LOSS_WEIGHT)

    trainable_params = get_trainable_params(sam_lora)
    for name, p in trainable_params:
        print(f"Training: {name} with {p.numel()} parameters")
    print()

    img_enc_params = [p for name, p in trainable_params if name.startswith(
        "sam_model.image_encoder.")]
    other_params = [p for name, p in trainable_params if not name.startswith(
        "sam_model.image_encoder.")]

    if SAM_LORA_IMAGE_ENCODER_LR is not None and img_enc_params:
        optimizer = torch.optim.AdamW([
            {"params": img_enc_params, "lr": SAM_LORA_IMAGE_ENCODER_LR},
            {"params": other_params, "lr": SAM_LORA_LR},
        ])
        scheduler_max_lrs = [SAM_LORA_IMAGE_ENCODER_LR, SAM_LORA_LR]
    else:
        optimizer = torch.optim.AdamW(
            params=[p for _, p in trainable_params],
            lr=SAM_LORA_LR,
        )
        scheduler_max_lrs = SAM_LORA_LR

    scheduler = init_scheduler(
        optimizer, SAM_LORA_MAX_EPOCHS, len(trainloader), scheduler_max_lrs)

    min_validation_loss = float('inf')
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_DELTA, min_epochs=EARLY_STOPPING_MIN_EPOCHS)

    for epoch in range(SAM_LORA_MAX_EPOCHS):
        print(f"\nEpoch {epoch+1}/{SAM_LORA_MAX_EPOCHS}")

        # training
        sam_lora.train()

        total_training_loss = 0.0
        total_loss_terms = {}

        for batched_input, target_masks, heatmap_weights in trainloader:
            batched_input = batched_input.to(device)
            target_masks = target_masks.to(device)
            heatmap_weights = heatmap_weights.to(device)

            batched_input = get_batched_input_list(batched_input)
            batch_size = len(batched_input)

            optimizer.zero_grad()
            outputs = sam_lora(batched_input=batched_input,
                               multimask_output=False)
            output_logits = torch.cat([d["low_res_logits"]
                                      for d in outputs], dim=0)

            loss, bce_total, bce_base, bce_heatmap_weighted, focal_loss_total, focal_loss_base, focal_loss_heatmap_weighted, \
                dice_loss, cl_dice_loss, skeleton_recall_loss, junction_loss, topo_loss = loss_fn(
                    output_logits,
                    target_masks,
                    heatmap_weights if SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE > 0.0 else None
                )

            total_training_loss += loss.item() * batch_size
            total_loss_terms["BCE"] = total_loss_terms.get(
                "BCE", 0.0) + bce_total.item() * batch_size
            total_loss_terms["BCE (base)"] = total_loss_terms.get(
                "BCE (base)", 0.0) + bce_base.item() * batch_size
            total_loss_terms["BCE (heatmap weighted)"] = total_loss_terms.get(
                "BCE (heatmap weighted)", 0.0) + bce_heatmap_weighted.item() * batch_size
            total_loss_terms["Focal"] = total_loss_terms.get(
                "Focal", 0.0) + focal_loss_total.item() * batch_size
            total_loss_terms["Focal (base)"] = total_loss_terms.get(
                "Focal (base)", 0.0) + focal_loss_base.item() * batch_size
            total_loss_terms["Focal (heatmap weighted)"] = total_loss_terms.get(
                "Focal (heatmap weighted)", 0.0) + focal_loss_heatmap_weighted.item() * batch_size
            total_loss_terms["Dice"] = total_loss_terms.get(
                "Dice", 0.0) + dice_loss.item() * batch_size
            total_loss_terms["ClDice"] = total_loss_terms.get(
                "ClDice", 0.0) + cl_dice_loss.item() * batch_size
            total_loss_terms["Skeleton Recall"] = total_loss_terms.get(
                "Skeleton Recall", 0.0) + skeleton_recall_loss.item() * batch_size
            total_loss_terms["Junction"] = total_loss_terms.get(
                "Junction", 0.0) + junction_loss.item() * batch_size
            total_loss_terms["Topological"] = total_loss_terms.get(
                "Topological", 0.0) + topo_loss.item() * batch_size

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sam_lora.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        # validation
        sam_lora.eval()
        total_validation_loss = 0.0
        val_cldice_scores = []
        val_dice_scores = []

        with torch.no_grad():
            for batched_input, target_masks, heatmap_weights in validationloader:
                batched_input = batched_input.to(device)
                target_masks = target_masks.to(device)
                heatmap_weights = heatmap_weights.to(device)
                batched_input_list = get_batched_input_list(batched_input)

                outputs = sam_lora(batched_input=batched_input_list,
                                   multimask_output=False)
                output_logits = torch.cat([d["low_res_logits"]
                                          for d in outputs], dim=0)
                loss, _, _, _, _, _, _, _, _, _, _, _ = loss_fn(
                    output_logits,
                    target_masks,
                    heatmap_weights if SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE > 0.0 else None
                )
                total_validation_loss += loss.item() * len(batched_input_list)

                for i, out in enumerate(outputs):
                    pred_mask = out['masks'].squeeze().cpu()
                    gt_mask = target_masks[i].squeeze().cpu()

                    dice, _, clDice, _, _ = compute_metrics(pred_mask, gt_mask)
                    val_cldice_scores.append(clDice.item()) 
                    val_dice_scores.append(dice.item())

        # epoch metrics
        num_training_samples = len(trainloader) * trainloader.batch_size
        mean_training_loss = total_training_loss / num_training_samples
        print(f"    Training Loss: {mean_training_loss:.4f}")

        for loss_term_name, total_loss in total_loss_terms.items():
            mean_loss = total_loss / num_training_samples
            print(f"        {loss_term_name} Loss: {mean_loss:.4f}")

        num_validation_samples = len(
            validationloader) * validationloader.batch_size
        mean_validation_loss = total_validation_loss / num_validation_samples
        mean_val_cldice = float(np.mean(val_cldice_scores))
        mean_val_dice = float(np.mean(val_dice_scores))
        # composite score: clDice-heavy since that's our topology focus
        composite_score = 0.75 * mean_val_cldice + 0.25 * mean_val_dice

        print(f"    Validation Loss: {mean_validation_loss:.4f}")
        print(f"    Validation clDice: {mean_val_cldice:.4f}")
        print(f"    Validation Dice: {mean_val_dice:.4f}")
        print(f"    Validation Composite: {composite_score:.4f}")
        last_lrs = scheduler.get_last_lr()
        main_lr = last_lrs[-1]
        print(f"    Learning Rate: {main_lr:.6f}")
        if len(last_lrs) > 1:
            print(f"    Image Encoder Learning Rate: {last_lrs[0]:.6f}")

        if mean_validation_loss < min_validation_loss:
            min_validation_loss = mean_validation_loss
            print("    New minimum validation loss achieved, saving model parameters")
            save_params(sam_lora, wandb_run, "minloss")

        if USE_WANDB and wandb_run is not None:
            wandb_log = {
                "train/loss": mean_training_loss,
                "validation/loss": mean_validation_loss,
                "validation/clDice": mean_val_cldice,
                "validation/dice": mean_val_dice,
                "validation/composite": composite_score,
                "learning_rate": main_lr,
            }
            if len(last_lrs) > 1:
                wandb_log["learning_rate_image_encoder"] = last_lrs[0]
            wandb_run.log(wandb_log)

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
                                 bce_weight=SAM_LORA_BCE_LOSS_WEIGHT, focal_weight=SAM_LORA_FOCAL_LOSS_WEIGHT,
                                 dice_weight=SAM_LORA_DICE_LOSS_WEIGHT, cl_dice_weight=SAM_LORA_CL_DICE_LOSS_WEIGHT,
                                 skeleton_recall_weight=SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT,
                                 focal_alpha=SAM_LORA_FOCAL_ALPHA, focal_gamma=SAM_LORA_FOCAL_GAMMA,
                                 skeletonize_iter=SAM_LORA_CL_DICE_SKELETONIZE_ITERATIONS)
        for metric_name, metric_value in metrics.items():
            print(f"        {metric_name}: {metric_value:.4f}")
            wandb_run.summary[metric_name] = metric_value

    wandb_run.tags = list(set(wandb_run.tags) | {EVALUATED_TAG})


def train_evaluate():
    # We write sweep-controlled values back to the module globals so that
    # init_model(), init_data_loaders(), train(), and evaluate_checkpoints()
    # (which all read the globals directly) pick up the sweep-sampled values.
    global SAM_LORA_LR, SAM_LORA_RANK, SAM_LORA_IMAGE_ENCODER_LR,  \
        SAM_LORA_FINETUNE_MASK_DECODER, SAM_LORA_FINETUNE_PROMPT_ENCODER, \
        SAM_LORA_FINETUNE_IMAGE_ENCODER, SAM_LORA_FINETUNE_IMAGE_ENCODER_N_BLOCKS,   \
        SAM_LORA_BCE_LOSS_WEIGHT, SAM_LORA_FOCAL_LOSS_WEIGHT, \
        SAM_LORA_DICE_LOSS_WEIGHT, \
        SAM_LORA_CL_DICE_LOSS_WEIGHT, SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT, \
        SAM_LORA_TOPOLOGICAL_LOSS_WEIGHT, \
        SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE, SAM_LORA_JUNCTION_PATCH_WEIGHT, \
        DATASET_DOWNSAMPLE_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_run = None
    if USE_WANDB:
        wandb.login(key=WANDB_API_KEY)

        finetuned_modules = get_cfg_string_from_finetuned_components(
            SAM_LORA_FINETUNE_IMAGE_ENCODER, SAM_LORA_FINETUNE_MASK_DECODER,
            SAM_LORA_FINETUNE_PROMPT_ENCODER, SAM_LORA_FINETUNE_IMAGE_ENCODER_N_BLOCKS)

        # Init wandb with ONLY non-swept metadata.  For sweep runs the
        # agent pre-populates wandb_run.config with its sampled values
        # *before* we touch it — we read those out below.
        wandb_run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=os.environ.get("WANDB_NAME", f"SAM_LoRA_Finetuning_{RUN_DATETIME_STR}"),
        )

        # --- Step 1: If sweep run, override globals from sweep-sampled config ---
        if wandb_run.sweep_id:
            cfg = wandb_run.config
            SAM_LORA_LR = cfg["learning_rate"] if "learning_rate" in cfg else SAM_LORA_LR
            SAM_LORA_RANK = cfg["lora_rank"] if "lora_rank" in cfg else SAM_LORA_RANK
            SAM_LORA_BCE_LOSS_WEIGHT = cfg["bce_loss_weight"] if "bce_loss_weight" in cfg else SAM_LORA_BCE_LOSS_WEIGHT
            SAM_LORA_FOCAL_LOSS_WEIGHT = cfg["focal_loss_weight"] if "focal_loss_weight" in cfg else SAM_LORA_FOCAL_LOSS_WEIGHT
            SAM_LORA_DICE_LOSS_WEIGHT = cfg["dice_loss_weight"] if "dice_loss_weight" in cfg else SAM_LORA_DICE_LOSS_WEIGHT
            SAM_LORA_CL_DICE_LOSS_WEIGHT = cfg["cl_dice_loss_weight"] if "cl_dice_loss_weight" in cfg else SAM_LORA_CL_DICE_LOSS_WEIGHT
            SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT = cfg[
                "skeleton_recall_loss_weight"] if "skeleton_recall_loss_weight" in cfg else SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT
            SAM_LORA_TOPOLOGICAL_LOSS_WEIGHT = cfg["topological_loss_weight"] if "topological_loss_weight" in cfg else SAM_LORA_TOPOLOGICAL_LOSS_WEIGHT
            SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE = cfg[
                "junction_heatmap_weight_scale"] if "junction_heatmap_weight_scale" in cfg else SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE
            SAM_LORA_JUNCTION_PATCH_WEIGHT = cfg["junction_patch_weight"] if "junction_patch_weight" in cfg else SAM_LORA_JUNCTION_PATCH_WEIGHT
            SAM_LORA_IMAGE_ENCODER_LR = cfg["learning_rate_image_encoder"] if "learning_rate_image_encoder" in cfg else SAM_LORA_IMAGE_ENCODER_LR

            if "finetuned_modules" in cfg:
                finetuned_modules = ast.literal_eval(cfg["finetuned_modules"])
                SAM_LORA_FINETUNE_IMAGE_ENCODER, SAM_LORA_FINETUNE_MASK_DECODER, \
                    SAM_LORA_FINETUNE_PROMPT_ENCODER, do_finetune_img_encoder_blocks = get_finetuned_components_from_cfg(
                        finetuned_modules)
                if do_finetune_img_encoder_blocks and "finetune_img_encoder_n_blocks" in cfg:
                    SAM_LORA_FINETUNE_IMAGE_ENCODER_N_BLOCKS = cfg["finetune_img_encoder_n_blocks"]

            if "dataset_downsample_size" in cfg:
                raw_ds = int(cfg["dataset_downsample_size"])
                DATASET_DOWNSAMPLE_SIZE = (
                    raw_ds, raw_ds) if raw_ds != 0 else None

        print(
            f"[sweep] Run {wandb_run.id} — overriding globals from sweep config:")
        for k, v in sorted(cfg.items()):
            print(f"  {k}: {v}")

        # --- Step 2: Record ALL hyperparameters (now final) to the run config ---
        wandb_run.config.update({
            "learning_rate": SAM_LORA_LR,
            "learning_rate_image_encoder": SAM_LORA_IMAGE_ENCODER_LR if SAM_LORA_IMAGE_ENCODER_LR is not None else SAM_LORA_LR,
            "learning_rate_scheduler": SAM_LORA_SCHEDULER_TYPE,
            "SAM_model_type": SAM_LORA_MODEL_TYPE,
            "SAM_checkpoint": SAM_LORA_MODEL_CHECKPOINT,
            "lora_rank": SAM_LORA_RANK,
            "finetuned_modules": str(finetuned_modules),
            "finetune_img_encoder_n_blocks": SAM_LORA_FINETUNE_IMAGE_ENCODER_N_BLOCKS,
            "dataset": DATASET_NAME,
            "epochs": SAM_LORA_MAX_EPOCHS,
            "batch_size": SAM_LORA_BATCH_SIZE,
            "bce_loss_weight": SAM_LORA_BCE_LOSS_WEIGHT,
            "focal_loss_weight": SAM_LORA_FOCAL_LOSS_WEIGHT,
            "focal_alpha": SAM_LORA_FOCAL_ALPHA,
            "focal_gamma": SAM_LORA_FOCAL_GAMMA,
            "dice_loss_weight": SAM_LORA_DICE_LOSS_WEIGHT,
            "cl_dice_loss_weight": SAM_LORA_CL_DICE_LOSS_WEIGHT,
            "cl_dice_skeletonize_iterations": SAM_LORA_CL_DICE_SKELETONIZE_ITERATIONS,
            "skeleton_recall_loss_weight": SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT,
            "junction_heatmap_weight_scale": SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE,
            "junction_patch_weight": SAM_LORA_JUNCTION_PATCH_WEIGHT,
            "junction_loss_type": SAM_LORA_JUNCTION_LOSS_TYPE,
            "topological_loss_weight": SAM_LORA_TOPOLOGICAL_LOSS_WEIGHT,
            "dataset_downsample_size": DATASET_DOWNSAMPLE_SIZE[0] if DATASET_DOWNSAMPLE_SIZE else None,
        }, allow_val_change=True)

        run_out_dir = get_init_run_out_dir(wandb_run)
        with open(str(run_out_dir / "wandb_run_id.txt"), "w") as f:
            f.write(wandb_run.id)

    # model and data init happen AFTER globals are updated from sweep config
    sam_lora = init_model(device)
    trainloader, validationloader, train_size, val_size = init_data_loaders()

    if USE_WANDB and wandb_run is not None:
        base_training_images = get_base_images(imgs_dir=TRAIN_IMAGES_DIR)
        wandb_run.config.update({
            "train_set_size": train_size,
            "val_set_size": val_size,
            "num_base_training_images": len(base_training_images),
            "base_training_images": str(base_training_images),
            "trainable_parameters": sum(p.numel() for _, p in get_trainable_params(sam_lora)),
        }, allow_val_change=True)

    train(sam_lora, wandb_run, trainloader, validationloader, device)

    if USE_WANDB and wandb_run is not None:
        evaluate_checkpoints(wandb_run, device)
        wandb_run.finish()


if __name__ == "__main__":
    seed_everything(SEED)
    train_evaluate()
