"""
Create two W&B Bayesian-optimization sweeps for hyperparameter tuning:
    - clDice as the topology-aware loss (Skeleton Recall fixed to 0)
    - Skeleton Recall as the topology-aware loss (clDice fixed to 0)

HutopoLoss is fixed to 0 in both sweeps.

SLURM-mode workflow:
    1. python -m Segmentation.SAM.sweep_wandb_init_slurm   # register sweeps (this file)
    2. bash Segmentation/SAM/sweep_wandb_submit_slurm.sh <entity/project/sweep_id> <n_trials>
       (repeat for each sweep ID printed below)

Parameter names MUST match both:
  - the cfg.get("key") calls in train_evaluate() in sam_lora_train.py
  - the keys in wandb.init(config={...}) in train_evaluate()
The sweep agent merges its params into the existing config by key name, so
sweep values automatically override the env-var defaults with no duplicates.
"""

import os
import wandb
from Segmentation.Util.env_utils import load_segmentation_env

load_segmentation_env()

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ForkSight-SAM")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Shared search space, identical across both sweeps.
# Key names match wandb.init(config={}) and cfg.get() in train_evaluate()
SHARED_PARAMETERS = {
    "learning_rate": {
        "distribution": "log_uniform_values",
        "min": 1e-5,
        "max": 1e-2,
    },
    "learning_rate_image_encoder": {
        "distribution": "log_uniform_values",
        "min": 1e-6,
        "max": 1e-3,
    },
    "lora_rank": {
        "distribution": "categorical",
        "values": [4, 8, 16],
    },
    "bce_loss_weight": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.5,
    },
    "dice_loss_weight": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
    "topological_loss_weight": {"value": 0.0},
    "junction_patch_weight": {"value": 0.0},
    "junction_heatmap_weight_scale": {"value": 0.0},
}

SWEEP_COMMAND = [
    "${env}",
    "python",
    "-u",
    "-m",
    "Segmentation.SAM.sweep_wandb_direct",
]

SWEEP_METRIC = {
    "name": "validation/composite",
    "goal": "maximize",
}

EARLY_TERMINATE = {
    "type": "hyperband",
    "min_iter": 25,
    "eta": 3,
}


def make_sweep_config(name: str, topo_loss_params: dict, img_encoder_params: dict) -> dict:
    parameters = {}
    parameters.update(SHARED_PARAMETERS)
    parameters.update(topo_loss_params)
    parameters.update(img_encoder_params)
    return {
        "name": name,
        "method": "bayes",
        "metric": SWEEP_METRIC,
        "early_terminate": EARLY_TERMINATE,
        "parameters": parameters,
        "command": SWEEP_COMMAND,
    }


clDice_config = {
    "cl_dice_loss_weight": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
    "skeleton_recall_loss_weight": {"value": 0.0},
}

skeleton_recall_config = {
    "skeleton_recall_loss_weight": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
    "cl_dice_loss_weight": {"value": 0.0},
}

img_encoder_lora_config = {
    "finetuned_modules": {"value": "['image_encoder_lora', 'mask_decoder', 'prompt_encoder']"},
    "finetune_img_encoder_n_blocks": {"value": 0},
}

img_encoder_n_blocks_config = {
    "finetuned_modules": {"value": "['image_encoder_last_N_blocks_full', 'mask_decoder', 'prompt_encoder']"},
    "finetune_img_encoder_n_blocks": {
        "distribution": "categorical",
        "values": [1, 2, 3, 4],
    },
}


# Sweep 1:  + image encoder LoRA
CLDICE_LORA_SWEEP = make_sweep_config(
    name="sweep-cldice-lora",
    topo_loss_params=clDice_config,
    img_encoder_params=img_encoder_lora_config,
)

# Sweep 2: clDice  + full fine-tuning of last N image encoder blocks
CLDICE_N_BLOCKS_SWEEP = make_sweep_config(
    name="sweep-clDice-n-blocks",
    topo_loss_params=clDice_config,
    img_encoder_params=img_encoder_n_blocks_config,
)

# Sweep 3: skeleton recall + image encoder LoRA
SKELETON_RECALL_LORA_SWEEP = make_sweep_config(
    name="sweep-skeleton-recall-lora",
    topo_loss_params=skeleton_recall_config,
    img_encoder_params=img_encoder_lora_config,
)

# Sweep 4: skeleton recall + full fine-tuning of last N image encoder blocks
SKELETON_RECALL_N_BLOCKS_SWEEP = make_sweep_config(
    name="sweep-skeleton-recall-n-blocks",
    topo_loss_params=skeleton_recall_config,
    img_encoder_params=img_encoder_n_blocks_config,
)


def main():
    wandb.login(key=WANDB_API_KEY)

    sweep_id_cldice_lora = wandb.sweep(
        CLDICE_LORA_SWEEP, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    sweep_id_cldice_n_blocks = wandb.sweep(
        CLDICE_N_BLOCKS_SWEEP, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    sweep_id_skel_lora = wandb.sweep(
        SKELETON_RECALL_LORA_SWEEP, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    sweep_id_skel_n_blocks = wandb.sweep(
        SKELETON_RECALL_N_BLOCKS_SWEEP, entity=WANDB_ENTITY, project=WANDB_PROJECT)

    path_cldice_lora = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id_cldice_lora}"
    path_cldice_n_blocks = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id_cldice_n_blocks}"
    path_skel_lora = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id_skel_lora}"
    path_skel_n_blocks = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id_skel_n_blocks}"

    print("\n" + "=" * 65)
    print("Sweeps registered successfully.")
    print("=" * 65)
    print(f"\n  clDice (LoRA):           {path_cldice_lora}")
    print(f"  clDice (N blocks):       {path_cldice_n_blocks}")
    print(f"  Skeleton Recall (LoRA):  {path_skel_lora}")
    print(f"  Skeleton Recall (N blocks):  {path_skel_n_blocks}")

    print("\nTo submit SLURM jobs in WAVES (for better exploitation) (run from repo root):")
    print(
        f"  bash Segmentation/SAM/sweep_wandb_submit_waves.sh {path_cldice_lora} <wave1_size> <later_wave_size> <num_later_waves>")
    print(
        f"  bash Segmentation/SAM/sweep_wandb_submit_waves.sh {path_cldice_n_blocks} <wave1_size> <later_wave_size> <num_later_waves>")
    print(
        f"  bash Segmentation/SAM/sweep_wandb_submit_waves.sh {path_skel_lora} <wave1_size> <later_wave_size> <num_later_waves>")
    print(
        f"  bash Segmentation/SAM/sweep_wandb_submit_waves.sh {path_skel_n_blocks                       } <wave1_size> <later_wave_size> <num_later_waves>")
    print("     <wave1_size>: number of parallel trials in the first wave")
    print("     <later_wave_size>: number of parallel trials in each subsequent wave")
    print("     <num_later_waves>: number of subsequent waves (after the first wave)")

    print(
        f"\nMonitor at: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/sweeps\n")


if __name__ == "__main__":
    main()
