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
    "lora_rank": {
        "distribution": "categorical",
        "values": [4, 8, 16],
    },
    "bce_loss_weight": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.5,
    },
    "focal_loss_weight": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.5,
    },
    "dice_loss_weight": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
    "junction_heatmap_weight_scale": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 2000.0,
    },
    "dataset_downsample_size": {
        "distribution": "categorical",
        "values": [0, 256, 512],
    },
}

# Fixed across both sweeps
SHARED_FIXED = {
    "topological_loss_weight": {"value": 0.0},
    "junction_patch_weight": {"value": 0.0},
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
    "min_iter": 30,
    "eta": 3,
}


def make_sweep_config(name: str, topo_loss_params: dict) -> dict:
    parameters = {}
    parameters.update(SHARED_PARAMETERS)
    parameters.update(SHARED_FIXED)
    parameters.update(topo_loss_params)
    return {
        "name": name,
        "method": "bayes",
        "metric": SWEEP_METRIC,
        "early_terminate": EARLY_TERMINATE,
        "parameters": parameters,
        "command": SWEEP_COMMAND,
    }


# Sweep 1: clDice [0, 1], fix skeleton_recall to 0
CLDICE_SWEEP = make_sweep_config(
    name="sweep-cldice",
    topo_loss_params={
        "cl_dice_loss_weight": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
        },
        "skeleton_recall_loss_weight": {"value": 0.0},
    },
)

# Sweep 2: Skeleton Recall [0, 1], fix clDice to 0
SKELETON_RECALL_SWEEP = make_sweep_config(
    name="sweep-skeleton-recall",
    topo_loss_params={
        "skeleton_recall_loss_weight": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
        },
        "cl_dice_loss_weight": {"value": 0.0},
    },
)


def main():
    wandb.login(key=WANDB_API_KEY)

    sweep_id_cldice = wandb.sweep(
        CLDICE_SWEEP, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    sweep_id_skel = wandb.sweep(
        SKELETON_RECALL_SWEEP, entity=WANDB_ENTITY, project=WANDB_PROJECT)

    path_cldice = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id_cldice}"
    path_skel = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id_skel}"

    print("\n" + "=" * 65)
    print("Sweeps registered successfully.")
    print("=" * 65)
    print(f"\n  clDice:           {path_cldice}")
    print(f"  Skeleton Recall:  {path_skel}")

    print("\nTo submit N SLURM jobs per sweep that MAY RUN IN PARALLEL (run from repo root):")
    print(
        f"bash Segmentation/SAM/sweep_wandb_submit_slurm.sh {path_cldice} <n_trials>")
    print(
        f"  bash Segmentation/SAM/sweep_wandb_submit_slurm.sh {path_skel} <n_trials>")

    print("\nTo submit SLURM jobs in WAVES (for better exploitation) (run from repo root):")
    print(
        f"bash Segmentation/SAM/sweep_wandb_submit_waves.sh {path_cldice} <wave1_size> <later_wave_size> <num_later_waves>")
    print(
        f"  bash Segmentation/SAM/sweep_wandb_submit_waves.sh {path_skel} <wave1_size> <later_wave_size> <num_later_waves>")
    print("     <wave1_size>: number of parallel trials in the first wave")
    print("     <later_wave_size>: number of parallel trials in each subsequent wave")
    print("     <num_later_waves>: number of subsequent waves (after the first wave)")

    print(
        f"\nMonitor at: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/sweeps\n")


if __name__ == "__main__":
    main()
