"""
Create two W&B Bayesian-optimization sweeps for hyperparameter tuning:
    - clDice as the topology-aware loss
    - Skeleton Recall as the topology-aware loss

Prints sweep IDs and instructions for starting agents.
"""

import os
import wandb
from Segmentation.Util.env_utils import load_segmentation_env

load_segmentation_env()

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ForkSight-SAM")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# shared search space (tuned in both sweeps)
SHARED_PARAMETERS = {
    "SAM_LORA_BCE_LOSS_WEIGHT": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
    "SAM_LORA_DICE_LOSS_WEIGHT": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
    "SAM_LORA_FOCAL_LOSS_WEIGHT": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
    "SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 2000.0,
    },
    "SAM_LORA_LR": {
        "distribution": "log_uniform_values",
        "min": 1e-5,
        "max": 1e-2,
    },
    "SAM_LORA_RANK": {
        "distribution": "categorical",
        "values": [2, 4, 8, 16],
    },
    "DATASET_DOWNSAMPLE_SIZE": {
        "distribution": "categorical",
        "values": [0, 256],
    },
}

# fixed parameters (identical in both sweeps)
SHARED_FIXED = {
    "SAM_LORA_TOPOLOGICAL_LOSS_WEIGHT": {"value": 0.0},
    "SAM_LORA_JUNCTION_PATCH_WEIGHT": {"value": 0.0},
}


def make_sweep_config(name: str, topo_loss_params: dict) -> dict:
    parameters = {}
    parameters.update(SHARED_PARAMETERS)
    parameters.update(SHARED_FIXED)
    parameters.update(topo_loss_params)

    return {
        "name": name,
        "method": "bayes",
        "metric": {
            "name": "validation/composite_metric",
            "goal": "maximize",
        },
        "parameters": parameters,
        "command": [
            "${env}",
            "python",
            "-u",
            "-m",
            "Segmentation.SAM.sweep_wandb_train",
            "${args_json_file}",
        ],
    }


# Sweep 1: clDice
CLDICE_SWEEP = make_sweep_config(
    name="sweep-cldice",
    topo_loss_params={
        "SAM_LORA_CL_DICE_LOSS_WEIGHT": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
        },
        "SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT": {"value": 0.0},
    },
)

# Sweep 2: Skeleton Recall
SKELETON_RECALL_SWEEP = make_sweep_config(
    name="sweep-skeleton-recall",
    topo_loss_params={
        "SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
        },
        "SAM_LORA_CL_DICE_LOSS_WEIGHT": {"value": 0.0},
    },
)


def main():
    wandb.login(key=WANDB_API_KEY)

    sweep_id_cldice = wandb.sweep(
        CLDICE_SWEEP, entity=WANDB_ENTITY, project=WANDB_PROJECT
    )
    sweep_id_skel = wandb.sweep(
        SKELETON_RECALL_SWEEP, entity=WANDB_ENTITY, project=WANDB_PROJECT
    )

    print("\n" + "=" * 60)
    print("Sweeps created successfully!")
    print("=" * 60)
    print(
        f"\n  clDice sweep ID:           {WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id_cldice}")
    print(
        f"  Skeleton Recall sweep ID:  {WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id_skel}")
    print("\nTo launch agents (run from repo root, inside tmux/screen):")
    print(
        f"\n  bash Segmentation/SAM/sweep_wandb_submit.sh {WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id_cldice}")
    print(
        f"  bash Segmentation/SAM/sweep_wandb_submit.sh {WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id_skel}")
    print("\nEach agent runs on the login node and submits a separate SLURM")
    print("job per trial. Run multiple agents (in separate tmux panes) to")
    print("run trials in parallel. Optionally pass a max trial count:")
    print(
        f"\n  bash Segmentation/SAM/sweep_wandb_submit.sh {WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id_cldice} 20\n")


if __name__ == "__main__":
    main()
