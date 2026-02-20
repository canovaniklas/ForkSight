"""
W&B Sweep entrypoint for SLURM mode.

The wandb agent sets WANDB_RUN_ID and WANDB_SWEEP_ID in the environment before
invoking this script. wandb.init() inside train_evaluate() picks these up
automatically, connects to the correct sweep run, and populates wandb.config
with the sampled hyperparameters. train_evaluate() then reads those values
from wandb.config and overrides the relevant globals before model/data init.
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    # Ensure local wandb dirs exist before the agent starts writing to them
    sweep_id = os.environ.get("WANDB_SWEEP_ID")
    Path("wandb").mkdir(parents=True, exist_ok=True)
    if sweep_id:
        Path(f"wandb/sweep-{sweep_id}").mkdir(parents=True, exist_ok=True)

    # Import training module after the agent has set WANDB_RUN_ID / WANDB_SWEEP_ID.
    # load_segmentation_env() runs at module level inside sam_lora_train — the base
    # .env was already sourced by the SLURM job script before this process started.
    from Segmentation.SAM import sam_lora_train as T

    # train_evaluate() calls wandb.init() (connects to the sweep run via WANDB_RUN_ID),
    # reads wandb.config, overrides globals, trains, evaluates, and finishes the run.
    T.seed_everything(T.SEED)
    T.train_evaluate()


if __name__ == "__main__":
    main()
