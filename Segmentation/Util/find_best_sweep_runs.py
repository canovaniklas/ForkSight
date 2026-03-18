import os
import wandb

from Environment.env_utils import load_forksight_env

load_forksight_env()

WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_SAM_PROJECT = os.getenv("WANDB_SAM_PROJECT")

api = wandb.Api()
sweeps = api.project(name=WANDB_SAM_PROJECT, entity=WANDB_ENTITY).sweeps()

for sweep in sweeps:
    best = sweep.best_run()
    if not best:
        print(f"Sweep: {sweep.id} | No best run found")
        continue

    # Check how far the run actually got
    epochs = best.summary.get("epoch", "N/A")
    steps = best.summary.get("_step", "N/A")
    max_epochs = best.config.get(
        "epochs", best.config.get("max_epochs", "N/A"))

    # Check if the sweep had early termination enabled
    early_term = sweep.config.get("early_terminate", None)

    # filter optimized params of best run
    sweep_params = sweep.config.get("parameters", {}).keys()
    best_config = {k: v for k, v in best.config.items() if k in sweep_params}

    print(f"Sweep: {sweep.name} ({sweep.id})")
    print(f"  Best run:       {best.name} ({best.id})")
    print(f"  Epochs:         {epochs} / {max_epochs}")
    print(f"  Steps:          {steps}")
    print(f"  Early stopping: {'enabled' if early_term else 'disabled'}")
    print(f"  State:          {best.state}")
    print(f"  Best params:    {best_config}")
    print()
