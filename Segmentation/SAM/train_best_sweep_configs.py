"""Submit one full-length training job per sweep, using the best run's hyperparameters.

For every sweep in the W&B project this script:
  1. Identifies the best run (by the sweep's optimisation metric).
  2. Extracts only the parameters that were actually swept.
  3. Maps them to SAM_LORA_* environment variables, expanding
     the `finetuned_modules` list into individual SAM_LORA_FINETUNE_* flags.
  4. Submits a SLURM job via SAM_LoRA_train_submit.sh, which accepts
     KEY=value overrides as positional arguments.

All sweeps are submitted in parallel (no inter-job dependencies).

Usage:
    # Dry-run — print sbatch commands without submitting:
    python -m Segmentation.SAM.train_best_sweep_configs

    # Actually submit:
    python -m Segmentation.SAM.train_best_sweep_configs --submit
"""

import ast
import os
import subprocess
import sys

import wandb

from Environment.env_utils import load_forksight_env


SUBMIT_SCRIPT = "Segmentation/SAM/SAM_LoRA_train_submit.sh"

# Direct 1-to-1 mapping: sweep param name → SAM_LORA_* env var.
# Params that require special handling (finetuned_modules,
# finetune_img_encoder_n_blocks) are dealt with in _build_env_overrides().
_PARAM_TO_ENV: dict[str, str] = {
    "learning_rate": "SAM_LORA_LR",
    "learning_rate_image_encoder": "SAM_LORA_IMAGE_ENCODER_LR",
    "lora_rank": "SAM_LORA_RANK",
    "bce_loss_weight": "SAM_LORA_BCE_LOSS_WEIGHT",
    "focal_loss_weight": "SAM_LORA_FOCAL_LOSS_WEIGHT",
    "dice_loss_weight": "SAM_LORA_DICE_LOSS_WEIGHT",
    "cl_dice_loss_weight": "SAM_LORA_CL_DICE_LOSS_WEIGHT",
    "skeleton_recall_loss_weight": "SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT",
    "topological_loss_weight": "SAM_LORA_TOPOLOGICAL_LOSS_WEIGHT",
    "junction_heatmap_weight_scale": "SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE",
    "junction_patch_weight": "SAM_LORA_JUNCTION_PATCH_WEIGHT",
}


def _build_env_overrides(best_config: dict) -> list[str]:
    """Convert a best-run config dict to KEY=value override strings for sbatch.

    finetuned_modules is expanded into the individual SAM_LORA_FINETUNE_*
    boolean flags that sam_lora_train.py reads in non-sweep mode.
    """
    overrides: dict[str, str] = {}

    for param, value in best_config.items():
        if param in _PARAM_TO_ENV:
            overrides[_PARAM_TO_ENV[param]] = str(value)

        elif param == "finetuned_modules":
            modules: list[str] = (
                ast.literal_eval(value) if isinstance(value, str) else value
            )

            overrides["SAM_LORA_FINETUNE_IMAGE_ENCODER"] = str(
                "image_encoder_lora" in modules)
            overrides["SAM_LORA_FINETUNE_MASK_DECODER"] = str(
                "mask_decoder" in modules)
            overrides["SAM_LORA_FINETUNE_PROMPT_ENCODER"] = str(
                "prompt_encoder" in modules)
            if "image_encoder_last_N_blocks_full" not in modules:
                overrides["SAM_LORA_FINETUNE_IMAGE_ENCODER_N_BLOCKS"] = "0"

        elif param == "finetune_img_encoder_n_blocks":
            overrides["SAM_LORA_FINETUNE_IMAGE_ENCODER_N_BLOCKS"] = str(
                int(value))

    return [f"{k}={v}" for k, v in overrides.items()]


def main(submit: bool) -> None:
    load_forksight_env()

    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_SAM_PROJECT")

    api = wandb.Api()
    sweeps = api.project(name=project, entity=entity).sweeps()

    submitted_jobs: list[str] = []
    for sweep in sweeps:
        best = sweep.best_run()

        sweep_params = set(sweep.config.get("parameters", {}).keys())
        best_config = {k: v for k, v in best.config.items()
                       if k in sweep_params}
        env_overrides = _build_env_overrides(best_config)

        epochs = best.summary.get("epoch", "?")
        max_epochs = best.config.get(
            "epochs", best.config.get("max_epochs", "?"))

        print(f"\nSweep:    {sweep.name} ({sweep.id})")
        print(
            f"Best run: {best.name} ({best.id})  —  epoch {epochs}/{max_epochs}")
        print(f"Params:   {' '.join(env_overrides)}")

        cmd = [
            "sbatch",
            f"--job-name={sweep.name}-BEST",
            SUBMIT_SCRIPT,
            *env_overrides,
        ]

        if submit:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr.strip()}")
            else:
                job_id = result.stdout.strip().split()[-1]
                submitted_jobs.append(job_id)
                print(f"  Submitted: job {job_id}")
        else:
            print(f"  [dry-run] {' '.join(cmd)}")

    if submit and submitted_jobs:
        print(
            f"\nSubmitted {len(submitted_jobs)} job(s): {', '.join(submitted_jobs)}")
        print(f"Monitor: https://wandb.ai/{entity}/{project}")
    elif not submit:
        print(
            f"\nDry run — pass --submit to actually submit {len(submitted_jobs or sweeps)} job(s).")


if __name__ == "__main__":
    main(submit="--submit" in sys.argv)
