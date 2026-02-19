"""
W&B Sweep agent entry-point (runs on the login node).

Called by `wandb agent` for each trial. Instead of training directly,
it submits a SLURM job and blocks until that job finishes, so every
trial gets its own GPU allocation.

Flow:
  1. wandb.init() joins the sweep run → populates wandb.config
  2. Sweep hyperparameters are written to a temporary env file
  3. A SLURM job (sweep_wandb_run.sh) is submitted with `sbatch --wait`
  4. The SLURM job runs sweep_wandb_worker.py which resumes the wandb
     run and executes the actual training
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = SCRIPT_DIR / "sweep_wandb_run.sh"


def main():
    # 1. join the sweep run
    run = wandb.init()
    run_id = run.id
    sweep_id = run.sweep_id
    entity = run.entity
    project = run.project

    # 2. write sweep params to a temporary env file
    env_lines = []
    for key, value in wandb.config.items():
        if key == "DATASET_DOWNSAMPLE_SIZE" and (value == 0 or value is None):
            # 0 is our sentinel for "no downsampling"
            env_lines.append(f"unset {key}")
        else:
            env_lines.append(f"export {key}='{value}'")

    params_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="sweep_params_", suffix=".env",
        delete=False, dir=os.environ.get("TMPDIR", "/tmp"),
    )
    params_file.write("\n".join(env_lines) + "\n")
    params_file.close()

    # finish the run here — the worker will resume it on the compute node
    wandb.finish(quiet=True)

    # 3. submit SLURM job and block until it completes
    cmd = [
        "sbatch", "--wait",
        f"--job-name=sweep-trial-{run_id}",
        str(RUN_SCRIPT),
        entity, project, run_id, sweep_id, params_file.name,
    ]
    print(f"[sweep agent] submitting: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    # clean up
    os.unlink(params_file.name)

    if result.returncode != 0:
        print(f"[sweep agent] SLURM job failed (exit {result.returncode})")
        sys.exit(result.returncode)

    print(f"[sweep agent] trial {run_id} completed")


if __name__ == "__main__":
    main()
