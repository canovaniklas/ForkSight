#!/usr/bin/env python3
"""
Submit a SLURM hyperparameter sweep from a YAML config.

Usage:
    python sweep_submit.py                           # uses default fixed_sweep_config.yaml
    python sweep_submit.py --config my_sweep.yaml    # uses custom config
    python sweep_submit.py --dry-run                 # preview without submitting

The YAML config supports two modes:
  - grid:  cartesian product of parameter lists
  - runs:  explicit list of parameter dicts
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml

SUBMIT_SCRIPT = Path(__file__).parent / "SAM_LoRA_train_submit.sh"
DEFAULT_CONFIG = Path(__file__).parent / "fixed_sweep_config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_runs(config: dict) -> list[dict[str, str]]:
    """Build the list of parameter-override dicts from the config."""
    if "runs" in config:
        return config["runs"]

    if "grid" in config:
        grid = config["grid"]
        keys = list(grid.keys())
        values = [grid[k] if isinstance(grid[k], list) else [
            grid[k]] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    print("Error: config must contain either 'grid' or 'runs'.")
    sys.exit(1)


def run_label(params: dict) -> str:
    """Short human-readable label for a parameter combination."""
    parts = []
    for k, v in params.items():
        if (type(v) == int or type(v) == float) and v == 0:
            continue  # skip zero values for brevity
        short_key = k.replace("SAM_LORA_", "").replace("LOSS_", "")
        parts.append(f"{short_key}={v}")
    return "_".join(parts)


def submit(params: dict, dry_run: bool = False) -> str | None:
    """Submit a single SLURM job with the given parameter overrides."""
    args = [f"{k}={v}" for k, v in params.items()]
    label = run_label(params)
    job_name = f"SAM-LoRA-train_sweep"

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        str(SUBMIT_SCRIPT),
        *args,
    ]

    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return None

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.strip()}")
        return None

    job_id = result.stdout.strip().split()[-1]
    print(f"  Submitted job {job_id}: {label}")
    return job_id


def main():
    parser = argparse.ArgumentParser(
        description="Submit a SLURM hyperparameter sweep.")
    parser.add_argument("--config", type=Path,
                        default=DEFAULT_CONFIG, help="Path to sweep YAML config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    args = parser.parse_args()

    config = load_config(args.config)
    runs = build_runs(config)

    print(f"Sweep: {len(runs)} run(s) from {args.config.name}\n")
    for i, params in enumerate(runs, 1):
        print(f"Run {i}/{len(runs)}: {run_label(params)}")

    print()
    job_ids = []
    for params in runs:
        job_id = submit(params, dry_run=args.dry_run)
        if job_id:
            job_ids.append(job_id)

    if not args.dry_run and job_ids:
        print(f"\nSubmitted {len(job_ids)} job(s): {', '.join(job_ids)}")


if __name__ == "__main__":
    main()
