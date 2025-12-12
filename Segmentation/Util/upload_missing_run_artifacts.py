import os
import wandb
from pathlib import Path

from Segmentation.Util.env_utils import load_segmentation_env

load_segmentation_env()

MODEL_OUT_DIR = os.getenv("MODEL_OUT_DIR")

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ForkSight-SAM")

if MODEL_OUT_DIR is None or not Path(MODEL_OUT_DIR).is_dir():
    raise ValueError(
        "MODEL_OUT_DIR environment variable is not set or is not a valid directory.")

api = wandb.Api()

for run_dir in Path(MODEL_OUT_DIR).iterdir():
    if not run_dir.is_dir():
        continue

    print("Processing directory:", run_dir)

    run_id_file = run_dir / "wandb_run_id.txt"
    if not run_id_file.exists():
        continue

    run_id = run_id_file.read_text().strip()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")

    existing_artifacts = [a.name for a in run.logged_artifacts()]
    print(f"Run {run.id} has existing artifacts: {existing_artifacts}")

    for param_file in run_dir.glob("*.pt"):
        artifact_name = f"{run_dir.name}_{param_file.stem}"
        if artifact_name not in existing_artifacts:
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(param_file)
            run.log_artifact(artifact)
            print(f"Uploaded {artifact_name} for run {run.id}")
