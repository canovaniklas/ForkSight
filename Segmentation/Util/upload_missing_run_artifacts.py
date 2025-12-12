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
        "MODEL_OUT_DIR environment variable is not set or is not a valid directory."
    )

api = wandb.Api()

for run_dir in Path(MODEL_OUT_DIR).iterdir():
    if not run_dir.is_dir():
        continue

    print("Processing directory:", run_dir)

    run_id_file = run_dir / "wandb_run_id.txt"
    if not run_id_file.exists():
        print("    no wandb_run_id.txt, skipping")
        continue

    run_id = run_id_file.read_text().strip()

    try:
        api_run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    except Exception as e:
        print(f"    could not find run {run_id} via API: {e}, skipping")
        continue

    existing_artifacts = [a.name for a in api_run.logged_artifacts()]
    print(f"    Run {run_id} has existing artifacts: {existing_artifacts}")

    for param_file in run_dir.glob("*.pt"):
        artifact_name = f"{run_dir.name}_{param_file.stem}"
        if artifact_name in existing_artifacts:
            continue

        print(
            f"    Uploading {param_file.name} as artifact '{artifact_name}' ...")
        try:
            sdk_run = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                id=run_id,
                resume="allow",
                reinit='finish_previous',
                settings=wandb.Settings(silent=True)
            )
        except Exception as e:
            print(f"    Failed to init/attach SDK run {run_id}: {e}")
            continue

        try:
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(str(param_file))
            sdk_run.log_artifact(artifact)
            print(f"    Uploaded {artifact_name} for run {run_id}")
        except Exception as e:
            print(f"    Failed to upload artifact for run {run_id}: {e}")
        finally:
            try:
                sdk_run.finish()
            except Exception:
                print("    Failed to finish SDK run")
                pass
