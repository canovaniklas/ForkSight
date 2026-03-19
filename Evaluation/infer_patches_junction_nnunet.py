"""Save nnUNet segmentation patch predictions for junction detection evaluation.

Run this script in the nnUNet conda environment.  For each configured nnUNet
model and each test image, it downloads the model artifact from WandB, splits
the full images into patches, and calls predict_from_files() with the patch
PNGs as inputs.  nnUNet writes class-label PNGs (0/1); these are converted
in-place to 0/255 to match the SAM inference output format.

Output layout under JUNCTION_PRED_DIR:
  <JUNCTION_PRED_DIR>/
    <safe_model_key>/
      metadata.json          {"model_key": str, "dataset": str}
      <image_stem>_patch_00.png
      ...
      <image_stem>_patch_15.png

Required environment variables (loaded via load_forksight_env):
  JUNCTION_DETECTION_DATASET_DIR   root of the junction detection dataset
  JUNCTION_PRED_DIR                output directory for predictions
  WANDB_ENTITY                     WandB entity
  WANDB_NNUNET_PROJECT             WandB nnUNet project name

Usage:
  python infer_patches_junction_nnunet.py [--device N]
                                          [--is-test] [--force-rerun]
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import wandb

import Environment.env_utils as env_utils
from Segmentation.PreProcessing.General.preprocessing_util import (
    create_patches_from_img,
)
from Segmentation.Util.nnunet_wandb_util import (
    download_nnunet_artifact,
    initialize_nnunet_predictor,
    nnunet_model_key,
    NNUNET_DEFAULT_FOLDS,
    NNUNET_DEFAULT_CHECKPOINT,
)
from Evaluation.pipeline_evaluation_shared import PATCH_SIZE
from Evaluation.compute_metrics_config import NNUNET_EVALUATIONS


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--is-test", action="store_true",
                        help="Only infer the first test image per model")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Re-run inference even if output already exists")
    args = parser.parse_args()

    env_utils.load_forksight_env()

    JUNCTION_DETECTION_DATASET_DIR = os.getenv(
        "JUNCTION_DETECTION_DATASET_DIR")
    JUNCTION_PRED_DIR = os.getenv("JUNCTION_PRED_DIR")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
    WANDB_NNUNET_PROJECT = os.getenv(
        "WANDB_NNUNET_PROJECT", "ForkSight-nnUNet")

    if JUNCTION_DETECTION_DATASET_DIR is None:
        raise ValueError(
            "JUNCTION_DETECTION_DATASET_DIR environment variable must be set.")
    if JUNCTION_PRED_DIR is None:
        raise ValueError("JUNCTION_PRED_DIR environment variable must be set.")

    test_images_dir = (Path(JUNCTION_DETECTION_DATASET_DIR) / "images")
    if not test_images_dir.is_dir():
        raise FileNotFoundError(
            f"Test images directory not found: {test_images_dir}")

    test_image_paths = sorted(p for p in test_images_dir.glob("*.png"))
    if not test_image_paths:
        raise FileNotFoundError(f"No PNG files found in {test_images_dir}")

    if args.is_test:
        test_image_paths = test_image_paths[:1]

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Processing {len(test_image_paths)} test image(s).")

    pred_base = Path(JUNCTION_PRED_DIR)
    pred_base.mkdir(parents=True, exist_ok=True)

    if not NNUNET_EVALUATIONS:
        print("NNUNET_EVALUATIONS is empty — nothing to run.")
        return

    api = wandb.Api()

    with tempfile.TemporaryDirectory(prefix="nnunet_jd_") as _tmproot:
        tmp_root = Path(_tmproot)

        for dataset, trainer in NNUNET_EVALUATIONS:
            model_key = nnunet_model_key(dataset, trainer)
            safe_name = model_key.replace("/", "_")
            model_pred_dir = pred_base / safe_name
            metadata_path = model_pred_dir / "metadata.json"

            if metadata_path.is_file() and not args.force_rerun:
                print(f"\n  Already inferred: {model_key} — skipping. "
                      f"(use --force-rerun to override)")
                continue

            print(f"\n{'='*60}")
            print(f"Inferring nnUNet: {model_key}")
            print(f"{'='*60}")

            print(f"  Downloading artifact from {WANDB_NNUNET_PROJECT}")
            model_dir = download_nnunet_artifact(
                api, WANDB_ENTITY, WANDB_NNUNET_PROJECT,
                dataset, trainer, tmp_root,
            )
            print(f"  Model dir: {model_dir}")

            predictor = initialize_nnunet_predictor(
                model_dir, device,
                folds=NNUNET_DEFAULT_FOLDS,
                checkpoint=NNUNET_DEFAULT_CHECKPOINT,
            )

            model_pred_dir.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps({
                "model_key": model_key,
                "dataset": dataset,
            }))

            # Save all input patches to a temp dir to run inference from files
            inp_dir = tmp_root / f"inp_{safe_name}"
            inp_dir.mkdir(exist_ok=True)

            input_file_lists: list[list[str]] = []
            for img_path in test_image_paths:
                patches = create_patches_from_img(
                    img_path, patch_size=PATCH_SIZE[0])
                for idx in range(patches.shape[0]):
                    fname = f"{img_path.stem}_patch_{idx:02d}_0000.png"
                    TF.to_pil_image(patches[idx]).save(inp_dir / fname)
                    input_file_lists.append([str(inp_dir / fname)])

            print(
                f"  Running predict_from_files on {len(input_file_lists)} patches")
            predictor.predict_from_files(
                input_file_lists,
                str(model_pred_dir),
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
            )

            # nnUNet saves class-label PNGs (0/1); convert to 0/255 to match SAM output
            for png in model_pred_dir.glob("*.png"):
                arr = np.array(Image.open(png))
                Image.fromarray((arr * 255).astype(np.uint8)).save(png)

            del predictor
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
