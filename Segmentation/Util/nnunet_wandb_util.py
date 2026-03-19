"""Utilities for loading nnUNet models from WandB artifacts and running
patch-level inference on 2D images.

nnUNetPredictor.initialize_from_trained_model_folder needs a directory with:
    plans.json
    dataset.json
    fold_*/checkpoint_*.pth
these files are all included in the artifact uploaded by upload_nnunet_artifact_wandb.py

Naming conventions:
    nnUNet folder:   {trainer}__nnUNetPlans__2d
    WandB artifact:  nnunet-{dataset}-{trainer}
    metrics CSV key: nnunet/{dataset}/{trainer}
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import wandb
import shutil
import tempfile
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from Segmentation.PreProcessing.General.preprocessing_util import create_patches_from_img

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


NNUNET_CONFIGURATION = "2d"
NNUNET_PLANS = "nnUNetPlans"
NNUNET_DEFAULT_FOLDS = (0, 1, 2, 3, 4)
NNUNET_DEFAULT_CHECKPOINT = "checkpoint_final.pth"


def nnunet_folder_name(trainer: str) -> str:
    return f"{trainer}__{NNUNET_PLANS}__{NNUNET_CONFIGURATION}"


def nnunet_artifact_name(dataset: str, trainer: str) -> str:
    return f"nnunet-{dataset}-{trainer}"


def nnunet_model_key(dataset: str, trainer: str) -> str:
    return f"nnunet/{dataset}/{trainer}"


def download_nnunet_artifact(
    api: wandb.Api,
    entity: str,
    project: str,
    dataset: str,
    trainer: str,
    target_dir: Path,
) -> Path:
    """Download the nnUNet WandB artifact for (dataset, trainer) to target_dir.

    Returns the path to the local artifact directory, which contains plans.json,
    dataset.json, and fold_*/checkpoint_final.pth as uploaded by
    upload_nnunet_artifact_wandb.py.
    """
    name = nnunet_artifact_name(dataset, trainer)
    artifact = api.artifact(f"{entity}/{project}/{name}:latest", type="model")
    artifact_dir = Path(artifact.download(root=str(target_dir / name)))
    return artifact_dir


def initialize_nnunet_predictor(
    model_dir: Path,
    device: torch.device,
    folds: Sequence[int] = NNUNET_DEFAULT_FOLDS,
    checkpoint: str = NNUNET_DEFAULT_CHECKPOINT,
) -> "nnUNetPredictor":
    """Initialize an nnUNetPredictor from a local model directory.

    model_dir must contain plans.json, dataset.json, and fold_*/checkpoint_*.pth,
    as produced by download_nnunet_artifact().
    """
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=tuple(folds),
        checkpoint_name=checkpoint,
    )
    return predictor
