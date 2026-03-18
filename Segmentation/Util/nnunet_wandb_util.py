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

Inference approach:
    predict_from_list_of_npy_arrays() is used for fully in-memory inference.
    predict_single_npy_array() is documented as SLOW and only processes one image at a
    time; predict_from_files() is the recommended batch mode but requires writing patches
    to disk and reading them back.

    predict_from_list_of_npy_arrays() avoids all disk I/O by accepting numpy arrays
    directly.  The arrays must be in the format produced by the I/O class recorded in
    plans.json under "image_reader_writer".  predict_image_patches_nnunet() assumes
    NaturalImage2DIO (the standard reader for 2D PNG datasets) and verifies this at
    runtime against predictor.plans_manager.image_reader_writer_class.

    NaturalImage2DIO format: (C, H, W) float32, raw pixel values (0-255 for uint8 images),
    properties = {'spacing': (999, 1, 1)}.

    All B = rows*cols patches are submitted in a single call, which is nnUNet's batch
    mode and lets nnUNet handle its own internal preprocessing pipeline.
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


def predict_image_patches_nnunet(
    predictor: nnUNetPredictor,
    image_path: Path,
    patch_size: int = 1024,
) -> torch.Tensor:
    """Split a full-resolution image into patches and run nnUNet inference.

    Uses create_patches_from_img() for patch extraction, saves patches to a
    temporary directory, and calls predict_from_files() which resolves the
    correct I/O class from plans.json automatically.

    Returns (B, 1, H, W) binary float32 mask tensor on CPU, in the same
    row-major patch order as create_patches_from_img().
    """
    patches = create_patches_from_img(image_path, patch_size=patch_size)
    B = patches.shape[0]

    tmp_dir = Path(tempfile.mkdtemp(prefix="nnunet_patches_"))
    inp_dir = tmp_dir / "inp"
    out_dir = tmp_dir / "out"
    inp_dir.mkdir()
    out_dir.mkdir()

    for idx in range(B):
        TF.to_pil_image(patches[idx]).save(
            inp_dir / f"patch_{idx:04d}_0000.png")

    predictor.predict_from_files(
        [[str(inp_dir / f"patch_{idx:04d}_0000.png")] for idx in range(B)],
        str(out_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )

    masks = []
    for idx in range(B):
        seg = np.array(PILImage.open(
            out_dir / f"patch_{idx:04d}.png"))
        if seg.ndim == 3:
            seg = seg[..., 0]
        masks.append(torch.from_numpy(
            (seg > 0).astype(np.float32)).unsqueeze(0))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    # (B, 1, H, W)
    return torch.stack(masks)
