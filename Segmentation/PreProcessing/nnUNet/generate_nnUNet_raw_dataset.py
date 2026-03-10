import json
import os
from pathlib import Path

import numpy as np
import torchvision.transforms.functional as F
from PIL import Image

from Environment.env_utils import load_as, load_forksight_env
from Segmentation.PreProcessing.General.preprocessing_util import create_patches_from_img, init_dir

load_forksight_env()

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
HIGHRES_IMG_DIR_NAME = os.getenv("HIGHRES_IMG_DIR_NAME", "images_4096")
HIGHRES_MASK_DIR_NAME = os.getenv("HIGHRES_MASK_DIR_NAME", "masks_4096")
NNUNET_RAW_DIR = os.getenv("NNUNET_RAW_DIR")

NNUNET_DATASET_ID = load_as("NNUNET_DATASET_ID", int, 1)
NNUNET_DATASET_NAME = os.getenv("NNUNET_DATASET_NAME", "Segmentation_v1")
NNUNET_CASE_PREFIX = os.getenv("NNUNET_CASE_PREFIX", "forksight")

if not RAW_DATA_DIR or not NNUNET_RAW_DIR:
    raise ValueError(
        "RAW_DATA_DIR and NNUNET_RAW_DIR environment variables must be set.")
if not Path(RAW_DATA_DIR).exists() or not Path(NNUNET_RAW_DIR).exists():
    raise ValueError(
        f"RAW_DATA_DIR ({RAW_DATA_DIR}) and NNUNET_RAW_DIR ({NNUNET_RAW_DIR}) must be valid directories.")

_SPLITS_FILE = Path(__file__).resolve().parent / "nnUNet_dataset_splits.json"


def load_split(dataset_name: str) -> dict[str, list[str]]:
    with open(_SPLITS_FILE) as f:
        splits = json.load(f)
    split = splits.get(dataset_name, None)
    if split is None:
        split = splits.get("common", None)
    if split is None:
        raise ValueError(
            f"No split found for dataset '{dataset_name}' in {_SPLITS_FILE}")
    return split


def save_cases(
    img_paths: list[Path],
    masks_dir: Path,
    images_out_dir: Path,
    labels_out_dir: Path,
    start_idx: int,
    split: str,
    mapping: list[dict],
) -> int:
    """Process and save image/mask pairs; returns the next available case index.
    Appends one row per saved case to mapping."""
    case_idx = start_idx
    for img_path in img_paths:
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            raise ValueError(
                f"Mask not found for image {img_path.name} at {mask_path}")

        is_soi = "_soi" in img_path.stem

        if is_soi:
            # 1024x1024 SoI image: save as is without patching
            img = Image.open(img_path)
            mask_arr = np.array(Image.open(mask_path))
            mask_label = (mask_arr > 0).astype(np.uint8)

            case_name = f"{NNUNET_CASE_PREFIX}_{case_idx:04d}"
            img.save(images_out_dir / f"{case_name}_0000.png")
            Image.fromarray(mask_label).save(
                labels_out_dir / f"{case_name}.png")
            mapping.append({
                "original_filename": img_path.name,
                "patch": "",
                "nnunet_case": case_name,
                "split": split,
            })

            print(f"Saved SoI image {img_path.name} as case {case_name}")

            case_idx += 1
        else:
            # 4096x4096 image: split into 1024x1024 patches
            img_patches = create_patches_from_img(img_path, patch_size=1024)
            mask_patches = create_patches_from_img(mask_path, patch_size=1024)

            for patch_idx, (img_patch, mask_patch) in enumerate(
                zip(img_patches, mask_patches)
            ):
                case_name = f"{NNUNET_CASE_PREFIX}_{case_idx:04d}"

                F.to_pil_image(img_patch).save(
                    images_out_dir / f"{case_name}_0000.png"
                )

                # threshold at 0.5 because create_patches_from_img normalizes to floats [0,1]
                mask_label = (mask_patch.squeeze(
                    0).numpy() > 0.5).astype(np.uint8)
                Image.fromarray(mask_label).save(
                    labels_out_dir / f"{case_name}.png")

                mapping.append({
                    "original_filename": img_path.name,
                    "patch": patch_idx,
                    "nnunet_case": case_name,
                    "split": split,
                })

                print(
                    f"Saved patch {patch_idx} of image {img_path.name} as case {case_name}")

                case_idx += 1

    return case_idx


def generate_dataset():
    raw_data_dir = Path(RAW_DATA_DIR)
    images_dir = raw_data_dir / HIGHRES_IMG_DIR_NAME
    masks_dir = raw_data_dir / HIGHRES_MASK_DIR_NAME

    split = load_split(NNUNET_DATASET_NAME)
    train_names = set(split["train"])
    test_names = set(split["test"])

    train_paths = sorted(p for p in images_dir.glob(
        "*.png") if p.name in train_names)
    test_paths = sorted(p for p in images_dir.glob(
        "*.png") if p.name in test_names)

    dataset_dir_name = f"Dataset{NNUNET_DATASET_ID:03d}_{NNUNET_DATASET_NAME}"
    dataset_dir = Path(NNUNET_RAW_DIR) / dataset_dir_name

    images_tr_dir = dataset_dir / "imagesTr"
    labels_tr_dir = dataset_dir / "labelsTr"
    images_ts_dir = dataset_dir / "imagesTs"
    labels_ts_dir = dataset_dir / "labelsTs"

    init_dir(dataset_dir)
    for d in [images_tr_dir, labels_tr_dir, images_ts_dir, labels_ts_dir]:
        d.mkdir(parents=True)

    mapping: list[dict] = []

    print(f"Processing {len(train_paths)} training images")
    case_idx = save_cases(
        train_paths, masks_dir, images_tr_dir, labels_tr_dir,
        start_idx=0, split="train", mapping=mapping,
    )
    num_training = case_idx

    print(f"\nProcessing {len(test_paths)} test images")
    case_idx = save_cases(
        test_paths, masks_dir, images_ts_dir, labels_ts_dir,
        start_idx=case_idx, split="test", mapping=mapping,
    )
    num_test = case_idx - num_training

    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {"background": 0, "DNA": 1},
        "numTraining": num_training,
        "file_ending": ".png",
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    mapping_path = dataset_dir / "case_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=4)

    print(
        f"\nDataset '{dataset_dir_name}' created with {num_training} training cases "
        f"and {num_test} test cases."
    )
    print(f"dataset.json written to {dataset_dir / 'dataset.json'}")
    print(f"case_mapping.json written to {mapping_path}")


if __name__ == "__main__":
    generate_dataset()
