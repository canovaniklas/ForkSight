import os
import random
from pathlib import Path

import numpy as np
import torch

from Segmentation.Util.env_utils import load_as, load_as_tuple, load_segmentation_env
from Segmentation.PreProcessing.preprocessing_utils import (
    set_seeds, load_png_as_tensor, init_dir,
    save_tensor_as_png, save_heatmap,
    AUG_TYPES, apply_augmentation, get_train_val_test_split_paths,
    create_patches_and_save, oversample_junction_patches, remove_highres_dirs,
)

load_segmentation_env()

SEED = load_as("SEED", int, 42)

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
DATASETS_DIR = os.getenv("DATASETS_DIR")
DATASET_NAME = os.getenv("DATASET_NAME", "Segmentation_v1")

HIGHRES_IMG_DIR_NAME = os.getenv("HIGHRES_IMG_DIR_NAME", "images_4096")
HIGHRES_MASK_DIR_NAME = os.getenv("HIGHRES_MASK_DIR_NAME", "masks_4096")
HIGHRES_HEATMAP_DIR_NAME = os.getenv(
    "HIGHRES_HEATMAP_DIR_NAME", "heatmaps_4096")

HIGHRES_IMG_PATCHES_DIR_NAME = os.getenv(
    "HIGHRES_IMG_PATCHES_DIR_NAME", "img_patches_1024")
HIGHRES_MASK_PATCHES_DIR_NAME = os.getenv(
    "HIGHRES_MASK_PATCHES_DIR_NAME", "mask_patches_1024")
HIGHRES_HEATMAP_PATCHES_DIR_NAME = os.getenv(
    "HIGHRES_HEATMAP_PATCHES_DIR_NAME", "heatmap_patches_1024")

DATASET_GAMMA_RANGE = load_as_tuple("DATASET_GAMMA_RANGE", "0.6,1.4", float)
DATASET_MAX_DISTORT = load_as("DATASET_MAX_DISTORT", float, "0.1")
DATASET_DISTORT_GRID_SIZE = load_as_tuple(
    "DATASET_DISTORT_GRID_SIZE", "4,4", int)
DATASET_OVERSAMPLE_JUNCTION_PATCHES = load_as(
    "DATASET_OVERSAMPLE_JUNCTION_PATCHES", int, 0)
DATASET_VAL_SPLIT = load_as("DATASET_VAL_SPLIT", float, "0.2")

if not RAW_DATA_DIR or not DATASETS_DIR:
    raise ValueError(
        "RAW_DATA_DIR and DATASETS_DIR environment variables must be set.")

# ABLATION_DATASETS: dataset configurations, as percentages of raw/augmented images relative to the number of raw training images
# Examples:
#   (1.0, 0.0): all training images, no augmentation
#   (1.0, 1.0): all training images + equal number of augmented images
#   (0.5, 0.5): 50% raw, 50% augmented (augmented only from the 50% raw)
#   (0.0, 1.0): no raw images, only augmented (sources randomly sampled but not included)
#
# The validation and test sets are always kept complete and unaugmented
ABLATION_DATASETS: list[tuple[float, float]] = [
    # no scaling, keep dataset size = raw train size
    (1.0, 0.0),                 # raw only
    (0.75, 0.25),               # 75% raw + 25% augmented
    (0.5, 0.5),                 # 50% raw + 50% augmented
    (0.25, 0.75),               # 25% raw + 75% augmented

    # scaling: increase dataset size over original raw train size
    (1.0, 0.5),                 # 100% raw + 50% augmented (1.5x size)
    (1.0, 1.0),                 # 100% raw + 100% augmented (2x size)
    (1.0, 2.0),                 # 100% raw + 200% augmented (3x size)
    (1.0, 4.0),                 # 100% raw + 400% augmented (5x size)
]


def _dataset_name(pct_raw: float, pct_aug: float) -> str:
    raw_pct = int(round(pct_raw * 100))
    aug_pct = int(round(pct_aug * 100))
    return f"{DATASET_NAME}_ablation_raw{raw_pct}_aug{aug_pct}"


def generate_ablation_dataset(pct_raw: float, pct_aug: float):
    assert 0.0 <= pct_raw <= 1.0, "pct_raw must be in [0, 1]"
    assert pct_aug >= 0.0, "pct_aug must be >= 0"

    ablation_name = _dataset_name(pct_raw, pct_aug)
    print(f"\n{'=' * 60}")
    print(f"Generating: {ablation_name}")
    print(f"  pct_raw={pct_raw}, pct_aug={pct_aug}")
    print(f"{'=' * 60}\n")

    raw_data_dir = Path(RAW_DATA_DIR)
    raw_images_dir = raw_data_dir / HIGHRES_IMG_DIR_NAME
    raw_masks_dir = raw_data_dir / HIGHRES_MASK_DIR_NAME
    raw_heatmaps_dir = raw_data_dir / HIGHRES_HEATMAP_DIR_NAME

    dataset_dir = Path(DATASETS_DIR) / ablation_name
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"
    test_dir = dataset_dir / "test"

    init_dir(dataset_dir)
    for split_dir in [train_dir, val_dir, test_dir]:
        (split_dir / HIGHRES_IMG_DIR_NAME).mkdir(parents=True, exist_ok=True)
        (split_dir / HIGHRES_MASK_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (train_dir / HIGHRES_HEATMAP_DIR_NAME).mkdir(parents=True, exist_ok=True)

    all_train_paths, val_paths, test_paths = get_train_val_test_split_paths(
        raw_images_dir, DATASET_NAME, DATASET_VAL_SPLIT, SEED
    )

    # Sample raw images for train
    rng = random.Random(SEED)
    n_train_raw = int(round(pct_raw * len(all_train_paths)))
    sampled_train_paths = rng.sample(
        all_train_paths, n_train_raw) if n_train_raw > 0 else []

    # a) pct_aug < pct_raw : randomly select a subset of sampled raw images, each gets 1 random aug
    # b) pct_aug == pct_raw: each sampled raw image gets exactly 1 random aug
    # c) pct_aug > pct_raw : N = pct_aug/pct_raw (integer), each sampled raw image gets N different augmentations
    # d) pct_raw == 0      : sample sources randomly from all train images (not included in dataset)
    if pct_raw == 0:
        # Case d)
        n_aug = int(round(pct_aug * len(all_train_paths)))
        all_pairs = [(p, a) for p in all_train_paths for a in AUG_TYPES]
        aug_assignments = rng.sample(all_pairs, min(n_aug, len(all_pairs)))
    elif pct_aug == 0:
        aug_assignments = []
    elif pct_aug <= pct_raw:
        # Cases a), b): each source gets exactly one aug; sources sampled without replacement
        n_aug = int(round(pct_aug * len(all_train_paths)))
        sources_for_aug = rng.sample(sampled_train_paths, n_aug)
        aug_assignments = [(p, rng.choice(AUG_TYPES)) for p in sources_for_aug]
    else:
        # Case c): N different augs per raw image, no repeated aug type per image
        N = round(pct_aug / pct_raw)
        assert N <= len(AUG_TYPES), (
            f"N={N} exceeds the number of available aug types ({len(AUG_TYPES)}). "
            f"pct_aug/pct_raw must be ≤ {len(AUG_TYPES)}."
        )
        aug_assignments = []
        for path in sampled_train_paths:
            for aug_type in rng.sample(AUG_TYPES, N):
                aug_assignments.append((path, aug_type))

    print(f"Train raw images:       {len(sampled_train_paths)}")
    print(f"Train augmented images: {len(aug_assignments)}")
    print(f"Val raw images:         {len(val_paths)}")
    print(f"Test images:            {len(test_paths)}")
    print()

    # Save raw train images
    for png_path in sampled_train_paths:
        img_tensor = load_png_as_tensor(png_path)
        mask_tensor = load_png_as_tensor(raw_masks_dir / png_path.name)
        heatmap_tensor = torch.from_numpy(
            np.load(raw_heatmaps_dir / f"{png_path.stem}.npy")
        ).unsqueeze(0)
        save_tensor_as_png(img_tensor, mask_tensor, png_path,
                           train_dir / HIGHRES_IMG_DIR_NAME,
                           train_dir / HIGHRES_MASK_DIR_NAME,
                           None)
        save_heatmap(heatmap_tensor, png_path, train_dir /
                     HIGHRES_HEATMAP_DIR_NAME, None)

    # Generate and save augmented train images
    for png_path, aug_type in aug_assignments:
        img_tensor = load_png_as_tensor(png_path)
        mask_tensor = load_png_as_tensor(raw_masks_dir / png_path.name)
        heatmap_tensor = torch.from_numpy(
            np.load(raw_heatmaps_dir / f"{png_path.stem}.npy")
        ).unsqueeze(0)

        img_aug, mask_aug, hm_aug = apply_augmentation(
            img_tensor, mask_tensor, heatmap_tensor,
            aug_type, DATASET_GAMMA_RANGE, DATASET_MAX_DISTORT, DATASET_DISTORT_GRID_SIZE,
        )

        save_tensor_as_png(img_aug, mask_aug, png_path,
                           train_dir / HIGHRES_IMG_DIR_NAME,
                           train_dir / HIGHRES_MASK_DIR_NAME,
                           aug_type)
        save_heatmap(hm_aug, png_path, train_dir /
                     HIGHRES_HEATMAP_DIR_NAME, aug_type)

    # Save val and test images (raw only, no augmentation)
    for base_dir, paths in [(val_dir, val_paths), (test_dir, test_paths)]:
        for png_path in paths:
            img_tensor = load_png_as_tensor(png_path)
            mask_tensor = load_png_as_tensor(raw_masks_dir / png_path.name)
            save_tensor_as_png(img_tensor, mask_tensor, png_path,
                               base_dir / HIGHRES_IMG_DIR_NAME,
                               base_dir / HIGHRES_MASK_DIR_NAME,
                               None)

    return dataset_dir


def main():
    set_seeds(SEED)

    for pct_raw, pct_aug in ABLATION_DATASETS:
        dataset_dir = generate_ablation_dataset(pct_raw, pct_aug)
        create_patches_and_save(
            dataset_dir,
            HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, HIGHRES_HEATMAP_DIR_NAME,
            HIGHRES_IMG_PATCHES_DIR_NAME, HIGHRES_MASK_PATCHES_DIR_NAME, HIGHRES_HEATMAP_PATCHES_DIR_NAME,
        )
        if DATASET_OVERSAMPLE_JUNCTION_PATCHES > 0:
            oversample_junction_patches(
                dataset_dir,
                HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, HIGHRES_HEATMAP_DIR_NAME,
                HIGHRES_IMG_PATCHES_DIR_NAME, HIGHRES_MASK_PATCHES_DIR_NAME, HIGHRES_HEATMAP_PATCHES_DIR_NAME,
                DATASET_OVERSAMPLE_JUNCTION_PATCHES,
            )
        remove_highres_dirs(
            dataset_dir,
            HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, HIGHRES_HEATMAP_DIR_NAME,
        )


if __name__ == "__main__":
    main()
