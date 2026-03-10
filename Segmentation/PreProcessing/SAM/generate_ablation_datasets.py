import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from Environment.env_utils import load_as, load_as_tuple, load_forksight_env
from Segmentation.PreProcessing.SAM.sam_preprocessing_util import (
    set_seeds, load_png_as_tensor, init_dir,
    save_tensor_as_png, save_heatmap,
    AUG_TYPES, apply_augmentation, get_train_val_test_split_paths,
    create_patches_and_save, oversample_junction_patches, remove_highres_dirs,
)

load_forksight_env()

SEED = load_as("SEED", int, 42)

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
DATASETS_DIR = os.getenv("DATASETS_DIR")

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
DATASET_VAL_SPLIT = load_as("DATASET_VAL_SPLIT", float, "0.2")

if not RAW_DATA_DIR or not DATASETS_DIR:
    raise ValueError(
        "RAW_DATA_DIR and DATASETS_DIR environment variables must be set.")

ABLATION_DATASET_BASE_NAME = "Ablation_Dataset"
SHARED_SPLIT_NAME = f"{ABLATION_DATASET_BASE_NAME}_shared"

# ABLATION_DATASETS: dataset configurations, as percentages of raw/augmented images relative to the number of raw training images
#                    additionally, generates two full datasets with all raw images + all augmentations, with and without junction oversampling
# Examples:
#   (1.0, 0.0): all training images, no augmentation
#   (1.0, 1.0): all training images + equal number of augmented images
#   (0.5, 0.5): 50% raw, 50% augmented (augmented only from the 50% raw)
# The validation and test sets are always kept complete and unaugmented
ABLATION_DATASETS: list[tuple[float, float]] = [
    # no scaling, keep dataset size = raw train size
    (1.0, 0.0),                 # raw only
    (0.75, 0.25),               # 75% raw + 25% augmented
    (0.5, 0.5),                 # 50% raw + 50% augmented
    (0.25, 0.75),               # 25% raw + 75% augmented

    # scaling: increase dataset size over original raw train size
    (1.0, 1.0),                 # 100% raw + 100% augmented (2x size)
    (1.0, 2.0),                 # 100% raw + 200% augmented (3x size)
    (1.0, 4.0),                 # 100% raw + 400% augmented (5x size)
]


def _dataset_name(pct_raw: float, pct_aug: float) -> str:
    raw_pct = int(round(pct_raw * 100))
    aug_pct = int(round(pct_aug * 100))
    return f"{ABLATION_DATASET_BASE_NAME}_Raw{raw_pct}_Aug{aug_pct}"


def generate_shared_splits() -> Path:
    """Generate val and test sets once, shared across all ablation datasets"""
    raw_images_dir = Path(RAW_DATA_DIR) / HIGHRES_IMG_DIR_NAME
    raw_masks_dir = Path(RAW_DATA_DIR) / HIGHRES_MASK_DIR_NAME

    shared_dir = Path(DATASETS_DIR) / SHARED_SPLIT_NAME
    init_dir(shared_dir)

    _, val_paths, test_paths = get_train_val_test_split_paths(
        raw_images_dir, ABLATION_DATASET_BASE_NAME, DATASET_VAL_SPLIT, SEED
    )

    print(f"\n{'=' * 60}")
    print(f"Generating shared splits: {SHARED_SPLIT_NAME}")
    print(f"  val images:  {len(val_paths)}")
    print(f"  test images: {len(test_paths)}")
    print(f"{'=' * 60}\n")

    for split_dir, paths in [
        (shared_dir / "validation", val_paths),
        (shared_dir / "test", test_paths),
    ]:
        (split_dir / HIGHRES_IMG_DIR_NAME).mkdir(parents=True, exist_ok=True)
        (split_dir / HIGHRES_MASK_DIR_NAME).mkdir(parents=True, exist_ok=True)
        for png_path in paths:
            img_tensor = load_png_as_tensor(png_path)
            mask_tensor = load_png_as_tensor(raw_masks_dir / png_path.name)
            save_tensor_as_png(img_tensor, mask_tensor, png_path,
                               split_dir / HIGHRES_IMG_DIR_NAME,
                               split_dir / HIGHRES_MASK_DIR_NAME,
                               None)

    return shared_dir


def generate_ablation_dataset(pct_raw: float, pct_aug: float, shared_dir: Path, name_override: str | None = None):
    assert 0.0 <= pct_raw <= 1.0, "pct_raw must be in [0, 1]"
    assert pct_aug >= 0.0, "pct_aug must be >= 0"

    ablation_name = name_override or _dataset_name(pct_raw, pct_aug)
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

    init_dir(dataset_dir)
    (train_dir / HIGHRES_IMG_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (train_dir / HIGHRES_MASK_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (train_dir / HIGHRES_HEATMAP_DIR_NAME).mkdir(parents=True, exist_ok=True)

    # Symlink validation and test into the shared splits folder
    for split in ["validation", "test"]:
        link = dataset_dir / split
        target = shared_dir / split
        link.symlink_to(target, target_is_directory=True)

    all_train_paths, _, _ = get_train_val_test_split_paths(
        raw_images_dir, ABLATION_DATASET_BASE_NAME, DATASET_VAL_SPLIT, SEED
    )

    # Sample raw images for train
    rng = random.Random(SEED)
    n_train_raw = int(round(pct_raw * len(all_train_paths)))
    sampled_train_paths = rng.sample(
        all_train_paths, n_train_raw) if n_train_raw > 0 else []

    # a) pct_aug < pct_raw : randomly select a subset of sampled raw images, each gets 1 random aug
    # b) pct_aug == pct_raw: each sampled raw image gets exactly 1 random aug
    # c) pct_aug > pct_raw : N = pct_aug/pct_raw (integer), each sampled raw image gets N different augmentations
    if pct_aug == 0:
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

    return dataset_dir


def generate_ablation_datasets():
    set_seeds(SEED)

    # Generate val and test once into a shared folder
    shared_dir = generate_shared_splits()
    create_patches_and_save(
        shared_dir,
        HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, HIGHRES_HEATMAP_DIR_NAME,
        HIGHRES_IMG_PATCHES_DIR_NAME, HIGHRES_MASK_PATCHES_DIR_NAME, HIGHRES_HEATMAP_PATCHES_DIR_NAME,
        splits=["validation", "test"],
    )
    remove_highres_dirs(
        shared_dir,
        HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, HIGHRES_HEATMAP_DIR_NAME,
        splits=["validation", "test"],
    )

    for pct_raw, pct_aug in ABLATION_DATASETS:
        dataset_dir = generate_ablation_dataset(pct_raw, pct_aug, shared_dir)
        # Only process train split; val/test are symlinks into the shared folder.
        # No junction oversampling for percentage-based ablation datasets.
        create_patches_and_save(
            dataset_dir,
            HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, HIGHRES_HEATMAP_DIR_NAME,
            HIGHRES_IMG_PATCHES_DIR_NAME, HIGHRES_MASK_PATCHES_DIR_NAME, HIGHRES_HEATMAP_PATCHES_DIR_NAME,
            splits=["train"],
        )
        remove_highres_dirs(
            dataset_dir,
            HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, HIGHRES_HEATMAP_DIR_NAME,
            splits=["train"],
        )

    # Full datasets: all raw images + all augmentation types (N = len(AUG_TYPES) per image)
    for name, oversample_count in [
        (f"{ABLATION_DATASET_BASE_NAME}_Full", 0),
        (f"{ABLATION_DATASET_BASE_NAME}_Full_JunctionsOversampled", 1),
    ]:
        dataset_dir = generate_ablation_dataset(
            pct_raw=1.0, pct_aug=float(len(AUG_TYPES)), shared_dir=shared_dir, name_override=name,
        )
        create_patches_and_save(
            dataset_dir,
            HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, HIGHRES_HEATMAP_DIR_NAME,
            HIGHRES_IMG_PATCHES_DIR_NAME, HIGHRES_MASK_PATCHES_DIR_NAME, HIGHRES_HEATMAP_PATCHES_DIR_NAME,
            splits=["train"],
        )
        if oversample_count > 0:
            oversample_junction_patches(
                dataset_dir,
                HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, HIGHRES_HEATMAP_DIR_NAME,
                HIGHRES_IMG_PATCHES_DIR_NAME, HIGHRES_MASK_PATCHES_DIR_NAME, HIGHRES_HEATMAP_PATCHES_DIR_NAME,
                oversample_count,
            )
        remove_highres_dirs(
            dataset_dir,
            HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, HIGHRES_HEATMAP_DIR_NAME,
            splits=["train"],
        )


def fix_75_25_dataset():
    """With the current implementation and seed, somehow the raw 75%, aug 25% dataset has 81 instead of 66 images, 
    so remove 15 random aug images to get the 75/25 ratio and 264 total patches
    """
    print("\nFixing the 75% raw + 25% aug dataset to have exactly 75% raw and 25% augmented patches")

    dataset_dir = Path(DATASETS_DIR) / _dataset_name(0.75, 0.25)
    train_img_patches_dir = dataset_dir / "train" / HIGHRES_IMG_PATCHES_DIR_NAME
    train_mask_patches_dir = dataset_dir / "train" / HIGHRES_MASK_PATCHES_DIR_NAME
    train_hm_patches_dir = dataset_dir / "train" / HIGHRES_HEATMAP_PATCHES_DIR_NAME

    all_aug_patches = sorted([p for p in train_img_patches_dir.glob(
        '*.png') if any(aug in p.name for aug in AUG_TYPES)])
    print(
        f"Found {len(all_aug_patches)} augmented patches in the 75% raw + 25% aug dataset")

    rng = random.Random(SEED)
    patches_to_remove = rng.sample(all_aug_patches, len(all_aug_patches) - 66)
    print(
        f"Removing {len(patches_to_remove)} randomly selected augmented patches to fix the dataset size to 66")

    for patch_path in patches_to_remove:
        mask_patch_path = train_mask_patches_dir / patch_path.name
        hm_patch_path = train_hm_patches_dir / \
            patch_path.with_suffix('.npy').name

        patch_path.unlink()
        mask_patch_path.unlink()
        hm_patch_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix_75_25", type=int, default=0,
                        help="Generate datasets (0), fix the 75% raw + 25% aug dataset (1), do both (2)")
    args = parser.parse_args()

    if args.fix_75_25 == 0:
        generate_ablation_datasets()
    elif args.fix_75_25 == 1:
        fix_75_25_dataset()
    elif args.fix_75_25 == 2:
        generate_ablation_datasets()
        fix_75_25_dataset()
    else:
        raise ValueError("Invalid value for --fix_75_25. Must be in [0, 1, 2]")


if __name__ == "__main__":
    main()
