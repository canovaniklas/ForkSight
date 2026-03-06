import os
from pathlib import Path

import numpy as np
import torch

from Segmentation.Util.env_utils import load_as, load_as_tuple, load_segmentation_env
from Segmentation.PreProcessing.preprocessing_utils import (
    set_seeds, load_png_as_tensor, init_dir,
    save_tensor_as_png, save_heatmap, visualize_heatmap,
    get_all_augmentations, get_train_val_test_split_paths,
    create_patches_and_save,
    oversample_junction_patches, remove_highres_dirs,
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
HEATMAP_VISUALIZATION_DIR_NAME = os.getenv(
    "HEATMAP_VISUALIZATION_DIR_NAME", "heatmap_visualizations")

DATASET_GAMMA_RANGE = load_as_tuple("DATASET_GAMMA_RANGE", "0.6,1.4", float)
DATASET_MAX_DISTORT = load_as("DATASET_MAX_DISTORT", float, "0.1")
DATASET_DISTORT_GRID_SIZE = load_as_tuple(
    "DATASET_DISTORT_GRID_SIZE", "4,4", int)
DATASET_OVERSAMPLE_JUNCTION_PATCHES = load_as(
    "DATASET_OVERSAMPLE_JUNCTION_PATCHES", int, 0)
DATASET_VAL_SPLIT = load_as("DATASET_VAL_SPLIT", float, "0.2")
DATASET_SAVE_HEATMAP_VISUALIZATIONS = load_as(
    "DATASET_SAVE_HEATMAP_VISUALIZATIONS", bool, False)

if not RAW_DATA_DIR or not DATASETS_DIR:
    raise ValueError(
        "RAW_DATA_DIR and DATASETS_DIR environment variables must be set.")


def augment_and_save():
    raw_data_dir = Path(RAW_DATA_DIR)
    raw_images_dir = raw_data_dir / HIGHRES_IMG_DIR_NAME
    raw_masks_dir = raw_data_dir / HIGHRES_MASK_DIR_NAME
    raw_heatmaps_dir = raw_data_dir / HIGHRES_HEATMAP_DIR_NAME

    dataset_dir = Path(DATASETS_DIR) / DATASET_NAME
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"
    test_dir = dataset_dir / "test"

    init_dir(dataset_dir)
    for split_dir in [train_dir, val_dir, test_dir]:
        (split_dir / HIGHRES_IMG_DIR_NAME).mkdir(parents=True, exist_ok=True)
        (split_dir / HIGHRES_MASK_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (train_dir / HIGHRES_HEATMAP_DIR_NAME).mkdir(parents=True, exist_ok=True)

    train_image_paths, val_image_paths, test_image_paths = get_train_val_test_split_paths(
        raw_images_dir, DATASET_NAME, DATASET_VAL_SPLIT, SEED)

    if DATASET_SAVE_HEATMAP_VISUALIZATIONS:
        viz_dir = dataset_dir / HEATMAP_VISUALIZATION_DIR_NAME
        viz_dir.mkdir(parents=True, exist_ok=True)

    print("training images: ", str([Path(p).name for p in train_image_paths]))
    print("validation images: ", str([Path(p).name for p in val_image_paths]))
    print("testing images: ", str([Path(p).name for p in test_image_paths]))

    for paths, split_dir, is_train in [
        (train_image_paths, train_dir, True),
        (val_image_paths, val_dir, False),
        (test_image_paths, test_dir, False),
    ]:
        for png_path in paths:
            img_tensor = load_png_as_tensor(png_path)
            mask_tensor = load_png_as_tensor(raw_masks_dir / png_path.name)
            heatmap_tensor = None

            if is_train:
                heatmap_npy_path = raw_heatmaps_dir / f"{png_path.stem}.npy"
                heatmap_tensor = torch.from_numpy(
                    np.load(heatmap_npy_path)).unsqueeze(0)

            image_versions = get_all_augmentations(
                img_tensor, mask_tensor, heatmap_tensor,
                DATASET_GAMMA_RANGE, DATASET_MAX_DISTORT, DATASET_DISTORT_GRID_SIZE,
            ) if is_train else [(img_tensor, mask_tensor, None, None)]

            if DATASET_SAVE_HEATMAP_VISUALIZATIONS and is_train and image_versions[0][2] is not None:
                viz_img, _, viz_hm, _ = image_versions[0]
                visualize_heatmap(viz_img, viz_hm, viz_dir /
                                  f"{png_path.stem}.png")

            for img_aug, mask_aug, hm_aug, aug_name in image_versions:
                save_tensor_as_png(img_aug, mask_aug, png_path,
                                   split_dir / HIGHRES_IMG_DIR_NAME,
                                   split_dir / HIGHRES_MASK_DIR_NAME,
                                   aug_name)
                if is_train and hm_aug is not None:
                    save_heatmap(hm_aug, png_path, split_dir /
                                 HIGHRES_HEATMAP_DIR_NAME, aug_name)


def main():
    set_seeds(SEED)
    augment_and_save()

    dataset_dir = Path(DATASETS_DIR) / DATASET_NAME
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
