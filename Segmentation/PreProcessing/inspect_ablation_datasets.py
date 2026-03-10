import os
import re
from pathlib import Path

from Segmentation.Util.env_utils import load_as, load_dotenv
from Segmentation.PreProcessing.generate_ablation_datasets import ABLATION_DATASET_BASE_NAME
from Segmentation.PreProcessing.preprocessing_utils import AUG_TYPES, load_dataset_split


load_dotenv()

DATASETS_DIR = os.getenv("DATASETS_DIR")
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
HIGHRES_IMG_DIR_NAME = os.getenv("HIGHRES_IMG_DIR_NAME", "images_4096")
HIGHRES_IMG_PATCHES_DIR_NAME = os.getenv(
    "HIGHRES_IMG_PATCHES_DIR_NAME", "img_patches_1024")
HIGHRES_MASK_PATCHES_DIR_NAME = os.getenv(
    "HIGHRES_MASK_PATCHES_DIR_NAME", "mask_patches_1024")
DATASET_VAL_SPLIT = load_as("DATASET_VAL_SPLIT", float, "0.2")


def main():
    # Inspect raw data
    raw_data_img_dir = Path(RAW_DATA_DIR) / HIGHRES_IMG_DIR_NAME
    raw_data_images = list(raw_data_img_dir.glob("*.png"))
    nof_full_images = len(
        [p for p in raw_data_images if not p.name.endswith("_soi.png")])
    nof_soi_images = len(
        [p for p in raw_data_images if p.name.endswith("_soi.png")])
    print("Inspecting raw data directory:")
    print(f"  Number of full images: {nof_full_images}")
    print(f"  Number of SOI images: {nof_soi_images}")

    num_patches = nof_full_images * 16 + nof_soi_images
    print(f"  Should result in {num_patches} total image patches")

    dataset_split = load_dataset_split(ABLATION_DATASET_BASE_NAME)
    train_split, test_split = dataset_split["train"], dataset_split["test"]
    num_full_train = len(
        [p for p in train_split if not p.endswith("_soi.png")])
    num_soi_train = len([p for p in train_split if p.endswith("_soi.png")])
    num_full_test = len([p for p in test_split if not p.endswith("_soi.png")])
    num_soi_test = len([p for p in test_split if p.endswith("_soi.png")])
    print(
        f"  number of training images: {len(train_split)}; full: {num_full_train}, SOI: {num_soi_train}")
    print(
        f"  -> {num_soi_train + 16 * num_full_train} total training image patches")
    print(
        f"  number of test images: {len(test_split)}; full: {num_full_test}, SOI: {num_soi_test}")
    print(f"  -> {num_soi_test + 16 * num_full_test} total test image patches")

    # Inspect ablation datasets
    datasets_dir = Path(DATASETS_DIR)
    ablation_datasets = sorted([d for d in datasets_dir.iterdir(
    ) if d.is_dir() and d.name.startswith(ABLATION_DATASET_BASE_NAME) and d.name != f"{ABLATION_DATASET_BASE_NAME}_shared"])
    print(f"\nFound {len(ablation_datasets)} ablation datasets")

    for dataset in ablation_datasets:
        train_dir = dataset / "train"
        val_dir = dataset / "validation"
        test_dir = dataset / "test"

        print(f"\nDataset {dataset.name}:")
        print("==============================")

        for split_name, split_dir in [("train", train_dir), ("validation", val_dir), ("test", test_dir)]:
            print(f"{split_name} split:")
            print("------------------------------")

            if split_dir.is_symlink():
                print(
                    f"  {split_name}: symlink to {split_dir.resolve()}")
            else:
                print(f"  {split_name}: no symlink")

            patches_dir = split_dir / HIGHRES_IMG_PATCHES_DIR_NAME
            masks_dir = split_dir / HIGHRES_MASK_PATCHES_DIR_NAME
            patches = list(patches_dir.glob('*.png'))
            masks = list(masks_dir.glob('*.png'))
            print(
                f"  number of image patches: {len(patches)}")
            print(
                f"  number of mask patches: {len(masks)}")

            if split_name != "train":
                continue

            num_aug_patches = len(
                [p for p in patches if any(aug in p.name for aug in AUG_TYPES)])
            num_raw_patches = len(patches) - num_aug_patches
            print(
                f"  number of raw image patches: {num_raw_patches}")
            print(
                f"  number of augmented image patches: {num_aug_patches}")
            print(
                f"  percentage of augmented image patches: {num_aug_patches / len(patches) * 100:.2f}%")


if __name__ == "__main__":
    main()
