import os
from pathlib import Path
import shutil
import torchvision.transforms.functional as F
from PIL import Image

from .env_utils import load_segmentation_env

load_segmentation_env()

DATASETS_DIR = os.getenv("DATASETS_DIR")

DATASET_NAME = os.getenv("DATASET_NAME", "SAM_LoRA_Augmented")

HIGHRES_IMG_DIR_NAME = os.getenv("HIGHRES_IMG_DIR_NAME", "images_4096")
HIGHRES_MASK_DIR_NAME = os.getenv("HIGHRES_MASK_DIR_NAME", "masks_4096")
LOWRES_IMG_DIR_NAME = os.getenv("LOWRES_IMG_DIR_NAME", "images_1024")
LOWRES_MASK_DIR_NAME = os.getenv("LOWRES_MASK_DIR_NAME", "masks_1024")
CROPPED_AUG_IMG_DIR_NAME = os.getenv("CROPPED_AUG_IMG_DIR_NAME", "images_256")
CROPPED_AUG_MASK_DIR_NAME = os.getenv("CROPPED_AUG_MASK_DIR_NAME", "masks_256")

if not DATASETS_DIR:
    raise ValueError("DATASETS_DIR environment variable must be set.")

base_dirs = [
    Path(DATASETS_DIR) / DATASET_NAME / "train",
    Path(DATASETS_DIR) / DATASET_NAME / "test",
]


def init_folder(folder_path: Path):
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


def crop_image_grid(input_image_path: Path, output_dir: Path):
    img = Image.open(input_image_path)
    img = F.to_tensor(img).unsqueeze(0)

    patch_size = 256
    patches = img.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size)
    _, C, _, _, H, W = patches.shape
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, H, W)

    for i, patch in enumerate(patches):
        patch_img = F.to_pil_image(patch)
        patch_img.save(output_dir / f"{input_image_path.stem}_patch_{i}.png")


for base_dir in base_dirs:
    images_dir = Path(base_dir) / LOWRES_IMG_DIR_NAME
    masks_dir = Path(base_dir) / LOWRES_MASK_DIR_NAME
    cropped_images_dir = Path(base_dir) / CROPPED_AUG_IMG_DIR_NAME
    cropped_masks_dir = Path(base_dir) / CROPPED_AUG_MASK_DIR_NAME

    if not images_dir.exists() or not masks_dir.exists():
        print(
            f"Skipping {base_dir} as it does not contain images/ and masks/ directories")
        continue

    init_folder(cropped_images_dir)
    init_folder(cropped_masks_dir)

    for png_file in images_dir.glob("*.png"):
        print(f"Cropping image patches from image {png_file.name}...")
        crop_image_grid(png_file, cropped_images_dir)

    for png_file in masks_dir.glob("*.png"):
        print(f"Cropping mask patches from mask {png_file.name}...")
        crop_image_grid(png_file, cropped_masks_dir)
