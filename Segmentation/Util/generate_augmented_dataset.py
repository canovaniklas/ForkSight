import os
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from Segmentation.Util.env_utils import load_as, load_as_tuple, load_segmentation_env
from Segmentation.Util.dataset_util import create_patches_from_img

load_segmentation_env()

SEED = load_as("SEED", int, 42)

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
DATASETS_DIR = os.getenv("DATASETS_DIR")
DATASET_NAME = os.getenv("DATASET_NAME", "SAM_LoRA_Augmented")

HIGHRES_IMG_DIR_NAME = os.getenv("HIGHRES_IMG_DIR_NAME", "images_4096")
HIGHRES_MASK_DIR_NAME = os.getenv("HIGHRES_MASK_DIR_NAME", "masks_4096")
LOWRES_IMG_DIR_NAME = os.getenv("LOWRES_IMG_DIR_NAME", "images_1024")
LOWRES_MASK_DIR_NAME = os.getenv("LOWRES_MASK_DIR_NAME", "masks_1024")

LOWRES_IMG_PATCHES_DIR_NAME = os.getenv(
    "LOWRES_IMG_PATCHES_DIR_NAME", "img_patches_256")
LOWRES_MASK_PATCHES_DIR_NAME = os.getenv(
    "LOWRES_MASK_PATCHES_DIR_NAME", "mask_patches_256")
HIGHRES_IMG_PATCHES_DIR_NAME = os.getenv(
    "HIGHRES_IMG_PATCHES_DIR_NAME", "img_patches_1024")
HIGHRES_MASK_PATCHES_DIR_NAME = os.getenv(
    "HIGHRES_MASK_PATCHES_DIR_NAME", "mask_patches_1024")


DATASET_LOWRES_RESIZE = load_as_tuple(
    "DATASET_LOWRES_RESIZE", "1024,1024", int)
DATASET_GAUSSIAN_NOISE = load_as("DATASET_GAUSSIAN_NOISE", float, "0.05")
DATASET_GAMMA_RANGE = load_as_tuple("DATASET_GAMMA_RANGE", "0.6,1.4", float)
DATASET_MAX_DISTORT = load_as("DATASET_MAX_DISTORT", float, "0.1")
DATASET_DISTORT_GRID_SIZE = load_as_tuple(
    "DATASET_DISTORT_GRID_SIZE", "4,4", int)
DATASET_RANDOM_CROP_SIZE = load_as_tuple(
    "DATASET_RANDOM_CROP_SIZE", "2048,2048", int)

if not RAW_DATA_DIR or not DATASETS_DIR:
    raise ValueError(
        "RAW_DATA_DIR and DATASETS_DIR environment variables must be set.")


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def load_png_as_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path)
    transform = transforms.ToTensor()
    return transform(img)


def grid_distort(img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if img.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(img.shape)}")
    if img.shape != mask.shape:
        raise ValueError(
            f"Image and mask must have the same shape, got {tuple(img.shape)} and {tuple(mask.shape)}")

    _, H, W = img.shape
    ny, nx = DATASET_DISTORT_GRID_SIZE
    if ny < 2 or nx < 2:
        return img, mask

    ys, xs = torch.linspace(-1, 1, H, device=img.device,
                            dtype=img.dtype), torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype)
    base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([base_x, base_y], dim=-1).unsqueeze(0)
    disp_coarse = torch.zeros(
        (1, 2, ny, nx), device=img.device, dtype=img.dtype)
    max_dy, max_dx = 2.0 * DATASET_MAX_DISTORT, 2.0 * DATASET_MAX_DISTORT

    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            disp_coarse[0, 0, iy, ix] = random.uniform(-max_dy, max_dy)
            disp_coarse[0, 1, iy, ix] = random.uniform(-max_dx, max_dx)

    disp_full = torch.nn.functional.interpolate(disp_coarse, size=(
        H, W), mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
    warped_grid = base_grid + disp_full
    img_out = torch.nn.functional.grid_sample(img.unsqueeze(0), warped_grid, mode="bilinear",
                                              padding_mode="border", align_corners=True)
    mask_out = torch.nn.functional.grid_sample(mask.unsqueeze(0), warped_grid, mode="bilinear",
                                               padding_mode="border", align_corners=True)
    return img_out.squeeze(0).clamp(0.0, 1.0), mask_out.squeeze(0).clamp(0.0, 1.0)


def random_crop_pair(img: torch.Tensor, mask: torch.Tensor):
    _, H, W = img.shape

    top = torch.randint(0, H - DATASET_RANDOM_CROP_SIZE[0] + 1, (1,)).item()
    left = torch.randint(0, W - DATASET_RANDOM_CROP_SIZE[1] + 1, (1,)).item()

    cropped_img = img[..., top:top + DATASET_RANDOM_CROP_SIZE[0],
                      left:left + DATASET_RANDOM_CROP_SIZE[1]]
    cropped_mask = mask[..., top:top + DATASET_RANDOM_CROP_SIZE[0],
                        left:left + DATASET_RANDOM_CROP_SIZE[1]]
    return (cropped_img, cropped_mask)


def save_tensor_as_png(tensor_img: torch.Tensor, tensor_mask: torch.Tensor, png_path: Path, out_dir_img: Path, out_dir_mask: Path, aug_name: str, resize: bool):
    if resize:
        img_transform = transforms.Resize(
            DATASET_LOWRES_RESIZE, interpolation=transforms.InterpolationMode.BILINEAR)
        mask_transform = transforms.Resize(
            DATASET_LOWRES_RESIZE, interpolation=transforms.InterpolationMode.NEAREST)
        tensor_img = img_transform(tensor_img)
        tensor_mask = mask_transform(tensor_mask)

    aug_suffix = f"_{aug_name}" if aug_name else ""
    out_file_name = F"{png_path.name.replace('.png', '')}{aug_suffix}.png"
    img_out_path = out_dir_img / out_file_name
    mask_out_path = out_dir_mask / out_file_name

    img_out_pil = Image.fromarray(
        (tensor_img.squeeze(0).numpy() * 255).astype(np.uint8))
    mask_out_pil = Image.fromarray(
        (tensor_mask.squeeze(0).numpy() * 255).astype(np.uint8))

    img_out_pil.save(img_out_path)
    mask_out_pil.save(mask_out_path)

    print(
        f"Saved augmented image and mask:\n{img_out_path.relative_to(out_dir_img.parent.parent)}\n{mask_out_path.relative_to(out_dir_mask.parent.parent)}")
    print()


def init_dir(folder_path: Path):
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


def augment_and_save():
    raw_data_dir = Path(RAW_DATA_DIR)
    raw_images_dir = raw_data_dir / HIGHRES_IMG_DIR_NAME
    raw_masks_dir = raw_data_dir / HIGHRES_MASK_DIR_NAME

    dataset_dir = Path(DATASETS_DIR) / DATASET_NAME
    train_dir, test_dir = dataset_dir / "train", dataset_dir / "test"

    init_dir(dataset_dir)
    for subdir in [HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, LOWRES_IMG_DIR_NAME, LOWRES_MASK_DIR_NAME]:
        (train_dir / subdir).mkdir(parents=True, exist_ok=True)
        (test_dir / subdir).mkdir(parents=True, exist_ok=True)

    img_paths = sorted([str(p) for p in raw_images_dir.rglob("*.png")])
    random.shuffle(img_paths)
    split_idx = int(0.9 * len(img_paths))
    train_image_paths = img_paths[:split_idx]
    test_image_paths = img_paths[split_idx:]

    print("training images: ", str([Path(p).name for p in train_image_paths]))
    print("testing images: ", str([Path(p).name for p in test_image_paths]))

    for png_path in raw_images_dir.rglob("*.png"):
        img_tensor = load_png_as_tensor(png_path)
        mask_tensor = load_png_as_tensor(raw_masks_dir / png_path.name)

        augmentations = [
            (img_tensor, mask_tensor, None),
            (torch.flip(img_tensor, dims=(-1,)),
             torch.flip(mask_tensor, dims=(-1,)), "hflip"),
            (torch.flip(img_tensor, dims=(-2,)),
             torch.flip(mask_tensor, dims=(-2,)), "vflip"),
            (torch.rot90(img_tensor, k=1, dims=(-2, -1)),
             torch.rot90(mask_tensor, k=1, dims=(-2, -1)), "rot90"),
            (torch.rot90(img_tensor, k=2, dims=(-2, -1)),
             torch.rot90(mask_tensor, k=2, dims=(-2, -1)), "rot180"),
            (torch.rot90(img_tensor, k=3, dims=(-2, -1)),
             torch.rot90(mask_tensor, k=3, dims=(-2, -1)), "rot270"),
            (img_tensor.clamp(1e-6, 1.0).pow(random.uniform(*DATASET_GAMMA_RANGE)),
             mask_tensor, "gamma"),
            ((img_tensor + torch.randn_like(img_tensor) *
              DATASET_GAUSSIAN_NOISE).clamp(0.0, 1.0), mask_tensor, "gaussiannoise"),
            (*grid_distort(img_tensor, mask_tensor), "griddistort")
        ]

        random_crops = [random_crop_pair(
            img_tensor, mask_tensor) for _ in range(5)]
        for idx, (img_crop, mask_crop) in enumerate(random_crops):
            augmentations.append((img_crop, mask_crop, f"randomcrop{idx}"))

        train_test_dir = train_dir if str(
            png_path) in train_image_paths else test_dir
        subdirs = [(HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, False),
                   (LOWRES_IMG_DIR_NAME, LOWRES_MASK_DIR_NAME, True)]

        for img_aug, mask_aug, aug_name in augmentations:
            for subdir_img, subdir_mask, resize in subdirs:
                out_dir_images = train_test_dir / subdir_img
                out_dir_masks = train_test_dir / subdir_mask
                save_tensor_as_png(img_aug, mask_aug, png_path,
                                   out_dir_images, out_dir_masks, aug_name, resize)


def create_patches_and_save():
    base_dirs = [
        Path(DATASETS_DIR) / DATASET_NAME / "train",
        Path(DATASETS_DIR) / DATASET_NAME / "test",
    ]

    for base_dir in base_dirs:
        lowres_images_dir = Path(base_dir) / LOWRES_IMG_DIR_NAME
        lowres_masks_dir = Path(base_dir) / LOWRES_MASK_DIR_NAME
        highres_images_dir = Path(base_dir) / HIGHRES_IMG_DIR_NAME
        highres_masks_dir = Path(base_dir) / HIGHRES_MASK_DIR_NAME

        lowres_img_patches_dir = Path(base_dir) / LOWRES_IMG_PATCHES_DIR_NAME
        lowres_mask_patches_dir = Path(base_dir) / LOWRES_MASK_PATCHES_DIR_NAME
        highres_img_patches_dir = Path(base_dir) / HIGHRES_IMG_PATCHES_DIR_NAME
        highres_mask_patches_dir = Path(
            base_dir) / HIGHRES_MASK_PATCHES_DIR_NAME

        for dir in [lowres_img_patches_dir, lowres_mask_patches_dir, highres_img_patches_dir, highres_mask_patches_dir]:
            init_dir(dir)

        for in_dir, out_dir in [(lowres_images_dir, lowres_img_patches_dir),
                                (lowres_masks_dir, lowres_mask_patches_dir),
                                (highres_images_dir, highres_img_patches_dir),
                                (highres_masks_dir, highres_mask_patches_dir)]:
            for png_file in in_dir.glob("*.png"):
                print(f"Cropping patches from image {png_file.name}")

                patches = create_patches_from_img(
                    png_file, nof_division_per_dim=4)
                for i, patch in enumerate(patches):
                    patch_img = F.to_pil_image(patch)
                    patch_img.save(
                        out_dir / f"{png_file.stem}_patch_{i:02d}.png")


def remove_highres_dirs():
    base_dirs = [
        Path(DATASETS_DIR) / DATASET_NAME / "train",
        Path(DATASETS_DIR) / DATASET_NAME / "test",
    ]

    for base_dir in base_dirs:
        highres_images_dir = Path(base_dir) / HIGHRES_IMG_DIR_NAME
        highres_masks_dir = Path(base_dir) / HIGHRES_MASK_DIR_NAME
        if highres_images_dir.exists():
            shutil.rmtree(highres_images_dir)
        if highres_masks_dir.exists():
            shutil.rmtree(highres_masks_dir)


def main():
    set_seeds()
    augment_and_save()
    create_patches_and_save()
    remove_highres_dirs()


if __name__ == "__main__":
    main()
