import os
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

SEED = 42

RAW_DATA_DIR = "/data/jhehli/raw_data"
DATASETS_DIR = "/data/jhehli/datasets"
DATASET_NAME = "SAM_LoRA_Augmented"

RESIZE = (1024, 1024)

P_HFLIP = 0.5
P_VFLIP = 0.5
P_ROT = 0.5
P_GAMMA = 0.5
P_NOISE = 0.5
P_GRID_DISTORT = 0.5
GAUSSIAN_NOISE = 0.05
GAMMA_RANGE = (0.6, 1.4)
MAX_DISTORT = 0.1
GRID_SIZE = (4, 4)
CROP_SIZE = (2048, 2048)

USE_BW_MASKS = True


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
    ny, nx = GRID_SIZE
    if ny < 2 or nx < 2:
        return img, mask

    ys, xs = torch.linspace(-1, 1, H, device=img.device,
                            dtype=img.dtype), torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype)
    base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([base_x, base_y], dim=-1).unsqueeze(0)
    disp_coarse = torch.zeros(
        (1, 2, ny, nx), device=img.device, dtype=img.dtype)
    max_dy, max_dx = 2.0 * MAX_DISTORT, 2.0 * MAX_DISTORT

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

    top = torch.randint(0, H - CROP_SIZE[0] + 1, (1,)).item()
    left = torch.randint(0, W - CROP_SIZE[1] + 1, (1,)).item()

    cropped_img = img[..., top:top + CROP_SIZE[0], left:left + CROP_SIZE[1]]
    cropped_mask = mask[..., top:top + CROP_SIZE[0], left:left + CROP_SIZE[1]]
    return (cropped_img, cropped_mask)


def save_tensor_as_png(tensor_img: torch.Tensor, tensor_mask: torch.Tensor, png_path: Path, out_dir_img: Path, out_dir_mask: Path, aug_name: str):
    transform = transforms.Resize(
        RESIZE, interpolation=transforms.InterpolationMode.BILINEAR)
    tensor_img = transform(tensor_img)
    tensor_mask = transform(tensor_mask)

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
        f"Saved augmented image and mask:\n{img_out_path.relative_to(out_dir_img.parent)}\n{mask_out_path.relative_to(out_dir_mask.parent)}")
    print()


def main():
    set_seeds()

    raw_data_dir = Path(RAW_DATA_DIR)
    raw_images_dir = raw_data_dir / "images"
    raw_masks_dir = raw_data_dir / "masks"

    dataset_dir = Path(DATASETS_DIR) / DATASET_NAME
    train_dir_images = dataset_dir / "train" / "images"
    train_dir_masks = dataset_dir / "train" / "masks"
    test_dir_images = dataset_dir / "test" / "images"
    test_dir_masks = dataset_dir / "test" / "masks"

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_dir_images.mkdir(parents=True, exist_ok=True)
    train_dir_masks.mkdir(parents=True, exist_ok=True)
    test_dir_images.mkdir(parents=True, exist_ok=True)
    test_dir_masks.mkdir(parents=True, exist_ok=True)

    for png_path in raw_images_dir.rglob("*.png"):
        img_tensor = load_png_as_tensor(png_path)

        mask_png_path = raw_masks_dir / png_path.name
        try:
            mask_tensor = load_png_as_tensor(mask_png_path)
        except FileNotFoundError as e:
            # If the corresponding mask file is not found, skip this image
            print(f"Mask file not found for image {png_path}: {e}")
            continue

        (img_tensor_griddistort, mask_tensor_griddistort) = grid_distort(
            img_tensor, mask_tensor)

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
            (img_tensor.clamp(1e-6, 1.0).pow(random.uniform(*(0.6, 1.4))),
             mask_tensor, "gamma"),
            ((img_tensor + torch.randn_like(img_tensor) *
              GAUSSIAN_NOISE).clamp(0.0, 1.0), mask_tensor, "gaussiannoise"),
            (img_tensor_griddistort, mask_tensor_griddistort, "griddistort")
        ]

        random_crops = [random_crop_pair(
            img_tensor, mask_tensor) for _ in range(5)]
        for idx, (img_crop, mask_crop) in enumerate(random_crops):
            augmentations.append((img_crop, mask_crop, f"randomcrop{idx}"))

        test_augmentations = [
            (torch.rot90(img_tensor, k=3, dims=(-2, -1)),
             torch.rot90(mask_tensor, k=3, dims=(-2, -1)), "rot270"),
        ]

        test_random_crops = [random_crop_pair(
            img_tensor, mask_tensor) for _ in range(2)]
        for idx, (img_crop, mask_crop) in enumerate(test_random_crops):
            test_augmentations.append(
                (img_crop, mask_crop, f"randomcrop{idx}"))

        for img_aug, mask_aug, aug_name in augmentations:
            save_tensor_as_png(img_aug, mask_aug, png_path,
                               train_dir_images, train_dir_masks, aug_name)

        for img_aug, mask_aug, aug_name in test_augmentations:
            save_tensor_as_png(img_aug, mask_aug, png_path,
                               test_dir_images, test_dir_masks, aug_name)


if __name__ == "__main__":
    main()
