'''
Creates augmented versions of EM png images and their corresponding integer segmentation masks
and saves them using the appropriate folder structure for nn-UNet training as described in
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#dataset-folder-structure
'''

import os
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import torch


SEED = 42
DATASET_FOLDER_NAME = "Dataset001_TestCvatAugmented"

MAX_DISTORT = 0.1
GRID_SIZE = (5, 5)


def load_png_as_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path)
    # divide by 255 to normalize to [0,1]
    return torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)


def grid_distort(img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if img.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(img.shape)}")
    if img.shape != mask.shape:
        raise ValueError(
            f"Image and mask must have the same shape, got {tuple(img.shape)} and {tuple(mask.shape)}")

    C, H, W = img.shape
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


def main():
    set_seeds()

    data_dir = Path(__file__).resolve().parent / "Data"
    png_dir = data_dir / "PngImages"
    integer_masks_dir = data_dir / "BinarySegmentationMasks"
    integer_masks_bw_dir = data_dir / "BinarySegmentationMasksBlackWhite"
    dataset_dir = data_dir / DATASET_FOLDER_NAME

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    (dataset_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    src_json = Path(__file__).resolve().parent / "nnUNet_dataset.json"
    if not src_json.exists():
        print("no dataset.json template found")
        return
    shutil.copy(src_json, dataset_dir / "dataset.json")

    for png_path in png_dir.rglob("*.png"):
        relative_path = png_path.relative_to(png_dir).parent
        mask_png_path = integer_masks_bw_dir / relative_path / png_path.name
        img_tensor = load_png_as_tensor(png_path)
        mask_tensor = load_png_as_tensor(mask_png_path)

        (img_tensor_griddistort, mask_tensor_griddistort) = grid_distort(
            img_tensor, mask_tensor)

        augmentations = [
            (img_tensor, mask_tensor, ""),
            (torch.flip(img_tensor, dims=(-1,)),
             torch.flip(mask_tensor, dims=(-1,)), "hflip"),
            (torch.flip(img_tensor, dims=(-2,)),
             torch.flip(mask_tensor, dims=(-2,)), "vflip"),
            (torch.rot90(img_tensor, k=1, dims=(-2, -1)),
             torch.rot90(mask_tensor, k=1, dims=(-2, -1)), "rot90"),
            (img_tensor.clamp(1e-6, 1.0).pow(random.uniform(*(0.8, 1.2))),
             mask_tensor, "gamma"),
            ((img_tensor + torch.randn_like(img_tensor) *
             0.4).clamp(0.0, 1.0), mask_tensor, "gaussiannoise"),
            (img_tensor_griddistort, mask_tensor_griddistort, "griddistort")
        ]

        for img_aug, mask_aug, aug_name in augmentations:
            target_img_folder = dataset_dir / "imagesTr" / relative_path
            target_img_folder.mkdir(parents=True, exist_ok=True)
            target_mask_folder = dataset_dir / "labelsTr" / relative_path
            target_mask_folder.mkdir(parents=True, exist_ok=True)

            base_name = png_path.stem + "_" + aug_name
            img_out_path = target_img_folder / (base_name + "_0000.png")
            mask_out_path = target_mask_folder / (base_name + ".png")

            img_out_pil = Image.fromarray(
                (img_aug.squeeze(0).numpy() * 255).astype(np.uint8))
            mask_out_pil = Image.fromarray(
                (mask_aug.squeeze(0).numpy() * 255).astype(np.uint8))

            img_out_pil.save(img_out_path)
            mask_out_pil.save(mask_out_path)

            print(
                f"Saved augmented image and mask: {img_out_path.relative_to(data_dir)} , {mask_out_path.relative_to(data_dir)}")


if __name__ == "__main__":
    main()
