import os
from pathlib import Path
import random
import shutil
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from scipy.ndimage import label, center_of_mass

from Segmentation.Util.env_utils import load_as, load_as_tuple, load_segmentation_env
from Segmentation.Util.dataset_util import create_patches_from_img

load_segmentation_env()

SEED = load_as("SEED", int, 42)

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
DATASETS_DIR = os.getenv("DATASETS_DIR")
DATASET_NAME = os.getenv("DATASET_NAME", "SAM_LoRA_Augmented")

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

DATASET_LOWRES_RESIZE = load_as_tuple(
    "DATASET_LOWRES_RESIZE", "1024,1024", int)
DATASET_GAUSSIAN_NOISE = load_as("DATASET_GAUSSIAN_NOISE", float, "0.05")
DATASET_GAMMA_RANGE = load_as_tuple("DATASET_GAMMA_RANGE", "0.6,1.4", float)
DATASET_MAX_DISTORT = load_as("DATASET_MAX_DISTORT", float, "0.1")
DATASET_DISTORT_GRID_SIZE = load_as_tuple(
    "DATASET_DISTORT_GRID_SIZE", "4,4", int)
DATASET_OVERSAMPLE_JUNCTION_PATCHES = load_as(
    "DATASET_OVERSAMPLE_JUNCTION_PATCHES", int, 0)

DATASET_SAVE_HEATMAP_VISUALIZATIONS = load_as(
    "DATASET_SAVE_HEATMAP_VISUALIZATIONS", bool, False)

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


def grid_distort(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Apply the same random grid distortion to all input tensors.
    All tensors must share the same spatial dimensions (H, W) but may differ in channels."""
    if not tensors:
        raise ValueError("At least one tensor required")
    ref = tensors[0]
    if ref.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(ref.shape)}")
    for t in tensors[1:]:
        if t.shape[-2:] != ref.shape[-2:]:
            raise ValueError(
                f"Spatial dimensions must match: {tuple(ref.shape[-2:])} vs {tuple(t.shape[-2:])}")

    _, H, W = ref.shape
    ny, nx = DATASET_DISTORT_GRID_SIZE
    if ny < 2 or nx < 2:
        return tensors

    ys, xs = torch.linspace(-1, 1, H, device=ref.device,
                            dtype=ref.dtype), torch.linspace(-1, 1, W, device=ref.device, dtype=ref.dtype)
    base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([base_x, base_y], dim=-1).unsqueeze(0)
    disp_coarse = torch.zeros(
        (1, 2, ny, nx), device=ref.device, dtype=ref.dtype)
    max_dy, max_dx = 2.0 * DATASET_MAX_DISTORT, 2.0 * DATASET_MAX_DISTORT

    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            disp_coarse[0, 0, iy, ix] = random.uniform(-max_dy, max_dy)
            disp_coarse[0, 1, iy, ix] = random.uniform(-max_dx, max_dx)

    disp_full = torch.nn.functional.interpolate(disp_coarse, size=(
        H, W), mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
    warped_grid = base_grid + disp_full

    results = []
    for t in tensors:
        out = torch.nn.functional.grid_sample(t.unsqueeze(0), warped_grid, mode="bilinear",
                                              padding_mode="border", align_corners=True)
        results.append(out.squeeze(0).clamp(0.0, 1.0))
    return tuple(results)


def save_tensor_as_png(tensor_img: torch.Tensor, tensor_mask: torch.Tensor, png_path: Path, out_dir_img: Path, out_dir_mask: Path, aug_name: str):
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


def save_heatmap(heatmap_tensor: torch.Tensor, png_path: Path, out_dir: Path, aug_name: str):
    aug_suffix = f"_{aug_name}" if aug_name else ""
    out_file_name = f"{png_path.stem}{aug_suffix}.npy"
    out_path = out_dir / out_file_name
    np.save(out_path, heatmap_tensor.squeeze(0).numpy())
    print(
        f"Saved augmented heatmap:\n{out_path.relative_to(out_dir.parent.parent)}")
    print()


def visualize_augmented_heatmap(img_tensor: torch.Tensor, hm_tensor: torch.Tensor, out_path: Path):
    img_np = img_tensor.squeeze(0).numpy()
    hm_np = hm_tensor.squeeze(0).numpy()

    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np, cmap="gray")
    ax.imshow(hm_np, cmap="hot", alpha=0.5,
              interpolation="bilinear", vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def init_dir(folder_path: Path):
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


def load_dataset_split() -> dict[str, list[str]]:
    splits_file = Path(__file__).resolve().parent / "dataset_splits.json"
    with open(splits_file, "r") as f:
        splits = json.load(f)
    return splits[DATASET_NAME] if DATASET_NAME in splits else None


def get_train_test_split_image_paths(raw_images_dir: Path) -> tuple[list[str], list[str]]:
    splits = load_dataset_split()

    if splits is not None:
        train_image_paths = [str(raw_images_dir / img_name)
                             for img_name in splits["train"]]
        test_image_paths = [str(raw_images_dir / img_name)
                            for img_name in splits["test"]]
    else:
        img_paths = sorted([str(p) for p in raw_images_dir.rglob("*.png")])
        random.shuffle(img_paths)
        split_idx = int(0.8 * len(img_paths))
        train_image_paths = img_paths[:split_idx]
        test_image_paths = img_paths[split_idx:]

    return train_image_paths, test_image_paths


def augment_and_save():
    raw_data_dir = Path(RAW_DATA_DIR)
    raw_images_dir = raw_data_dir / HIGHRES_IMG_DIR_NAME
    raw_masks_dir = raw_data_dir / HIGHRES_MASK_DIR_NAME
    raw_heatmaps_dir = raw_data_dir / HIGHRES_HEATMAP_DIR_NAME

    dataset_dir = Path(DATASETS_DIR) / DATASET_NAME
    train_dir, test_dir = dataset_dir / "train", dataset_dir / "test"

    init_dir(dataset_dir)
    subdirs = [HIGHRES_IMG_DIR_NAME,
               HIGHRES_MASK_DIR_NAME, HIGHRES_HEATMAP_DIR_NAME]
    for subdir in subdirs:
        (train_dir / subdir).mkdir(parents=True, exist_ok=True)
        (test_dir / subdir).mkdir(parents=True, exist_ok=True)

    train_image_paths, test_image_paths = get_train_test_split_image_paths(
        raw_images_dir)

    if DATASET_SAVE_HEATMAP_VISUALIZATIONS:
        viz_dir = dataset_dir / HEATMAP_VISUALIZATION_DIR_NAME
        viz_dir.mkdir(parents=True, exist_ok=True)

    print("training images: ", str([Path(p).name for p in train_image_paths]))
    print("testing images: ", str([Path(p).name for p in test_image_paths]))

    for png_path in list(raw_images_dir.rglob("*.png")):
        if str(png_path) not in train_image_paths and str(png_path) not in test_image_paths:
            continue

        img_tensor = load_png_as_tensor(png_path)
        mask_tensor = load_png_as_tensor(raw_masks_dir / png_path.name)

        heatmap_npy_path = raw_heatmaps_dir / f"{png_path.stem}.npy"
        heatmap_tensor = torch.from_numpy(
            np.load(heatmap_npy_path)).unsqueeze(0)

        # Precompute grid distortion with same displacement field for image, mask, and heatmap
        distort_inputs = [img_tensor, mask_tensor, heatmap_tensor]
        distorted = grid_distort(*distort_inputs)
        img_dist, mask_dist, hm_dist = distorted[0], distorted[1], distorted[2]

        augmentations = [
            (img_tensor, mask_tensor, heatmap_tensor, None),
            (torch.flip(img_tensor, dims=(-1,)),
             torch.flip(mask_tensor, dims=(-1,)),
             torch.flip(heatmap_tensor, dims=(-1,)), "hflip"),
            (torch.flip(img_tensor, dims=(-2,)),
             torch.flip(mask_tensor, dims=(-2,)),
             torch.flip(heatmap_tensor, dims=(-2,)), "vflip"),
            (torch.rot90(img_tensor, k=1, dims=(-2, -1)),
             torch.rot90(mask_tensor, k=1, dims=(-2, -1)),
             torch.rot90(heatmap_tensor, k=1, dims=(-2, -1)), "rot90"),
            (torch.rot90(img_tensor, k=2, dims=(-2, -1)),
             torch.rot90(mask_tensor, k=2, dims=(-2, -1)),
             torch.rot90(heatmap_tensor, k=2, dims=(-2, -1)), "rot180"),
            (torch.rot90(img_tensor, k=3, dims=(-2, -1)),
             torch.rot90(mask_tensor, k=3, dims=(-2, -1)),
             torch.rot90(heatmap_tensor, k=3, dims=(-2, -1)), "rot270"),
            (img_tensor.clamp(1e-6, 1.0).pow(random.uniform(*DATASET_GAMMA_RANGE)),
             mask_tensor, heatmap_tensor, "gamma"),
            (img_dist, mask_dist, hm_dist, "griddistort"),
        ]

        if DATASET_SAVE_HEATMAP_VISUALIZATIONS:
            viz_idx = random.randint(0, len(augmentations) - 1)
            viz_img, _, viz_hm, viz_aug_name = augmentations[viz_idx]
            aug_label = viz_aug_name if viz_aug_name else "original"
            visualize_augmented_heatmap(
                viz_img, viz_hm, viz_dir / f"{png_path.stem}_{aug_label}.png")

        train_test_dir = train_dir if str(
            png_path) in train_image_paths else test_dir

        for img_aug, mask_aug, hm_aug, aug_name in augmentations:
            out_dir_images = train_test_dir / HIGHRES_IMG_DIR_NAME
            out_dir_masks = train_test_dir / HIGHRES_MASK_DIR_NAME
            save_tensor_as_png(img_aug, mask_aug, png_path,
                               out_dir_images, out_dir_masks, aug_name)
            save_heatmap(hm_aug, png_path, train_test_dir /
                         HIGHRES_HEATMAP_DIR_NAME, aug_name)


def create_patches_from_npy(npy_path: Path, patch_size: int = 1024) -> list[np.ndarray]:
    heatmap = np.load(npy_path).astype(np.float32)
    t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    patches = t.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size)
    patches = patches.permute(
        0, 2, 3, 1, 4, 5).reshape(-1, 1, patch_size, patch_size)
    return [p.squeeze(0).numpy() for p in patches]


def create_patches_and_save():
    base_dirs = [
        Path(DATASETS_DIR) / DATASET_NAME / "train",
        Path(DATASETS_DIR) / DATASET_NAME / "test",
    ]

    for base_dir in base_dirs:
        highres_images_dir = Path(base_dir) / HIGHRES_IMG_DIR_NAME
        highres_masks_dir = Path(base_dir) / HIGHRES_MASK_DIR_NAME
        highres_heatmaps_dir = Path(base_dir) / HIGHRES_HEATMAP_DIR_NAME

        highres_img_patches_dir = Path(base_dir) / HIGHRES_IMG_PATCHES_DIR_NAME
        highres_mask_patches_dir = Path(
            base_dir) / HIGHRES_MASK_PATCHES_DIR_NAME
        highres_heatmap_patches_dir = Path(
            base_dir) / HIGHRES_HEATMAP_PATCHES_DIR_NAME

        patch_dirs = [highres_img_patches_dir,
                      highres_mask_patches_dir, highres_heatmap_patches_dir]
        for dir in patch_dirs:
            init_dir(dir)

        for in_dir, out_dir in [(highres_images_dir, highres_img_patches_dir),
                                (highres_masks_dir, highres_mask_patches_dir)]:
            for png_file in in_dir.glob("*.png"):
                print(f"Cropping patches from image {png_file.name}")

                patches = create_patches_from_img(png_file, patch_size=1024)
                for i, patch in enumerate(patches):
                    patch_img = F.to_pil_image(patch)
                    patch_img.save(
                        out_dir / f"{png_file.stem}_patch_{i:02d}.png")

        for npy_file in highres_heatmaps_dir.glob("*.npy"):
            print(f"Cropping patches from heatmap {npy_file.name}")
            patches = create_patches_from_npy(npy_file, patch_size=1024)
            for i, patch in enumerate(patches):
                np.save(highres_heatmap_patches_dir /
                        f"{npy_file.stem}_patch_{i:02d}.npy", patch)


def find_junction_centers(heatmap: np.ndarray, threshold: float = 0.95) -> list[tuple[int, int]]:
    """Find junction centers as local maxima in the heatmap.
    Returns list of (x, y) integer coordinates."""
    binary = heatmap >= threshold
    labeled, num_features = label(binary)
    if num_features == 0:
        return []
    centers = center_of_mass(heatmap, labeled, range(1, num_features + 1))
    return [(int(round(x)), int(round(y))) for y, x in centers]


def oversample_junction_patches():
    """For each junction point in each augmented training image, create one random
    1024x1024 patch that contains the junction at a random (non-centered) position.
    Saves image, mask, and heatmap patches alongside the grid patches."""
    train_dir = Path(DATASETS_DIR) / DATASET_NAME / "train"

    highres_images_dir = train_dir / HIGHRES_IMG_DIR_NAME
    highres_masks_dir = train_dir / HIGHRES_MASK_DIR_NAME
    highres_heatmaps_dir = train_dir / HIGHRES_HEATMAP_DIR_NAME

    img_patches_dir = train_dir / HIGHRES_IMG_PATCHES_DIR_NAME
    mask_patches_dir = train_dir / HIGHRES_MASK_PATCHES_DIR_NAME
    heatmap_patches_dir = train_dir / HIGHRES_HEATMAP_PATCHES_DIR_NAME

    patch_size = 1024
    total_patches = 0

    for npy_file in sorted(highres_heatmaps_dir.glob("*.npy")):
        stem = npy_file.stem

        # if stem.endswith("_soi"):
        #    # skip SoI images, because they're already 1024x1024
        #    continue

        img_path = highres_images_dir / f"{stem}.png"
        mask_path = highres_masks_dir / f"{stem}.png"

        heatmap = np.load(npy_file).astype(np.float32)
        centers = find_junction_centers(heatmap)
        if not centers:
            continue

        img_tensor = load_png_as_tensor(img_path)
        mask_tensor = load_png_as_tensor(mask_path)
        hm_tensor = torch.from_numpy(heatmap).unsqueeze(0)

        _, H, W = img_tensor.shape
        if H < patch_size or W < patch_size:
            print(
                f"Skipping {stem}: image too small for {patch_size}x{patch_size} patches")
            continue

        for j, (cx, cy) in enumerate(centers):
            min_left = max(0, cx - patch_size + 1)
            max_left = min(W - patch_size, cx)
            min_top = max(0, cy - patch_size + 1)
            max_top = min(H - patch_size, cy)

            if min_left > max_left or min_top > max_top:
                continue

            for k in range(DATASET_OVERSAMPLE_JUNCTION_PATCHES):
                left = random.randint(min_left, max_left)
                top = random.randint(min_top, max_top)

                img_patch = img_tensor[:, top:top +
                                       patch_size, left:left + patch_size]
                mask_patch = mask_tensor[:, top:top +
                                         patch_size, left:left + patch_size]
                hm_patch = hm_tensor[:, top:top +
                                     patch_size, left:left + patch_size]

                patch_name = f"{stem}_junctionpatch_{j:02d}_{k:02d}"

                F.to_pil_image(img_patch).save(
                    img_patches_dir / f"{patch_name}.png")
                F.to_pil_image(mask_patch).save(
                    mask_patches_dir / f"{patch_name}.png")
                np.save(heatmap_patches_dir / f"{patch_name}.npy",
                        hm_patch.squeeze(0).numpy())

                total_patches += 1

        print(f"Created {len(centers)} junction patches for {stem}.png")

    print(f"\nTotal junction oversampling patches created: {total_patches}")


def remove_highres_dirs():
    base_dirs = [
        Path(DATASETS_DIR) / DATASET_NAME / "train",
        Path(DATASETS_DIR) / DATASET_NAME / "test",
    ]

    for base_dir in base_dirs:
        for dir_name in [HIGHRES_IMG_DIR_NAME, HIGHRES_MASK_DIR_NAME, HIGHRES_HEATMAP_DIR_NAME]:
            d = Path(base_dir) / dir_name
            if d.exists():
                shutil.rmtree(d)


def main():
    set_seeds()
    augment_and_save()
    create_patches_and_save()
    if DATASET_OVERSAMPLE_JUNCTION_PATCHES > 0:
        oversample_junction_patches()
    remove_highres_dirs()


if __name__ == "__main__":
    main()
