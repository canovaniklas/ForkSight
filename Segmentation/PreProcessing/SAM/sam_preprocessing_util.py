import json
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from scipy.ndimage import label, center_of_mass

from Segmentation.PreProcessing.General.preprocessing_util import create_patches_from_img


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_png_as_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path)
    return transforms.ToTensor()(img)


def init_dir(dir_path: Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)


def grid_distort(
    *tensors: torch.Tensor,
    max_distort: float,
    grid_size: tuple[int, int],
) -> tuple[torch.Tensor, ...]:
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
    ny, nx = grid_size
    if ny < 2 or nx < 2:
        return tensors

    ys = torch.linspace(-1, 1, H, device=ref.device, dtype=ref.dtype)
    xs = torch.linspace(-1, 1, W, device=ref.device, dtype=ref.dtype)
    base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([base_x, base_y], dim=-1).unsqueeze(0)
    disp_coarse = torch.zeros(
        (1, 2, ny, nx), device=ref.device, dtype=ref.dtype)
    max_d = 2.0 * max_distort

    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            disp_coarse[0, 0, iy, ix] = random.uniform(-max_d, max_d)
            disp_coarse[0, 1, iy, ix] = random.uniform(-max_d, max_d)

    disp_full = torch.nn.functional.interpolate(
        disp_coarse, size=(H, W), mode="bicubic", align_corners=True
    ).permute(0, 2, 3, 1)
    warped_grid = base_grid + disp_full

    results = []
    for t in tensors:
        out = torch.nn.functional.grid_sample(
            t.unsqueeze(0), warped_grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        results.append(out.squeeze(0).clamp(0.0, 1.0))
    return tuple(results)


def save_tensor_as_png(
    tensor_img: torch.Tensor,
    tensor_mask: torch.Tensor,
    png_path: Path,
    out_dir_img: Path,
    out_dir_mask: Path,
    aug_name: str | None,
):
    aug_suffix = f"_{aug_name}" if aug_name else ""
    out_file_name = f"{png_path.stem}{aug_suffix}.png"
    img_out_path = out_dir_img / out_file_name
    mask_out_path = out_dir_mask / out_file_name

    Image.fromarray(
        (tensor_img.squeeze(0).numpy() * 255).astype(np.uint8)
    ).save(img_out_path)
    Image.fromarray(
        (tensor_mask.squeeze(0).numpy() * 255).astype(np.uint8)
    ).save(mask_out_path)

    print(
        f"Saved image and mask:\n{img_out_path.relative_to(out_dir_img.parent.parent)}\n{mask_out_path.relative_to(out_dir_mask.parent.parent)}")
    print()


def save_heatmap(
    heatmap_tensor: torch.Tensor,
    png_path: Path,
    out_dir: Path,
    aug_name: str | None,
):
    aug_suffix = f"_{aug_name}" if aug_name else ""
    out_file_name = f"{png_path.stem}{aug_suffix}.npy"
    out_path = out_dir / out_file_name
    np.save(out_path, heatmap_tensor.squeeze(0).numpy())
    print(f"Saved heatmap:\n{out_path.relative_to(out_dir.parent.parent)}")
    print()


def visualize_heatmap(img_tensor: torch.Tensor, hm_tensor: torch.Tensor, out_path: Path):
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


def get_all_augmentations(
    img_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    heatmap_tensor: torch.Tensor,
    gamma_range: tuple[float, float],
    max_distort: float,
    grid_size: tuple[int, int],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str | None]]:
    """Returns all augmentation variants including the original (aug_name=None)."""
    results = [(img_tensor, mask_tensor, heatmap_tensor, None)]
    for aug_type in AUG_TYPES:
        img_aug, mask_aug, hm_aug = apply_augmentation(
            img_tensor, mask_tensor, heatmap_tensor,
            aug_type, gamma_range, max_distort, grid_size,
        )
        results.append((img_aug, mask_aug, hm_aug, aug_type))
    return results


AUG_TYPES = ["hflip", "vflip", "rot90",
             "rot180", "rot270", "gamma", "griddistort"]


def apply_augmentation(
    img_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    heatmap_tensor: torch.Tensor,
    aug_type: str,
    gamma_range: tuple[float, float],
    max_distort: float,
    grid_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a specific named augmentation to img, mask, and heatmap."""
    if aug_type == "hflip":
        return (torch.flip(img_tensor, dims=(-1,)), torch.flip(mask_tensor, dims=(-1,)), torch.flip(heatmap_tensor, dims=(-1,)))
    elif aug_type == "vflip":
        return (torch.flip(img_tensor, dims=(-2,)), torch.flip(mask_tensor, dims=(-2,)), torch.flip(heatmap_tensor, dims=(-2,)))
    elif aug_type == "rot90":
        return (torch.rot90(img_tensor, k=1, dims=(-2, -1)), torch.rot90(mask_tensor, k=1, dims=(-2, -1)), torch.rot90(heatmap_tensor, k=1, dims=(-2, -1)))
    elif aug_type == "rot180":
        return (torch.rot90(img_tensor, k=2, dims=(-2, -1)), torch.rot90(mask_tensor, k=2, dims=(-2, -1)), torch.rot90(heatmap_tensor, k=2, dims=(-2, -1)))
    elif aug_type == "rot270":
        return (torch.rot90(img_tensor, k=3, dims=(-2, -1)), torch.rot90(mask_tensor, k=3, dims=(-2, -1)), torch.rot90(heatmap_tensor, k=3, dims=(-2, -1)))
    elif aug_type == "gamma":
        return (img_tensor.clamp(1e-6, 1.0).pow(random.uniform(*gamma_range)), mask_tensor, heatmap_tensor)
    elif aug_type == "griddistort":
        img_d, mask_d, hm_d = grid_distort(
            img_tensor, mask_tensor, heatmap_tensor,
            max_distort=max_distort, grid_size=grid_size,
        )
        return (img_d, mask_d, hm_d)
    else:
        raise ValueError(
            f"Unknown aug_type: {aug_type!r}. Must be one of {AUG_TYPES}")


def load_dataset_split(dataset_name: str) -> dict[str, list[str]] | None:
    splits_file = Path(__file__).resolve().parent / "dataset_splits.json"
    with open(splits_file, "r") as f:
        splits = json.load(f)
    return splits.get(dataset_name)


def get_train_val_test_split_paths(
    raw_images_dir: Path,
    dataset_name: str,
    val_split: float,
    seed: int,

) -> tuple[list[Path], list[Path], list[Path]]:
    splits = load_dataset_split(dataset_name)
    assert splits is not None, f"dataset_splits.json doesn't contain split for '{dataset_name}'"
    assert 0.0 <= val_split < 1.0, "val_split must be in [0, 1)"

    train_paths = sorted([raw_images_dir / n for n in splits["train"]])
    test_paths = sorted([raw_images_dir / n for n in splits["test"]])

    train_paths_full = [
        p for p in train_paths if not p.name.endswith("_soi.png")]
    train_paths_soi = [p for p in train_paths if p.name.endswith("_soi.png")]

    rng = random.Random(seed)
    rng.shuffle(train_paths_full)
    val_count_full = int(round(val_split * len(train_paths_full)))
    val_paths_full = train_paths_full[:val_count_full]
    train_paths_full = train_paths_full[val_count_full:]

    rng = random.Random(seed)
    rng.shuffle(train_paths_soi)
    val_count_soi = int(round(val_split * len(train_paths_soi)))
    val_paths_soi = train_paths_soi[:val_count_soi]
    train_paths_soi = train_paths_soi[val_count_soi:]

    return train_paths_full + train_paths_soi, val_paths_full + val_paths_soi, test_paths


def create_patches_from_npy(npy_path: Path, patch_size: int = 1024) -> list[np.ndarray]:
    heatmap = np.load(npy_path).astype(np.float32)
    t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    patches = t.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size)
    patches = patches.permute(
        0, 2, 3, 1, 4, 5).reshape(-1, 1, patch_size, patch_size)
    return [p.squeeze(0).numpy() for p in patches]


def create_patches_and_save(
    dataset_dir: Path,
    img_dir_name: str,
    mask_dir_name: str,
    heatmap_dir_name: str,
    img_patches_dir_name: str,
    mask_patches_dir_name: str,
    heatmap_patches_dir_name: str,
    splits: list[str] = ["train", "validation", "test"],
):
    for split in splits:
        base_dir = dataset_dir / split
        if not base_dir.exists():
            raise ValueError(
                f"Base directory for split '{split}' does not exist: {base_dir}")

        img_patches_dir = base_dir / img_patches_dir_name
        mask_patches_dir = base_dir / mask_patches_dir_name
        heatmap_patches_dir = base_dir / heatmap_patches_dir_name

        init_dir(img_patches_dir)
        init_dir(mask_patches_dir)
        init_dir(heatmap_patches_dir)

        for in_dir, out_dir in [
            (base_dir / img_dir_name, img_patches_dir),
            (base_dir / mask_dir_name, mask_patches_dir),
        ]:
            if not in_dir.exists():
                raise ValueError(f"Input directory does not exist: {in_dir}")

            for png_file in in_dir.glob("*.png"):
                print(f"Cropping patches from image {png_file.name}")
                patches = create_patches_from_img(png_file, patch_size=1024)
                for i, patch in enumerate(patches):
                    F.to_pil_image(patch).save(
                        out_dir / f"{png_file.stem}_patch_{i:02d}.png")

        heatmaps_dir = base_dir / heatmap_dir_name
        if heatmaps_dir.exists():
            for npy_file in heatmaps_dir.glob("*.npy"):
                print(f"Cropping patches from heatmap {npy_file.name}")
                patches = create_patches_from_npy(npy_file, patch_size=1024)
                for i, patch in enumerate(patches):
                    np.save(heatmap_patches_dir /
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


def oversample_junction_patches(
    dataset_dir: Path,
    img_dir_name: str,
    mask_dir_name: str,
    heatmap_dir_name: str,
    img_patches_dir_name: str,
    mask_patches_dir_name: str,
    heatmap_patches_dir_name: str,
    oversample_count: int,
):
    """For each junction point in each augmented training image, create random
    1024x1024 patches containing the junction at a random (non-centered) position."""
    train_dir = dataset_dir / "train"
    highres_images_dir = train_dir / img_dir_name
    highres_masks_dir = train_dir / mask_dir_name
    highres_heatmaps_dir = train_dir / heatmap_dir_name
    img_patches_dir = train_dir / img_patches_dir_name
    mask_patches_dir = train_dir / mask_patches_dir_name
    heatmap_patches_dir = train_dir / heatmap_patches_dir_name

    patch_size = 1024
    total_patches = 0

    for npy_file in sorted(highres_heatmaps_dir.glob("*.npy")):
        stem = npy_file.stem
        heatmap = np.load(npy_file).astype(np.float32)
        centers = find_junction_centers(heatmap)
        if not centers:
            continue

        img_tensor = load_png_as_tensor(highres_images_dir / f"{stem}.png")
        mask_tensor = load_png_as_tensor(highres_masks_dir / f"{stem}.png")
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

            for k in range(oversample_count):
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
                np.save(heatmap_patches_dir /
                        f"{patch_name}.npy", hm_patch.squeeze(0).numpy())
                total_patches += 1

        print(f"Created {len(centers)} junction patches for {stem}.png")

    print(f"\nTotal junction oversampling patches created: {total_patches}")


def remove_highres_dirs(
    dataset_dir: Path,
    img_dir_name: str,
    mask_dir_name: str,
    heatmap_dir_name: str,
    splits: list[str] = ["train", "validation", "test"],
):
    for split in splits:
        base_dir = dataset_dir / split
        for dir_name in [img_dir_name, mask_dir_name, heatmap_dir_name]:
            d = base_dir / dir_name
            if d.exists():
                shutil.rmtree(d)
