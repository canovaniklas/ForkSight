from pathlib import Path
import shutil
import torchvision.transforms.functional as F
from PIL import Image


base_dirs = [
    "/data/jhehli/datasets/SAM_LoRA_Augmented/train/",
    "/data/jhehli/datasets/SAM_LoRA_Augmented/test/",
]


def init_folder(folder_path: Path):
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


def crop_image_to_lowres(input_image_path: Path, output_dir: Path):
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
    images_dir = Path(base_dir) / "images"
    masks_dir = Path(base_dir) / "masks"
    lowres_images_dir = Path(base_dir) / "lowres_images"
    lowres_masks_dir = Path(base_dir) / "lowres_masks"

    if not images_dir.exists() or not masks_dir.exists():
        print(
            f"Skipping {base_dir} as it does not contain images/ and masks/ directories")
        continue

    init_folder(lowres_images_dir)
    init_folder(lowres_masks_dir)

    for png_file in images_dir.glob("*.png"):
        crop_image_to_lowres(png_file, lowres_images_dir)

    for png_file in masks_dir.glob("*.png"):
        crop_image_to_lowres(png_file, lowres_masks_dir)
