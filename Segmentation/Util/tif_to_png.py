from pathlib import Path
from PIL import Image
import shutil
import numpy as np

DATASETS_DIR = "C:\\Users\\juhe9\\repos\\MasterThesis\\ForkSight\\Segmentation\\Data"
IN_IMAGES = [
    "Z:\\imcrdata\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R1\\20240523_Andrea_Orange\\LayersData\\highmag\\Tile Set (17)\\Tile_016-003-000000_0-000.tif"
]


def init_folder(folder_path: Path):
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


def get_exp_dir_name(img_path: Path) -> str:
    parts = img_path.parts
    idx = parts.index("LayersData")
    return parts[idx - 1].lower()


def get_tileset_tile_name(img_path: Path) -> str:
    tileset = img_path.parent.name
    tileset = tileset.replace("Tile Set ", "tileset_").replace(
        " ", "_").replace("(", "").replace(")", "")
    tile = img_path.stem.replace("-000000_0-000", "").replace("-", "_")
    return f"{tileset}_{tile}".lower()


def get_new_name(img_path: Path) -> str:
    exp_dir_name = get_exp_dir_name(img_path)
    tileset_tile_name = get_tileset_tile_name(img_path)
    return f"{exp_dir_name}_{tileset_tile_name}.png"


def save_image_as_png(img: Image.Image, out_dir: Path, out_name: str, resize: tuple = None):
    if resize is not None:
        img = img.resize(resize, Image.Resampling.BILINEAR)
    out_path = out_dir / out_name
    img.save(out_path, format="PNG")


def normalize_convert_uint8(img: Image.Image) -> Image.Image:
    img_arr = np.array(img).astype(np.float32)

    p_low, p_high = np.percentile(img_arr, (1, 99))
    p_low, p_high = float(p_low), float(p_high)
    img_arr = (img_arr - p_low) * (255.0 / (p_high - p_low))
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(img_arr)


def main():
    out_dir_path = Path(DATASETS_DIR) / "images_4096"
    img_paths = [Path(p) for p in IN_IMAGES]

    init_folder(out_dir_path)

    for img_path in img_paths:
        try:
            img = Image.open(img_path)
            img = normalize_convert_uint8(img)

            new_name = get_new_name(img_path)
            save_image_as_png(img, out_dir_path, new_name)

            print(f"{new_name}")

        except Exception as e:
            print(f"Failed to convert {img_path}: {e}")


if __name__ == "__main__":
    main()
