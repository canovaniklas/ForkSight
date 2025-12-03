from pathlib import Path
from PIL import Image
import shutil
import re

'''
Converts all .tif images in 'Data/TifImages' to .png format and saves them in 'Data/PngImages', preserving the directory structure.
The TIF images are assumed to have 1 channel, and the PNG images will also have 1 channel.
'''

RAW_DATA_DIR = "C:\\Users\\juhe9\\repos\\MasterThesis\\ForkSight\\Segmentation\\Data"


def init_folder(folder_path: Path):
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


def save_image_as_png(img: Image.Image, out_dir: Path, out_name: str, resize: tuple = None):
    if resize is not None:
        img = img.resize(resize, Image.LANCZOS)
    out_path = out_dir / out_name
    img.save(out_path, format="PNG")


def main():
    raw_data_dir = Path(RAW_DATA_DIR)
    tif_folder = raw_data_dir / "TifImages"
    png_folder = raw_data_dir / "images"
    lowres_png_folder = raw_data_dir / "images_lowres"

    if not tif_folder.is_dir():
        print("Error: inputfolder not found")
        return

    init_folder(png_folder)
    init_folder(lowres_png_folder)

    for tif_path in tif_folder.rglob("*.tif"):
        try:
            img = Image.open(tif_path)

            orig_folder_name = tif_path.parent.name
            orig_tif_name = tif_path.name
            new_name = re.sub(r"Site of interest\s*\((\d+)\)",
                              r"soi_\1", orig_tif_name).lower()
            new_name = f"{orig_folder_name.lower()}_{new_name}".replace(
                ".tif", ".png")

            save_image_as_png(img, png_folder, new_name)
            save_image_as_png(img, lowres_png_folder,
                              new_name, resize=(256, 256))
            print(
                f"Converted: {tif_path.relative_to(tif_folder)} → {new_name}")

        except Exception as e:
            print(f"Failed to convert {tif_path}: {e}")


if __name__ == "__main__":
    main()
