from pathlib import Path
from PIL import Image
import shutil

'''
Converts all .tif images in 'Data/TifImages' to .png format and saves them in 'Data/PngImages', preserving the directory structure.
The TIF images are assumed to have 1 channel, and the PNG images will also have 1 channel.
'''


def main():
    script_dir = Path(__file__).resolve().parent
    data_folder = script_dir / "Data"
    tif_folder = data_folder / "TifImages"
    png_folder = data_folder / "PngImages"

    if not tif_folder.is_dir() or not png_folder.is_dir():
        print("Error: input or output folder not found")
        return

    if png_folder.exists():
        shutil.rmtree(png_folder)
    png_folder.mkdir(parents=True, exist_ok=True)

    for tif_path in tif_folder.rglob("*.tif"):
        try:
            relative_path = tif_path.relative_to(tif_folder).parent
            target_folder = png_folder / relative_path
            target_folder.mkdir(parents=True, exist_ok=True)

            img = Image.open(tif_path)

            out_path = target_folder / (tif_path.stem + ".png")
            img.save(out_path, format="PNG")
            print(
                f"Converted: {tif_path.relative_to(data_folder)} → {out_path.relative_to(data_folder)}")

        except Exception as e:
            print(f"Failed to convert {tif_path}: {e}")


if __name__ == "__main__":
    main()
