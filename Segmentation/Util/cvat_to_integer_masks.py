from pathlib import Path
from PIL import Image
import numpy as np
import shutil
import re

'''
This script converts CVAT (semantic) segmentation masks which have 3 channels and a specific RGB color for the foreground class
into binary integer masks. The background pixels are set to 0 and the foreground pixels to 1
'''

# RGB value of the mask class in the CVAT segmentation mask images
CLASS_1_COLOR = (250, 50, 83)
# Whether to save black and white masks (0 and 255) instead of integer masks (0 and 1)
BW_MASKS = True

RAW_DATA_DIR = "C:\\Users\\juhe9\\repos\\MasterThesis\\ForkSight\\Segmentation\\Data"


def main():
    raw_data_dir = Path(RAW_DATA_DIR)
    input_folder = raw_data_dir / "CvatSegmentationMasks"
    output_folder = raw_data_dir / "masks"

    if not input_folder.is_dir():
        print("Error: input folder")
        return

    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # for png_path in input_folder.rglob("*.png"):
    for png_path in input_folder.rglob("SegmentationClass/*.png"):
        try:
            img = Image.open(png_path)
            img_array = np.array(img)

            binary_mask = np.all(img_array == CLASS_1_COLOR,
                                 axis=-1).astype(np.uint8)
            out_img = Image.fromarray(
                binary_mask * 255 if BW_MASKS else binary_mask.astype(np.uint8))

            orig_folder_name = png_path.parent.parent.name
            orig_png_name = png_path.name
            new_name = re.sub(r"Site of interest\s*\((\d+)\)",
                              r"soi_\1", orig_png_name).lower()
            new_name = f"{orig_folder_name.lower()}_{new_name}"

            out_img.save(output_folder / new_name)

            print(
                f"Converted: {png_path.relative_to(raw_data_dir)} → {(output_folder / new_name).relative_to(raw_data_dir)}")

        except Exception as e:
            print(f"Failed to convert {png_path}: {e}")


if __name__ == "__main__":
    main()
