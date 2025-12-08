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
CVAT_EXPORT_DIRS = ["20251208_segmentation_mask_1.1"]


def main():
    raw_data_dir = Path(RAW_DATA_DIR)
    cvat_dir = raw_data_dir / "cvat"
    output_folder = raw_data_dir / "masks_4096"

    for cvat_export_dir in CVAT_EXPORT_DIRS:
        input_folder = cvat_dir / cvat_export_dir

        if not input_folder.is_dir():
            print("Error: input folder")
            return

        if output_folder.exists():
            shutil.rmtree(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        for png_path in input_folder.rglob("SegmentationClass/*.png"):
            try:
                img = Image.open(png_path)
                img_array = np.array(img)

                binary_mask = np.all(img_array == CLASS_1_COLOR,
                                     axis=-1).astype(np.uint8)
                out_img = Image.fromarray(
                    binary_mask * 255 if BW_MASKS else binary_mask.astype(np.uint8))

                out_img.save(output_folder / png_path.name)

                print(
                    f"Converted: {png_path.relative_to(cvat_dir)} → {png_path.name}")

            except Exception as e:
                print(f"Failed to convert {png_path}: {e}")


if __name__ == "__main__":
    main()
