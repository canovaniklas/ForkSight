import os
from pathlib import Path
from PIL import Image
import numpy as np
import shutil

from Environment.env_utils import load_as_bool, load_as_tuple, load_forksight_env

load_forksight_env()

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
CVAT_DIR_NAME = os.getenv("CVAT_DIR_NAME", "cvat")
HIGHRES_MASK_DIR_NAME = os.getenv("HIGHRES_MASK_DIR_NAME", "masks_4096")

CVAT_MASK_COLOR = load_as_tuple("CVAT_MASK_COLOR", "250,50,83", int)
CVAT_GENERATE_BW_MASKS = load_as_bool("CVAT_GENERATE_BW_MASKS", True)

if RAW_DATA_DIR is None:
    raise ValueError("RAW_DATA_DIR environment variable is not set")


def main():
    raw_data_dir = Path(RAW_DATA_DIR)
    cvat_dir = raw_data_dir / CVAT_DIR_NAME
    output_dir = raw_data_dir / HIGHRES_MASK_DIR_NAME

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_dir in [p for p in cvat_dir.iterdir() if p.is_dir()]:
        for png_path in input_dir.rglob("SegmentationClass/*.png"):
            try:
                img = Image.open(png_path)
                img_array = np.array(img)

                binary_mask = np.all(img_array == CVAT_MASK_COLOR,
                                     axis=-1).astype(np.uint8)
                out_img = Image.fromarray(
                    binary_mask * 255 if CVAT_GENERATE_BW_MASKS else binary_mask.astype(np.uint8))

                if not png_path.stem.endswith("_soi"):
                    # ensure all full images are same size (some are 4000x400)
                    out_img = out_img.resize(
                        (4096, 4096), resample=Image.Resampling.NEAREST)

                out_img.save(output_dir / png_path.name)

                print(
                    f"Converted: {png_path.relative_to(cvat_dir)} → {png_path.name}")

            except Exception as e:
                print(f"Failed to convert {png_path}: {e}")


if __name__ == "__main__":
    main()
