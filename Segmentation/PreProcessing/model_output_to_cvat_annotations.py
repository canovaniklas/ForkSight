# creates a CVAT-compatible annotation ZIP from SAM3 output PNGs in the Segmentation Mask 1.1 format
# workflow:
#   create base annotations with sam3_zeroshot_segmentation or finetuned_sam_soi_segmentation (only for SoI patches) scripts
#   create ZIP file with this script
#   create a task in CVAT with two labels: "DNA" (250, 50, 83) and "background" (0, 0, 0)
#   upload the ZIP file as annotations using the Segmentation Mask 1.1 format
#   run actions -> shapes converter: polygons to masks
#   remove the background label from the task
#   review and adjust annotations as needed

import argparse
from datetime import datetime
import os
import numpy as np
from PIL import Image
import zipfile
from io import BytesIO
from pathlib import Path

from Segmentation.Util.env_utils import load_as_tuple, load_segmentation_env

load_segmentation_env()

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
SAM3_OUTPUT_DIR_NAME = os.getenv("SAM3_OUTPUT_DIR_NAME", "sam3_output")
FINETUNED_SAM_OUTPUT_DIR_NAME = os.getenv(
    "FINETUNED_SAM_OUTPUT_DIR_NAME", "finetuned_sam_output")

if not RAW_DATA_DIR:
    raise ValueError("RAW_DATA_DIR environment variable is not set.")

CVAT_BACKGROUND_COLOR = load_as_tuple("CVAT_BACKGROUND_COLOR", "0,0,0", int)
CVAT_MASK_COLOR = load_as_tuple("CVAT_MASK_COLOR", "250,50,83", int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="model used for segmentation", default="sam3")
    args = parser.parse_args()
    print(f"Generating CVAT annotations for model: {args.model}")

    if args.model == "sam3":
        output_dir = SAM3_OUTPUT_DIR_NAME
    elif args.model == "finetuned_sam":
        output_dir = FINETUNED_SAM_OUTPUT_DIR_NAME
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model_output_dir = Path(RAW_DATA_DIR) / output_dir
    output_zip_path = model_output_dir / \
        f"cvat_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

    color_map = {
        0: CVAT_BACKGROUND_COLOR,   # background
        255: CVAT_MASK_COLOR        # DNA
    }

    with zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:

        labelmap_content = f"""# label:color_rgb:parts:actions
    DNA:{','.join(map(str, CVAT_MASK_COLOR))}::
    background:{','.join(map(str, CVAT_BACKGROUND_COLOR))}::
    """
        zipf.writestr("labelmap.txt", labelmap_content)

        filenames = []
        for img_path in model_output_dir.glob("*.png"):
            img = Image.open(str(img_path))
            arr = np.array(img)

            rgb_arr = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
            rgb_arr[arr == 0] = color_map[0]
            rgb_arr[arr == 255] = color_map[255]

            rgb_img = Image.fromarray(rgb_arr, mode='RGB')

            img_buffer = BytesIO()
            rgb_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            zipf.writestr(
                f"SegmentationClass/{img_path.name}", img_buffer.read())

            filenames.append(img_path.stem)
            print(f"Processed {img_path.name}")

        default_txt_content = "\n".join(filenames) + "\n"
        zipf.writestr("ImageSets/Segmentation/default.txt",
                      default_txt_content)

    print(f"ZIP file created: {output_zip_path}")


if __name__ == "__main__":
    main()
