from pathlib import Path
from PIL import Image
import numpy as np
import shutil

'''
nn-UNet requires integer masks for segmentation tasks (0=background, consecutive classes 1, 2, ...), 
see https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

This script converts CVAT segmentation masks which have 3 channels and a specific RGB color for the foreground class
into binary integer masks suitable for nn-UNet. The background is set to 0 and the foreground pixels to 1
'''

# RGB value of the mask class in the CVAT segmentation mask images
CLASS_1_COLOR = (250, 50, 83)


def main():
    script_dir = Path(__file__).resolve().parent
    data_folder = script_dir / "Data"
    input_folder = data_folder / "CvatSegmentationMasks"
    output_folder = data_folder / "BinarySegmentationMasks"
    output_folder_bw = data_folder / "BinarySegmentationMasksBlackWhite"

    if not input_folder.is_dir() or not output_folder.is_dir():
        print("Error: input folder or output folder not found")
        return

    if output_folder.exists():
        shutil.rmtree(output_folder)
    if output_folder_bw.exists():
        shutil.rmtree(output_folder_bw)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_folder_bw.mkdir(parents=True, exist_ok=True)

    for png_path in input_folder.rglob("*.png"):
        try:
            img = Image.open(png_path)
            img_array = np.array(img)

            binary_mask = np.all(img_array == CLASS_1_COLOR,
                                 axis=-1).astype(np.uint8)
            out_img = Image.fromarray(binary_mask.astype(np.uint8))
            out_img_bw = Image.fromarray(binary_mask * 255)

            relative_path = png_path.relative_to(input_folder).parent
            target_folder = output_folder / relative_path
            target_folder.mkdir(parents=True, exist_ok=True)
            out_path = target_folder / png_path.name
            out_img.save(out_path)

            target_folder_bw = output_folder_bw / relative_path
            target_folder_bw.mkdir(parents=True, exist_ok=True)
            out_path_bw = target_folder_bw / png_path.name
            out_img_bw.save(out_path_bw)

            print(
                f"Converted: {png_path.relative_to(data_folder)} → {out_path.relative_to(data_folder)}")

        except Exception as e:
            print(f"Failed to convert {png_path}: {e}")


if __name__ == "__main__":
    main()
