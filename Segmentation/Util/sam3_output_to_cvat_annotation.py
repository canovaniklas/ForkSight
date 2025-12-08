import os
import numpy as np
from PIL import Image
import zipfile
from io import BytesIO
from pathlib import Path

RAW_DATA_DIR = "C:\\Users\\juhe9\\repos\\MasterThesis\\ForkSight\\Segmentation\\Data"

input_folder = Path(RAW_DATA_DIR) / "sam3_output"
output_zip_path = input_folder / "cvat_annotations.zip"

color_map = {
    0: (0, 0, 0),       # background
    255: (250, 50, 83)  # DNA
}

with zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:

    labelmap_content = """# label:color_rgb:parts:actions
DNA:250,50,83::
background:0,0,0::
"""
    zipf.writestr("labelmap.txt", labelmap_content)

    filenames = []

    for img_path in input_folder.glob("*.png"):
        img = Image.open(str(img_path))
        arr = np.array(img)

        rgb_arr = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        rgb_arr[arr == 0] = color_map[0]
        rgb_arr[arr == 255] = color_map[255]

        rgb_img = Image.fromarray(rgb_arr, mode='RGB')

        img_buffer = BytesIO()
        rgb_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        zipf.writestr(f"SegmentationClass/{img_path.name}", img_buffer.read())

        filenames.append(img_path.stem)
        print(f"Processed {img_path.name}")

    default_txt_content = "\n".join(filenames) + "\n"
    zipf.writestr("ImageSets/Segmentation/default.txt", default_txt_content)

print(f"ZIP file created: {output_zip_path}")
