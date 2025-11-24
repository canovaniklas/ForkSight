from pathlib import Path
import tifffile
from PIL import Image
import numpy as np

script_dir = Path(__file__).resolve().parent.parent
tif_dir = script_dir / "Data" / "TifImages"
png_dir = script_dir / "Data" / "PngImages"
nnUNet_dataset_dir = script_dir / "Data" / \
    "Dataset001_TestCvatAugmented" / "imagesTr"

for tif_path in tif_dir.rglob("*.tif"):
    im = tifffile.imread(tif_path)
    print(tif_path, im.shape)

print()
for png_path in png_dir.rglob("*.png"):
    img = Image.open(png_path)
    arr = np.array(img)
    print(png_path, arr.shape, img.mode)