"""Count raw images and junction points from heatmaps."""

import os
from pathlib import Path
import numpy as np
from scipy.ndimage import label

from Environment.env_utils import load_forksight_env

load_forksight_env()

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
HIGHRES_IMG_DIR_NAME = os.getenv("HIGHRES_IMG_DIR_NAME", "images_4096")
HIGHRES_HEATMAP_DIR_NAME = os.getenv(
    "HIGHRES_HEATMAP_DIR_NAME", "heatmaps_4096")

image_dir = Path(RAW_DATA_DIR) / HIGHRES_IMG_DIR_NAME
heatmap_dir = Path(RAW_DATA_DIR) / HIGHRES_HEATMAP_DIR_NAME

total_junctions = 0
image_files = sorted(image_dir.glob("*.png"))

for img_path in image_files:
    heatmap_path = heatmap_dir / f"{img_path.stem}.npy"
    heatmap = np.load(heatmap_path).astype(np.float32)
    _, num_junctions = label(heatmap >= 0.95)
    total_junctions += num_junctions
    print(f"  {img_path.name}: {num_junctions} junctions")

print(f"\nTotal images: {len(image_files)}")
print(f"Total junctions: {total_junctions}")
print(
    f"Mean junctions per image: {total_junctions / max(len(image_files), 1):.1f}")
