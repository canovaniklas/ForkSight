"""
Generate Gaussian heatmaps from CVAT 1.1 point annotations.

This script processes images, segmentation masks, and CVAT XML point annotations
to create weighted heatmaps centered on annotated points. The heatmaps use a
Gaussian weighting function and can be used for loss function weighting.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from Environment.env_utils import load_as, load_forksight_env
from Segmentation.PreProcessing.dataset_util import parse_junction_annotations_xml

load_forksight_env()

SEED = load_as("SEED", int, 42)
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", None)
HIGHRES_IMG_DIR_NAME = os.getenv("HIGHRES_IMG_DIR_NAME", "images_4096")
HIGHRES_MASK_DIR_NAME = os.getenv("HIGHRES_MASK_DIR_NAME", "masks_4096")

HIGHRES_HEATMAP_DIR_NAME = os.getenv(
    "HIGHRES_HEATMAP_DIR_NAME", "heatmaps_4096")
HEATMAP_VISUALIZATION_DIR_NAME = os.getenv(
    "HEATMAP_VISUALIZATION_DIR_NAME", "heatmap_visualizations")

DATASET_JUNCTION_COORDS_CVAT_XML_PATH = os.getenv(
    "DATASET_JUNCTION_COORDS_CVAT_XML_PATH", None)
DATASET_JUNCTION_WEIGHT_SIGMA = load_as(
    "DATASET_JUNCTION_WEIGHT_SIGMA", float, 30.0)
DATASET_JUNCTION_WEIGHT_CLIP_THRESHOLD = load_as(
    "DATASET_JUNCTION_WEIGHT_CLIP_THRESHOLD", float, 0.1)
DATASET_JUNCTION_WEIGHT_RADIUS_MULTIPLIER = load_as(
    "DATASET_JUNCTION_WEIGHT_RADIUS_MULTIPLIER", float, 3.0)

if RAW_DATA_DIR is None or DATASET_JUNCTION_COORDS_CVAT_XML_PATH is None:
    raise ValueError(
        "RAW_DATA_DIR and DATASET_JUNCTION_COORDS_CVAT_XML_PATH environment variables must be set")

IMAGE_DIR = Path(RAW_DATA_DIR) / HIGHRES_IMG_DIR_NAME
MASK_DIR = Path(RAW_DATA_DIR) / HIGHRES_MASK_DIR_NAME
OUTPUT_DIR = Path(RAW_DATA_DIR) / HIGHRES_HEATMAP_DIR_NAME
VISUALIZATION_DIR = Path(RAW_DATA_DIR) / HEATMAP_VISUALIZATION_DIR_NAME

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)
if VISUALIZATION_DIR.exists():
    shutil.rmtree(VISUALIZATION_DIR)
VISUALIZATION_DIR.mkdir(parents=True)


def create_gaussian_heatmap(image_shape: Tuple[int, int], points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Create a Gaussian heatmap from a list of points.
    The Gaussian function is: Y(u,v) = exp(-((u-x)^2 + (v-y)^2) / (2*SIGMA^2))
    """
    height, width = image_shape
    heatmap = np.zeros((height, width), dtype=np.float32)

    radius = int(np.ceil(
        DATASET_JUNCTION_WEIGHT_RADIUS_MULTIPLIER * DATASET_JUNCTION_WEIGHT_SIGMA))

    for x, y in points:
        x_int, y_int = int(round(x)), int(round(y))

        x_min = max(0, x_int - radius)
        x_max = min(width, x_int + radius + 1)
        y_min = max(0, y_int - radius)
        y_max = min(height, y_int + radius + 1)

        u = np.arange(x_min, x_max)
        v = np.arange(y_min, y_max)
        uu, vv = np.meshgrid(u, v)

        gaussian = np.exp(-((uu - x)**2 + (vv - y)**2) /
                          (2 * DATASET_JUNCTION_WEIGHT_SIGMA**2))

        gaussian[gaussian < DATASET_JUNCTION_WEIGHT_CLIP_THRESHOLD] = 0

        heatmap[y_min:y_max, x_min:x_max] = np.maximum(
            heatmap[y_min:y_max, x_min:x_max],
            gaussian
        )

    return heatmap


def visualize_heatmap_overlay(
    image_path: Path,
    mask_path: Path,
    heatmap: np.ndarray,
    output_path: Path
):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path)
    print(f"Visualizing heatmap overlay for: {image_path.name}")
    print(
        f"  Image size: {image.size}, Mask size: {mask.size}, Heatmap shape: {heatmap.shape}")

    image_np = np.array(image)
    mask_np = np.array(mask)
    print(
        f"  Image array shape: {image_np.shape}, Mask array shape: {mask_np.shape}")

    _, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.imshow(image_np)

    mask_overlay = np.zeros((*mask_np.shape, 4))
    mask_overlay[mask_np > 0] = [0.0, 1.0, 1.0, 0.6]
    ax.imshow(mask_overlay)

    im = ax.imshow(heatmap, cmap='hot', alpha=0.5,
                   interpolation='bilinear', vmin=0, vmax=1)

    ax.axis('off')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()


def process_images():
    print(
        f"Loading CVAT annotations from: {DATASET_JUNCTION_COORDS_CVAT_XML_PATH}")

    points_per_image = parse_junction_annotations_xml(
        DATASET_JUNCTION_COORDS_CVAT_XML_PATH)
    print(f"Found {len(points_per_image)} annotated images in XML")

    # Get all images in the image directory
    image_files = sorted(IMAGE_DIR.glob('*.png'))
    print(f"Found {len(image_files)} images in directory")

    for idx, image_path in enumerate(image_files):
        image_name = image_path.name

        points = points_per_image.get(image_name, [])
        print(f"  Points: {len(points)}")

        with Image.open(image_path) as img:
            width, height = img.size

        print(
            f"\nProcessing {idx}/{len(image_files)}: {image_name} with dimencions (width={width}, height={height})")

        if len(points) > 0:
            heatmap = create_gaussian_heatmap(
                image_shape=(height, width), points=points)
        else:
            # Create zero heatmap if no points annotations found
            heatmap = np.zeros((height, width), dtype=np.float32)

        print(f"heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")
        num_nonzero_pixels = np.sum(heatmap > 0)
        print(f"  Heatmap non-zero pixels: {num_nonzero_pixels}")
        px_mean = heatmap.mean()
        print(f"  Heatmap mean value: {px_mean:.6f}")
        px_over_zero_mean = heatmap[heatmap > 0].mean(
        ) if num_nonzero_pixels > 0 else 0.0
        print(
            f"  Heatmap mean value (non-zero pixels): {px_over_zero_mean:.6f}")
        heatmap_max = heatmap.max()
        print(f"  Heatmap max value: {heatmap_max:.6f}")
        num_max_pixels = np.sum(heatmap == heatmap_max)
        print(f"  Heatmap max pixels: {num_max_pixels}")

        heatmap_filename = Path(image_name).stem + '.npy'
        heatmap_path = OUTPUT_DIR / heatmap_filename
        np.save(heatmap_path, heatmap)

        viz_filename = Path(image_name).stem + '_heatmap_visualization.png'
        viz_path = VISUALIZATION_DIR / viz_filename
        visualize_heatmap_overlay(
            image_path=image_path,
            mask_path=MASK_DIR / image_name,
            heatmap=heatmap,
            output_path=viz_path
        )


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    process_images()
