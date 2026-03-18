"""
Visualize nnUNet segmentation mask predictions overlaid on their original input images.

Usage:
    python visualize_nnunet_predictions.py \
        --images-dir /path/to/original/images \
        --masks-dir /path/to/nnunet/predictions \
        --output-dir /path/to/output \
        --mapping-json /path/to/mapping.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


MASK_COLOR = np.array([0.0, 1.0, 1.0, 0.6])


def load_image(path: Path) -> np.ndarray:
    """Load an image as an RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_mask(path: Path) -> np.ndarray:
    """Load a PNG segmentation mask as a 2D binary numpy array (0 or 1)."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def overlay_mask(ax, mask: np.ndarray, color: np.ndarray = MASK_COLOR):
    """Overlay a binary mask on an axes using the given RGBA color."""
    h, w = mask.shape
    mask_rgba = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_rgba)


def make_output_filename(original_filename: str, patch: str | None) -> str:
    if patch:
        return f"{original_filename}_patch{str(patch).zfill(2)}"
    return original_filename


def make_title(nnunet_case: str, original_filename: str, patch: str | None) -> str:
    original_part = f"{original_filename}_{patch}" if patch else original_filename
    return f"nnUNet: {nnunet_case}  |  Original: {original_part}"


def process_entry(
    entry: dict,
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
):
    original_filename = entry["original_filename"]
    patch = entry.get("patch") or None
    nnunet_case = entry["nnunet_case"]

    img_path = images_dir / f"{nnunet_case}_0000.png"
    if not img_path.exists():
        print(
            f"  [WARN] Original image not found for case '{nnunet_case}' in {images_dir}, skipping.")
        return

    # Locate predicted mask by nnunet_case stem
    mask_path = masks_dir / f"{nnunet_case}.png"
    if not mask_path.exists():
        print(
            f"  [WARN] Predicted mask not found for case '{nnunet_case}' in {masks_dir}, skipping.")
        return

    image = load_image(img_path)
    mask = load_mask(mask_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    overlay_mask(ax, mask)
    ax.axis("off")
    ax.set_title(make_title(
        nnunet_case, original_filename, patch), fontsize=10)

    output_stem = make_output_filename(original_filename, patch)
    output_path = output_dir / f"{output_stem}.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Save plots of nnUNet segmentation masks overlaid on their original images."
    )
    parser.add_argument(
        "--images-dir", required=True, type=Path,
        help="Directory containing the original input images."
    )
    parser.add_argument(
        "--masks-dir", required=True, type=Path,
        help="Directory containing nnUNet predicted mask files."
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory where output plots will be saved."
    )
    parser.add_argument(
        "--mapping-json", required=True, type=Path,
        help=(
            "JSON file with a list of dicts mapping nnUNet cases to original images. "
            'Expected keys: "original_filename", "patch", "nnunet_case", "split".'
        ),
    )
    args = parser.parse_args()

    for d in (args.images_dir, args.masks_dir):
        if not d.exists():
            print(f"Error: directory does not exist: {d}")
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.mapping_json, "r") as f:
        mapping = json.load(f)

    mapping = [e for e in mapping if e.get("split") == "test"]
    print(f"Processing {len(mapping)} entries...")

    for entry in mapping:
        process_entry(entry, args.images_dir, args.masks_dir, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
