"""
Merge point annotations from final re-labeling into one CSV.

Output is written to $JUNCTION_DETECTION_DIR/$JUNCTION_DETECTION_RELABELING_DATA/merged_annotations.csv

Usage:
    python merge_final_relabeling_annotations.py --json Final.json --csv agreed.csv
"""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import unquote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import Environment.env_utils as env_utils


def strip_float_suffix(label: str) -> str:
    """Remove trailing float suffix from a label, e.g. 'Normal Fork 0.67' -> 'Normal Fork'."""
    return re.sub(r"\s+\d+(\.\d+)?\s*$", "", label).strip()


def load_final_json(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)

    rows = []
    for entry in data:
        image_path = entry.get("data", {}).get("image", "")
        image_name = Path(unquote(image_path)).stem.replace(
            "-000000_0-000", "")

        for annotation in entry.get("annotations", []):
            for result in annotation.get("result", []):
                value = result.get("value", {})
                x = value.get("x")
                y = value.get("y")
                keypointlabels = value.get("keypointlabels", [])
                if not keypointlabels:
                    continue
                label = strip_float_suffix(keypointlabels[0])
                if label == "Unsure":
                    label = "Negative"
                    x = y = None
                rows.append({
                    "image": image_name,
                    "label": label,
                    "x": x,
                    "y": y,
                    "source": "json",
                })

    return pd.DataFrame(rows, columns=["image", "label", "x", "y", "source"])


def load_agreed_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["image"] = df["image"].apply(lambda i: i.replace(
        "-000000_0-000", "").replace(".png", ""))
    df["label"] = df["label"].apply(strip_float_suffix)
    result = df[["image", "label", "mean_x_px", "mean_y_px"]].copy()
    result.columns = ["image", "label", "x", "y"]
    result["source"] = "csv"
    return result


def normalize_image_filenames(junction_detection_dir: Path):
    """Remove '-000000_0-000' suffix from image filenames in images/ in-place."""
    image_dir = junction_detection_dir / "images"
    suffix = "-000000_0-000"
    renamed = 0
    for img_path in image_dir.iterdir():
        if suffix in img_path.stem:
            new_name = img_path.stem.replace(suffix, "") + img_path.suffix
            img_path.rename(img_path.parent / new_name)
            renamed += 1
    if renamed:
        print(f"Renamed {renamed} image files (removed '{suffix}')")


def _get_image_points(df: pd.DataFrame, image_name: str):
    """Return (has_any_annotation, point_rows) for an image name (stem, suffix stripped)."""
    rows = df[df["image"] == image_name]
    points = rows[rows["label"] != "Negative"].dropna(subset=["x", "y"])
    return rows, points


def move_excluded(df: pd.DataFrame, junction_detection_dir: Path):
    """Move images with no annotations at all to images_excluded/. Negative-labelled images stay."""
    image_dir = junction_detection_dir / "images"
    excluded_dir = junction_detection_dir / "images_excluded"
    excluded_dir.mkdir(exist_ok=True)

    unannotated = []
    for img_path in sorted(image_dir.iterdir()):
        rows, points = _get_image_points(df, img_path.stem)
        if rows.empty or points.empty and not (rows["label"] == "Negative").any():
            unannotated.append(img_path.stem)
            img_path.rename(excluded_dir / img_path.name)

    if unannotated:
        print(
            f"\nMoved {len(unannotated)} unannotated images to {excluded_dir}:")
        for name in unannotated:
            print(f"  {name}")
    else:
        print("No unannotated images found.")


def plot_junctions(df: pd.DataFrame, junction_detection_dir: Path):
    """For each image in images/, save a plot with annotated junction points."""
    image_dir = junction_detection_dir / "images"
    plots_dir = junction_detection_dir / "junction_plots"
    plots_dir.mkdir(exist_ok=True)

    image_files = sorted(image_dir.iterdir())
    no_points_images = []

    for img_path in image_files:
        rows, points = _get_image_points(df, img_path.stem)

        if points.empty:
            no_points_images.append(img_path.stem)

        pil_img = Image.open(img_path)
        orig_w, orig_h = pil_img.size
        pil_img = pil_img.resize((1024, 1024), Image.LANCZOS)
        img = np.array(pil_img)
        scale_x, scale_y = 1024 / orig_w, 1024 / orig_h

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

        for _, row in points.iterrows():
            if row["source"] == "json":
                px, py = row["x"] * 1024 / 100, row["y"] * 1024 / 100
            else:
                px, py = row["x"] * scale_x, row["y"] * scale_y
            ax.plot(px, py, "o", color="none", markersize=40, markeredgewidth=3,
                    markeredgecolor="red")

        ax.set_title(img_path.stem, fontsize=8)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / (img_path.stem + ".png"), dpi=150)
        plt.close(fig)

    if no_points_images:
        print(f"\nImages with no plotted points ({len(no_points_images)}):")
        for name in no_points_images:
            print(f"  {name}")

    n_with_points = len(image_files) - len(no_points_images)
    print(
        f"\nSaved {len(image_files)} plots to {plots_dir} ({n_with_points} with points)")


def plot_label_stats(df: pd.DataFrame, junction_detection_dir: Path):
    """Bar plot: images per label + images with multiple annotations / mixed label types."""
    plots_dir = junction_detection_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Count distinct images per label
    per_label = (
        df.groupby("label")["image"]
        .nunique()
        .sort_values(ascending=False)
    )

    # Images with more than one annotation row
    annotation_counts = df.groupby("image").size()
    n_multi_annot = (annotation_counts > 1).sum()

    # Images with more than one distinct label type
    label_counts = df.groupby("image")["label"].nunique()
    n_mixed_labels = (label_counts > 1).sum()

    labels = list(per_label.index) + ["Multiple annotations", "Mixed label types"]
    counts = list(per_label.values) + [n_multi_annot, n_mixed_labels]
    colors = ["steelblue"] * len(per_label) + ["orange", "tomato"]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(labels, counts, color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Number of images")
    ax.set_title("Images per annotation label")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    out = plots_dir / "label_stats.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved label stats plot to {out}")


def main():
    env_utils.load_forksight_env()

    RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", None)
    JUNCTION_DETECTION_DIR_NAME = os.getenv(
        "JUNCTION_DETECTION_DIR_NAME", None)
    JUNCTION_DETECTION_RELABELING_FILE_NAME = os.getenv(
        "JUNCTION_DETECTION_RELABELING_FILE_NAME", None)

    if not all([RAW_DATA_DIR, JUNCTION_DETECTION_DIR_NAME, JUNCTION_DETECTION_RELABELING_FILE_NAME]):
        raise ValueError(
            "One or more required environment variables are not set: RAW_DATA_DIR, JUNCTION_DETECTION_DIR, JUNCTION_DETECTION_RELABELING_DATA")

    parser = argparse.ArgumentParser(
        description="Merge Final.json and agreed.csv annotations.")
    parser.add_argument("--json", required=True, help="Path to Final.json")
    parser.add_argument("--csv", required=True, help="Path to agreed.csv")
    parser.add_argument("--move-excluded", action="store_true", default=False,
                        help="Detect, list, and move images with no annotations to images_excluded/")
    parser.add_argument("--plot-junctions", action="store_true", default=False,
                        help="Save per-image junction plots to junction_plots/ (images must be in images/)")
    parser.add_argument("--plot-stats", action="store_true", default=False,
                        help="Save label distribution bar plot to plots/")
    parser.add_argument("--force-recompute", action="store_true", default=False,
                        help="Recompute and overwrite the output CSV even if it already exists")
    args = parser.parse_args()

    junction_detection_dir = Path(RAW_DATA_DIR) / JUNCTION_DETECTION_DIR_NAME

    normalize_image_filenames(junction_detection_dir)

    output_path = junction_detection_dir / JUNCTION_DETECTION_RELABELING_FILE_NAME
    if output_path.exists() and not args.force_recompute:
        print(
            f"Loading existing CSV from {output_path} (use --force-recompute to regenerate)")
        merged = pd.read_csv(output_path)
    else:
        df_json = load_final_json(args.json)
        df_csv = load_agreed_csv(args.csv)
        print(f"JSON points: {len(df_json)}, CSV points: {len(df_csv)}")
        merged = pd.concat([df_json, df_csv], ignore_index=True)
        merged.to_csv(output_path, index=False)
        print(f"Wrote {len(merged)} rows to {output_path}")

    if args.move_excluded:
        move_excluded(merged, junction_detection_dir)

    if args.plot_junctions:
        plot_junctions(merged, junction_detection_dir)

    if args.plot_stats:
        plot_label_stats(merged, junction_detection_dir)


if __name__ == "__main__":
    main()
