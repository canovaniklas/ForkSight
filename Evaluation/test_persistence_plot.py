from pathlib import Path
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from Evaluation.evaluation_util import (
    _save_sdt_persistence_plot,
    get_persistence_pairs,
    _save_raw_persistence_plot,
    get_sdt_persistence_pairs,
)

CSV_PATH = Path(__file__).resolve().parent / "persistencetest.csv"


def main():
    print(f"Reading: {CSV_PATH}")

    data = pd.read_csv(CSV_PATH, header=None).values
    print(f"  Shape after read: {data.shape}")

    if data.shape != (1024, 1024):
        raise ValueError(f"Expected 1024×1024 values, got shape {data.shape}")

    tensor = torch.tensor(data, dtype=torch.float32)
    print(f"  Value range: [{tensor.min():.4f}, {tensor.max():.4f}]")

    print("\nComputing persistence pairs...")
    b0_pairs, b1_pairs = get_persistence_pairs(tensor)
    print(f"  B0 pairs: {b0_pairs}")
    print(f"  B1 pairs: {b1_pairs}")

    pred = tensor > 0.5
    gt = tensor > 0.5
    pred_sdt_b0, pred_sdt_b1 = get_sdt_persistence_pairs(pred)
    gt_sdt_b0, gt_sdt_b1 = get_sdt_persistence_pairs(gt)
    print(f"  number of SDT B0 pairs: {len(pred_sdt_b0)}")
    print(f"  number of SDT B1 pairs: {len(pred_sdt_b1)}")

    out_path = CSV_PATH.with_name(CSV_PATH.stem + "_persistence_raw.png")
    print(f"\nSaving persistence figure to: {out_path}")
    _save_raw_persistence_plot(
        pred_b0=b0_pairs,
        pred_b1=b1_pairs,
        patch_name=CSV_PATH.stem,
        save_path=out_path,
    )

    out_path = CSV_PATH.with_name(CSV_PATH.stem + "_persistence_sdt.png")
    print(f"\nSaving persistence figure to: {out_path}")
    _save_sdt_persistence_plot(
        pred_b0=pred_sdt_b0,
        pred_b1=pred_sdt_b1,
        gt_b0=gt_sdt_b0,
        gt_b1=gt_sdt_b1,
        patch_name=CSV_PATH.stem,
        save_path=out_path,
    )

    thresholds = np.arange(0.1, 1.01, 0.1)
    arr = tensor.numpy()
    ncols = 5
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    for ax, t in zip(axes.flat, thresholds):
        ax.imshow(arr >= t, cmap="gray", vmin=0,
                  vmax=1, interpolation="nearest")
        ax.set_title(f"t = {t:.1f}")
        ax.axis("off")
    fig.suptitle(f"{CSV_PATH.stem} — threshold masks")
    fig.tight_layout()
    thresh_path = CSV_PATH.with_name(CSV_PATH.stem + "_thresholds.png")
    fig.savefig(thresh_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saving threshold figure to: {thresh_path}")
    print("Done.")


if __name__ == "__main__":
    main()
