#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute attention mass inside a ROI.")
    parser.add_argument("--dir", required=True, help="Directory with img_XXXX.png and attn_XXXX.npy")
    parser.add_argument("--frame", required=True, help="Frame index, e.g. 0005")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.dir)
    img_path = base / f"img_{args.frame}.png"
    attn_path = base / f"attn_{args.frame}.npy"

    if not img_path.exists() or not attn_path.exists():
        raise SystemExit(f"Missing {img_path} or {attn_path}")

    img = plt.imread(img_path)
    attn = np.load(attn_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(img)
    ax.set_title("Click two corners for ROI (top-left, bottom-right)")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(pts) != 2:
        raise SystemExit("ROI selection cancelled.")

    (x1, y1), (x2, y2) = pts
    x1, x2 = int(min(x1, x2)), int(max(x1, x2))
    y1, y2 = int(min(y1, y2)), int(max(y1, y2))

    roi = attn[y1:y2, x1:x2]
    total = float(attn.sum())
    roi_sum = float(roi.sum())
    roi_mean = float(roi.mean()) if roi.size > 0 else 0.0

    print(f"ROI sum: {roi_sum:.6f}")
    print(f"Total sum: {total:.6f}")
    print(f"ROI share: {(roi_sum / total * 100) if total > 0 else 0.0:.2f}%")
    print(f"ROI mean: {roi_mean:.6f}")


if __name__ == "__main__":
    main()
