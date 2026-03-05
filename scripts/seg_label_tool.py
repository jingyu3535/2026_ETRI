#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="Manual polygon mask labeling tool (single-object).")
    ap.add_argument("--input_dir", required=True, help="Directory with seed frames (pngs).")
    ap.add_argument("--output_dir", required=True, help="Directory to save masks (pngs).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing masks.")
    ap.add_argument("--start", type=int, default=0, help="Start index in sorted file list.")
    ap.add_argument("--end", type=int, default=-1, help="End index (exclusive). -1 = all.")
    return ap.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.png"))
    if args.end >= 0:
        files = files[args.start:args.end]
    else:
        files = files[args.start:]

    if not files:
        print("No png files found.")
        return

    idx = 0
    points = []
    polygons = []

    def on_mouse(event, x, y, flags, param):
        nonlocal points, polygons
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("label", 1280, 960)
    cv2.setMouseCallback("label", on_mouse)

    while idx < len(files):
        img_path = files[idx]
        out_path = out_dir / img_path.name
        if out_path.exists() and not args.overwrite:
            idx += 1
            points = []
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read {img_path}")
            idx += 1
            points = []
            continue

        while True:
            disp = img.copy()
            if points:
                # Draw polygon lines
                for i in range(len(points) - 1):
                    cv2.line(disp, points[i], points[i + 1], (0, 255, 0), 2)
                # Draw points
                for p in points:
                    cv2.circle(disp, p, 3, (0, 0, 255), -1)
            # Draw completed polygons
            for poly in polygons:
                if len(poly) >= 3:
                    cv2.polylines(disp, [np.array(poly, dtype=np.int32)], True, (255, 0, 0), 2)

            cv2.putText(
                disp,
                f"{idx+1}/{len(files)} {img_path.name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("label", disp)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("r"):
                points = []
            elif key == ord("a"):
                # Add current polygon and start a new one
                if len(points) >= 3:
                    polygons.append(points)
                points = []
            elif key == ord("s"):
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                polys = []
                if len(points) >= 3:
                    polys.append(np.array(points, dtype=np.int32))
                for poly in polygons:
                    if len(poly) >= 3:
                        polys.append(np.array(poly, dtype=np.int32))
                if polys:
                    cv2.fillPoly(mask, polys, 255)
                cv2.imwrite(str(out_path), mask)
                points = []
                polygons = []
                idx += 1
                break
            elif key == ord("n"):
                # Save empty mask
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.imwrite(str(out_path), mask)
                points = []
                polygons = []
                idx += 1
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
