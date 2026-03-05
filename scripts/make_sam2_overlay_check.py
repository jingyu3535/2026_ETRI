#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Create per-frame overlay images from frames_by_ep and SAM2 masks."
    )
    p.add_argument("--frames_root", required=True, help="Root path of frames_by_ep.")
    p.add_argument("--masks_root", required=True, help="Root path of SAM2 masks.")
    p.add_argument("--out_root", required=True, help="Root output directory.")
    p.add_argument("--cameras", default="camera1,camera2", help="Comma-separated camera keys.")
    p.add_argument("--episodes", default="0-35", help="Episode range like 0-35 or list 0,1,2.")
    p.add_argument("--alpha", type=float, default=0.22, help="Overlay opacity for mask region.")
    p.add_argument(
        "--mask_color",
        default="0,255,255",
        help="Mask color in B,G,R format. Default is yellow.",
    )
    p.add_argument("--draw_contour", action="store_true", help="Draw contour line on mask boundary.")
    p.add_argument("--contour_thickness", type=int, default=1, help="Contour thickness in px.")
    return p.parse_args()


def parse_episode_arg(s: str) -> list[int]:
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def to_bgr_color(s: str) -> tuple[int, int, int]:
    vals = [int(x.strip()) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Invalid --mask_color: {s}")
    return tuple(vals)  # type: ignore[return-value]


def overlay_mask(
    image_bgr: np.ndarray,
    mask_gray: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float,
    draw_contour: bool,
    contour_thickness: int,
) -> np.ndarray:
    out = image_bgr.copy()
    if mask_gray is None:
        return out

    mask_bin = mask_gray > 0
    if not np.any(mask_bin):
        return out

    # Blend only inside mask region.
    color_img = np.zeros_like(out, dtype=np.uint8)
    color_img[:, :] = color_bgr
    blended = cv2.addWeighted(out, 1.0 - alpha, color_img, alpha, 0)
    out[mask_bin] = blended[mask_bin]

    if draw_contour:
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(out, contours, -1, color_bgr, contour_thickness)

    return out


def main():
    args = parse_args()

    frames_root = Path(args.frames_root)
    masks_root = Path(args.masks_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    episodes = parse_episode_arg(args.episodes)
    mask_color = to_bgr_color(args.mask_color)

    total_saved = 0
    total_missing_mask = 0

    for cam in cameras:
        for ep in episodes:
            ep_str = f"ep{ep:03d}"
            frame_dir = frames_root / cam / ep_str
            mask_dir = masks_root / cam / ep_str
            out_dir = out_root / cam / ep_str

            if not frame_dir.exists():
                print(f"[skip] missing frame dir: {frame_dir}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            frame_files = sorted(frame_dir.glob("frame_*.png"))
            if not frame_files:
                print(f"[skip] no frames: {frame_dir}")
                continue

            saved_here = 0
            missing_here = 0
            for fp in frame_files:
                mp = mask_dir / fp.name
                frame = cv2.imread(str(fp), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                if mp.exists():
                    mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                else:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    missing_here += 1

                over = overlay_mask(
                    frame,
                    mask,
                    color_bgr=mask_color,
                    alpha=args.alpha,
                    draw_contour=args.draw_contour,
                    contour_thickness=args.contour_thickness,
                )
                out_path = out_dir / fp.name
                cv2.imwrite(str(out_path), over)
                saved_here += 1

            total_saved += saved_here
            total_missing_mask += missing_here
            print(
                f"[done] {cam}/{ep_str}: saved={saved_here} missing_mask_fallback={missing_here}"
            )

    print(f"[summary] total_saved={total_saved} total_missing_mask_fallback={total_missing_mask}")


if __name__ == "__main__":
    main()
