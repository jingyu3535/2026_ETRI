#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_episode_arg(spec: str) -> list[int]:
    eps: list[int] = []
    spec = spec.strip()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start > end:
                raise ValueError(f"invalid episode range: {part}")
            eps.extend(range(start, end + 1))
        else:
            eps.append(int(part))
    # keep order, remove duplicates
    seen = set()
    out = []
    for ep in eps:
        if ep in seen:
            continue
        seen.add(ep)
        out.append(ep)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Check frame/mask filename-set equality per episode/camera.")
    ap.add_argument("--frames_root", required=True, help="Root path of frames_by_ep.")
    ap.add_argument("--masks_root", required=True, help="Root path of masks (e.g., sam2_large).")
    ap.add_argument("--episodes", required=True, help="Episode spec like 0-35 or 0,1,2.")
    ap.add_argument("--cameras", default="camera1,camera2", help="Comma-separated camera keys.")
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    masks_root = Path(args.masks_root)
    episodes = parse_episode_arg(args.episodes)
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    all_ok = True
    for cam in cameras:
        for ep in episodes:
            epn = f"ep{ep:03d}"
            frame_dir = frames_root / cam / epn
            mask_dir = masks_root / cam / epn

            if not frame_dir.exists():
                all_ok = False
                print(f"[missing] frame dir: {frame_dir}")
                continue
            if not mask_dir.exists():
                all_ok = False
                print(f"[missing] mask dir: {mask_dir}")
                continue

            frames = {p.name for p in frame_dir.glob("frame_*.png")}
            masks = {p.name for p in mask_dir.glob("frame_*.png")}
            if frames != masks:
                all_ok = False
                print(
                    f"[mismatch] {cam} {epn} missing={len(frames - masks)} extra={len(masks - frames)}"
                )

    print(f"ALL_OK = {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
