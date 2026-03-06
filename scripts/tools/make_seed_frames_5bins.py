#!/usr/bin/env python3
import argparse
import shutil
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


def pick_5bin_indices(num_frames: int) -> list[int]:
    if num_frames <= 0:
        return []
    if num_frames == 1:
        return [0]
    idxs = [
        0,
        round((num_frames - 1) * 0.25),
        round((num_frames - 1) * 0.50),
        round((num_frames - 1) * 0.75),
        num_frames - 1,
    ]
    return sorted(set(idxs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy 5-bin seed frames per episode for SAM2 labeling.")
    ap.add_argument("--frames_root", required=True, help="Root of frames_by_ep (contains camera1/camera2).")
    ap.add_argument("--out_root", required=True, help="Root output directory for seed frames.")
    ap.add_argument("--episodes", default="", help="Episode spec like 0-35 or 0,1,2. Empty means all ep dirs found.")
    ap.add_argument("--cameras", default="camera1,camera2", help="Comma-separated camera keys.")
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    episodes = parse_episode_arg(args.episodes) if args.episodes.strip() else []

    total_copied = 0
    for cam in cameras:
        in_cam = frames_root / cam
        out_cam = out_root / cam
        out_cam.mkdir(parents=True, exist_ok=True)
        if not in_cam.exists():
            print(f"[skip] missing camera dir: {in_cam}")
            continue

        if episodes:
            ep_dirs = [in_cam / f"ep{ep:03d}" for ep in episodes]
        else:
            ep_dirs = sorted(in_cam.glob("ep*"))

        for ep_dir in ep_dirs:
            if not ep_dir.exists():
                print(f"[skip] missing episode dir: {ep_dir}")
                continue
            frames = sorted(ep_dir.glob("frame_*.png"))
            if not frames:
                print(f"[skip] no frames: {ep_dir}")
                continue

            idxs = pick_5bin_indices(len(frames))
            copied = 0
            for idx in idxs:
                src = frames[idx]
                dst = out_cam / f"{ep_dir.name}_{src.name}"
                shutil.copy2(src, dst)
                copied += 1
                total_copied += 1
            print(f"[done] {cam}/{ep_dir.name} -> {copied} seeds")

    print(f"[summary] copied={total_copied}")


if __name__ == "__main__":
    main()
