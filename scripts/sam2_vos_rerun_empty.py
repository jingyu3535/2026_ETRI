#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser(description="Re-run SAM2 only for empty-mask frames and overwrite outputs.")
    ap.add_argument("--video_dir", required=True, help="Path to video_by_ep root.")
    ap.add_argument("--seed_frames_dir", required=True, help="Path to seg_seed_frames root.")
    ap.add_argument("--seed_masks_dir", required=True, help="Path to seg_seed_masks root.")
    ap.add_argument("--empty_frames_dir", required=True, help="Path to empty_mask_frames root.")
    ap.add_argument("--out_dir", required=True, help="Output directory to overwrite masks.")
    ap.add_argument("--episodes", required=True, help="Comma-separated list or ranges, e.g. 1,12,24 or 0-35")
    ap.add_argument("--cameras", default="camera1,camera2", help="Comma-separated camera names.")
    ap.add_argument("--model_cfg", required=True, help="SAM2 model config yaml.")
    ap.add_argument("--checkpoint", required=True, help="SAM2 checkpoint path.")
    ap.add_argument("--device", default="cuda", help="cuda or cpu.")
    ap.add_argument("--offload_video_to_cpu", action="store_true", help="Keep video frames on CPU to save GPU memory.")
    ap.add_argument("--offload_state_to_cpu", action="store_true", help="Keep inference state on CPU to save GPU memory.")
    ap.add_argument("--vos_optimized", action="store_true", help="Use VOS-optimized predictor (torch.compile).")
    ap.add_argument("--debug", action="store_true", help="Print debug info.")
    return ap.parse_args()


def parse_episode_list(s: str):
    eps = []
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        if "-" in part:
            a, b = part.split("-", 1)
            eps.extend(range(int(a), int(b) + 1))
        else:
            eps.append(int(part))
    return eps


def load_seed_prompts(seed_frames_dir: Path, seed_masks_dir: Path, ep_str: str):
    prompts = []
    for frame_path in sorted(seed_frames_dir.glob(f"ep{ep_str}_frame_*.png")):
        mask_path = seed_masks_dir / frame_path.name
        if not mask_path.exists():
            continue
        if mask_path.stat().st_size == 0:
            continue
        frame_num = int(frame_path.stem.split("_")[-1])  # 1-based
        frame_idx = frame_num - 1
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.sum() == 0:
            continue
        prompts.append((frame_idx, mask))
    return prompts


def load_empty_frame_indices(empty_dir: Path):
    # empty_dir has frame_XXXXX.png files (1-based)
    idxs = set()
    for p in empty_dir.glob("frame_*.png"):
        num = int(p.stem.split("_")[-1])
        idxs.add(num - 1)  # to 0-based
    return idxs


def main():
    args = parse_args()
    sam2_root = Path("/tmp/segment-anything-2")
    if sam2_root.exists():
        sys.path.insert(0, str(sam2_root))

    from sam2.build_sam import build_sam2_video_predictor

    video_root = Path(args.video_dir)
    seed_frames_root = Path(args.seed_frames_dir)
    seed_masks_root = Path(args.seed_masks_dir)
    empty_root = Path(args.empty_frames_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    episodes = parse_episode_list(args.episodes)
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    model_cfg = args.model_cfg
    cfg_path = Path(model_cfg)
    if cfg_path.is_absolute() and cfg_path.exists():
        try:
            rel = cfg_path.relative_to(sam2_root / "sam2")
            model_cfg = str(rel)
        except Exception:
            model_cfg = args.model_cfg

    predictor = build_sam2_video_predictor(
        model_cfg,
        args.checkpoint,
        device=args.device,
        vos_optimized=args.vos_optimized,
    )

    for ep in episodes:
        ep_str = f"{ep:03d}"
        for cam in cameras:
            video_path = video_root / cam / f"ep{ep_str}.mp4"
            if not video_path.exists():
                print(f"[skip] missing video: {video_path}")
                continue

            empty_dir = empty_root / cam / f"ep{ep_str}"
            if not empty_dir.exists():
                print(f"[skip] no empty frames: {empty_dir}")
                continue
            empty_idxs = load_empty_frame_indices(empty_dir)
            if not empty_idxs:
                print(f"[skip] empty list: {empty_dir}")
                continue

            prompts = load_seed_prompts(seed_frames_root / cam, seed_masks_root / cam, ep_str)
            if not prompts:
                print(f"[skip] no seed masks for ep{ep_str} {cam}")
                continue

            if args.debug:
                print(f"[debug] ep{ep_str} {cam} empty_frames={len(empty_idxs)} prompts={len(prompts)}")

            out_dir = out_root / cam / f"ep{ep_str}"
            out_dir.mkdir(parents=True, exist_ok=True)

            max_empty = max(empty_idxs)
            with torch.no_grad():
                state = predictor.init_state(
                    video_path=str(video_path),
                    offload_video_to_cpu=args.offload_video_to_cpu,
                    offload_state_to_cpu=args.offload_state_to_cpu,
                )
                obj_id = 1
                for frame_idx, mask in prompts:
                    predictor.add_new_mask(state, frame_idx=frame_idx, obj_id=obj_id, mask=mask)

                saved = 0
                for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
                    if frame_idx > max_empty:
                        break
                    if frame_idx not in empty_idxs:
                        continue
                    if not obj_ids:
                        continue
                    obj_idx = obj_ids.index(obj_id) if obj_id in obj_ids else 0
                    mask = masks[obj_idx]
                    mask_bin = (mask > 0).to(torch.uint8).cpu().numpy() * 255
                    if mask_bin.ndim == 3 and mask_bin.shape[0] == 1:
                        mask_bin = mask_bin[0]
                    out_path = out_dir / f"frame_{frame_idx+1:05d}.png"
                    cv2.imwrite(str(out_path), mask_bin)
                    saved += 1
                print(f"[done] ep{ep_str} {cam} saved={saved}")


if __name__ == "__main__":
    main()
