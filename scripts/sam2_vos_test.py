#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser(description="SAM2 VOS test on selected episodes/cameras.")
    ap.add_argument("--video_dir", required=True, help="Path to video_by1ep root.")
    ap.add_argument("--seed_frames_dir", required=True, help="Path to seg_seed_frames root.")
    ap.add_argument("--seed_masks_dir", required=True, help="Path to seg_seed_masks root.")
    ap.add_argument("--out_dir", required=True, help="Output directory for masks.")
    ap.add_argument("--episodes", required=True, help="Comma-separated list, e.g. 1,12,24")
    ap.add_argument("--cameras", default="camera1,camera2", help="Comma-separated camera names.")
    ap.add_argument("--model_cfg", required=True, help="SAM2 model config yaml.")
    ap.add_argument("--checkpoint", required=True, help="SAM2 checkpoint path.")
    ap.add_argument("--save_stride", type=int, default=20, help="Save every N frames.")
    ap.add_argument("--device", default="cuda", help="cuda or cpu.")
    ap.add_argument("--offload_video_to_cpu", action="store_true", help="Keep video frames on CPU to save GPU memory.")
    ap.add_argument("--offload_state_to_cpu", action="store_true", help="Keep inference state on CPU to save GPU memory.")
    ap.add_argument("--vos_optimized", action="store_true", help="Use VOS-optimized predictor (torch.compile).")
    ap.add_argument("--max_frames", type=int, default=-1, help="Stop after processing N frames (for quick tests).")
    ap.add_argument("--save_from_frame", type=int, default=-1, help="Skip saving until this frame idx (inclusive).")
    ap.add_argument("--fill_before_prompt", action="store_true", help="Save empty masks before first prompt frame.")
    ap.add_argument(
        "--out_index_base",
        type=int,
        default=0,
        choices=[0, 1],
        help="Output filename index base. 0 -> frame_00000, 1 -> frame_00001.",
    )
    ap.add_argument("--debug", action="store_true", help="Print debug info for first few frames.")
    return ap.parse_args()


def load_seed_prompts(seed_frames_dir: Path, seed_masks_dir: Path, ep_str: str):
    """Return list of (frame_idx0, mask) for the episode."""
    prompts = []
    for frame_path in sorted(seed_frames_dir.glob(f"ep{ep_str}_frame_*.png")):
        mask_path = seed_masks_dir / frame_path.name
        if not mask_path.exists():
            continue
        if mask_path.stat().st_size == 0:
            continue
        # frame number in filename is 1-based (00001). Convert to 0-based.
        frame_num = int(frame_path.stem.split("_")[-1])
        frame_idx = frame_num - 1
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.sum() == 0:
            continue
        prompts.append((frame_idx, mask))
    return prompts


def main():
    args = parse_args()
    # Allow importing sam2 from /tmp/segment-anything-2 if not installed.
    sam2_root = Path("/tmp/segment-anything-2")
    if sam2_root.exists():
        sys.path.insert(0, str(sam2_root))

    from sam2.build_sam import build_sam2_video_predictor

    video_root = Path(args.video_dir)
    seed_frames_root = Path(args.seed_frames_dir)
    seed_masks_root = Path(args.seed_masks_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    model_cfg = args.model_cfg
    cfg_path = Path(model_cfg)
    if cfg_path.is_absolute() and cfg_path.exists():
        # Convert to package-relative path expected by hydra compose, e.g. "configs/sam2.1/xxx.yaml"
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

            seed_frames_dir = seed_frames_root / cam
            seed_masks_dir = seed_masks_root / cam
            prompts = load_seed_prompts(seed_frames_dir, seed_masks_dir, ep_str)
            if not prompts:
                print(f"[skip] no seed masks for ep{ep_str} {cam}")
                continue
            if args.debug:
                print(f"[debug] ep{ep_str} {cam} prompts:", len(prompts))
            first_prompt = min(p[0] for p in prompts)
            # Decide save-from frame without mutating args across episodes.
            if args.save_from_frame >= 0:
                save_from_frame = args.save_from_frame
            elif args.fill_before_prompt:
                save_from_frame = 0
            else:
                save_from_frame = first_prompt

            out_dir = out_root / cam / f"ep{ep_str}"
            out_dir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                state = predictor.init_state(
                    video_path=str(video_path),
                    offload_video_to_cpu=args.offload_video_to_cpu,
                    offload_state_to_cpu=args.offload_state_to_cpu,
                )
                obj_id = 1
                # Add all seed masks as prompts for the same object
                for frame_idx, mask in prompts:
                    predictor.add_new_mask(state, frame_idx=frame_idx, obj_id=obj_id, mask=mask)

                processed = 0
                # If requested, explicitly create blank masks before first prompt.
                # propagate_in_video can start from the first prompted frame, so we
                # cannot rely on the loop below to emit pre-prompt frames.
                if args.fill_before_prompt and save_from_frame < first_prompt:
                    h = state.get("video_height")
                    w = state.get("video_width")
                    if h is None or w is None:
                        cap = cv2.VideoCapture(str(video_path))
                        ok, frame = cap.read()
                        cap.release()
                        if ok:
                            h, w = frame.shape[:2]
                    if h is None or w is None:
                        if args.debug:
                            print("[debug] could not determine video size for blank masks")
                    else:
                        blank = np.zeros((h, w), dtype=np.uint8)
                        start_idx = max(0, save_from_frame)
                        for frame_idx in range(start_idx, first_prompt):
                            if frame_idx % args.save_stride != 0:
                                continue
                            out_path = out_dir / f"frame_{frame_idx + args.out_index_base:05d}.png"
                            cv2.imwrite(str(out_path), blank)
                            processed += 1
                            if args.max_frames > 0 and processed >= args.max_frames:
                                break

                if args.max_frames > 0 and processed >= args.max_frames:
                    print(f"[done] ep{ep_str} {cam} -> {out_dir}")
                    continue

                for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
                    if args.debug and frame_idx < 3:
                        print(f"[debug] frame {frame_idx} obj_ids={obj_ids} masks={len(masks)}")
                    if frame_idx < save_from_frame:
                        continue
                    if frame_idx % args.save_stride != 0:
                        continue
                    if not obj_ids:
                        continue
                    if obj_id in obj_ids:
                        obj_idx = obj_ids.index(obj_id)
                    else:
                        # Fallback: use first available object
                        obj_idx = 0
                    mask = masks[obj_idx]
                    if args.debug and frame_idx == args.save_from_frame:
                        mmin = float(mask.min().item())
                        mmax = float(mask.max().item())
                        print(f"[debug] mask logits range: {mmin:.4f}..{mmax:.4f}")
                    # masks are logits; threshold at 0
                    mask_bin = (mask > 0).to(torch.uint8).cpu().numpy() * 255
                    if mask_bin.ndim == 3 and mask_bin.shape[0] == 1:
                        mask_bin = mask_bin[0]
                    out_path = out_dir / f"frame_{frame_idx + args.out_index_base:05d}.png"
                    ok = cv2.imwrite(str(out_path), mask_bin)
                    if args.debug and not ok:
                        print(f"[debug] imwrite failed: {out_path} shape={mask_bin.shape} dtype={mask_bin.dtype}")
                    processed += 1
                    if args.max_frames > 0 and processed >= args.max_frames:
                        break

            if processed == 0:
                print(f"[warn] no masks saved for ep{ep_str} {cam}")
            print(f"[done] ep{ep_str} {cam} -> {out_dir}")


if __name__ == "__main__":
    main()
