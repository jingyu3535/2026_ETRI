#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE


def parse_args():
    p = argparse.ArgumentParser(description="Realtime action-attention overlay for SmolVLA.")
    p.add_argument("--pretrained_path", required=True, help="Path to SmolVLA pretrained_model")
    p.add_argument("--task", required=True, help="Language instruction")
    p.add_argument("--camera1", type=int, default=0, help="Camera1 index (e.g. front)")
    p.add_argument("--camera2", type=int, default=1, help="Camera2 index (e.g. top)")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--step_every", type=int, default=10, help="Run model every N frames")
    p.add_argument("--layer", type=int, default=15, help="Layer to visualize")
    p.add_argument("--action_step", type=int, default=0, help="Action step row to visualize")
    return p.parse_args()


def overlay_heatmap(img_bgr, heat):
    heat = np.clip(heat, 0.0, 1.0)
    heat_uint8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.5, heat_color, 0.5, 0)


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    policy = SmolVLAPolicy.from_pretrained(args.pretrained_path)
    policy.to(device)
    policy.eval()

    policy.config.dump_action_attn = True
    policy.config.dump_action_attn_layers = [args.layer]
    policy.config.dump_action_attn_action_step = args.action_step
    policy.config.dump_action_attn_last_denoise_only = True

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokenized = tokenizer(
        args.task,
        padding="max_length",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
        return_tensors="pt",
    )
    lang_tokens = tokenized["input_ids"].to(device)
    lang_masks = tokenized["attention_mask"].to(device).bool()

    cap1 = cv2.VideoCapture(args.camera1)
    cap2 = cv2.VideoCapture(args.camera2)
    if not cap1.isOpened() or not cap2.isOpened():
        raise SystemExit("Failed to open cameras.")

    frame_id = 0
    last_overlay_1 = None
    last_overlay_2 = None

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        if frame_id % args.step_every == 0:
            img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)
            img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device)

            batch = {
                "observation.images.camera1": img1_t,
                "observation.images.camera2": img2_t,
                OBS_STATE: torch.zeros((1, 6), dtype=torch.float32, device=device),
                OBS_LANGUAGE_TOKENS: lang_tokens,
                OBS_LANGUAGE_ATTENTION_MASK: lang_masks,
            }

            with torch.no_grad():
                _, attn_buffer, meta = policy.predict_action_chunk_with_attn(batch)

            action_buf = attn_buffer.get("action", {})
            attn = action_buf.get(args.layer)
            if attn is not None:
                attn = attn[0].detach().cpu().numpy()
                camera_keys = meta.get("camera_keys", [])
                img_spans = meta.get("img_spans", [])
                img_grids = meta.get("img_grids", [])

                for cam_idx, cam_key in enumerate(camera_keys):
                    if cam_idx >= len(img_spans):
                        continue
                    start, end = img_spans[cam_idx]
                    h, w = img_grids[cam_idx]
                    patch = attn[start:end]
                    if h * w != patch.shape[0]:
                        continue
                    heat = patch.reshape(h, w)
                    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
                    heat = cv2.resize(heat, (frame1.shape[1], frame1.shape[0]), interpolation=cv2.INTER_LINEAR)

                    if cam_key.endswith("camera1"):
                        last_overlay_1 = overlay_heatmap(frame1, heat)
                    elif cam_key.endswith("camera2"):
                        last_overlay_2 = overlay_heatmap(frame2, heat)

        if last_overlay_1 is None:
            last_overlay_1 = frame1
        if last_overlay_2 is None:
            last_overlay_2 = frame2

        cv2.imshow("camera1_action_attn", last_overlay_1)
        cv2.imshow("camera2_action_attn", last_overlay_2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_id += 1

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
