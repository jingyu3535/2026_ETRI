#!/usr/bin/env python3
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig, ColorMode
from lerobot.datasets.utils import load_stats
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyAction
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def parse_args():
    p = argparse.ArgumentParser(description="Realtime action-attention with robot control (no saving).")
    p.add_argument("--pretrained_path", required=True, help="Path to SmolVLA pretrained_model")
    p.add_argument("--eval_root", required=True, help="Eval dataset root (for stats)")
    p.add_argument("--task", required=True, help="Language instruction")
    p.add_argument("--port", default="/dev/ttyACM1", help="Follower robot port")
    p.add_argument("--robot_id", default="my_follower", help="Robot id for calibration file lookup")
    p.add_argument("--calibration_dir", default="", help="Optional calibration dir")
    p.add_argument("--camera1", type=int, default=1, help="camera1 index (front)")
    p.add_argument("--camera2", type=int, default=0, help="camera2 index (top)")
    p.add_argument("--cam1_width", type=int, default=640)
    p.add_argument("--cam1_height", type=int, default=480)
    p.add_argument("--cam1_fps", type=int, default=30)
    p.add_argument("--cam2_width", type=int, default=640)
    p.add_argument("--cam2_height", type=int, default=480)
    p.add_argument("--cam2_fps", type=int, default=30)
    p.add_argument("--single_camera", action="store_true", help="Use only camera1 device; duplicate frame for camera2")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--layer", type=int, default=15, help="Layer to visualize")
    p.add_argument("--action_step", type=int, default=0, help="Action step row to visualize")
    p.add_argument("--control_dt", type=float, default=1 / 30, help="Control loop dt (s)")
    p.add_argument("--update_every", type=int, default=1, help="Update overlays/plots every N chunks")
    p.add_argument("--plot_tokens", action="store_true", help="Show token bar plot")
    p.add_argument("--downscale", type=float, default=1.0, help="Downscale camera frames before inference (e.g., 0.5)")
    p.add_argument("--show_fps", action="store_true", help="Overlay loop fps")
    p.add_argument("--record_out", default="", help="Optional path to save overlay video (mp4)")
    p.add_argument("--record_fps", type=float, default=20.0, help="FPS for recording output video")
    p.add_argument("--record_codec", default="mp4v", help="FourCC for recording (mp4v or avc1)")
    return p.parse_args()


def overlay_heatmap(img_bgr, heat):
    heat = np.clip(heat, 0.0, 1.0)
    heat_uint8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.5, heat_color, 0.5, 0)


def normalize_token(tok: str) -> str:
    # Common BPE/SentencePiece markers: Ġ (space), ▁ (space), Ċ (newline)
    return tok.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\\n")


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    # Policy + processors
    policy = SmolVLAPolicy.from_pretrained(args.pretrained_path)
    policy.to(device)
    policy.eval()

    policy.config.dump_action_attn = True
    policy.config.dump_action_attn_layers = [args.layer]
    policy.config.dump_action_attn_action_step = args.action_step
    policy.config.dump_action_attn_last_denoise_only = True

    stats = load_stats(Path(args.eval_root))
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=args.pretrained_path,
        dataset_stats=stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # Robot
    cam1_cfg = OpenCVCameraConfig(
        index_or_path=args.camera1,
        fps=args.cam1_fps,
        width=args.cam1_width,
        height=args.cam1_height,
        color_mode=ColorMode.RGB,
    )
    cam2_cfg = None
    if not args.single_camera:
        cam2_cfg = OpenCVCameraConfig(
            index_or_path=args.camera2,
            fps=args.cam2_fps,
            width=args.cam2_width,
            height=args.cam2_height,
            color_mode=ColorMode.RGB,
        )
    calib_dir = Path(args.calibration_dir).expanduser() if args.calibration_dir else None
    cameras = {"camera1": cam1_cfg}
    if cam2_cfg is not None:
        cameras["camera2"] = cam2_cfg
    robot_cfg = SOFollowerRobotConfig(
        port=args.port,
        id=args.robot_id,
        calibration_dir=calib_dir,
        cameras=cameras,
    )
    robot = SOFollower(robot_cfg)
    robot.connect()

    # Token bar plot (optional)
    if args.plot_tokens:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax = None

    # Action queue
    action_queue: deque[PolicyAction] = deque()
    last_overlay_1 = None
    last_overlay_2 = None

    motor_order = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    writer = None
    record_path = args.record_out.strip()

    chunk_idx = 0
    last_t = time.time()
    fps_hist: deque[float] = deque(maxlen=30)
    try:
        while True:
            obs = robot.get_observation()

            # Build batch for preprocess
            img1_raw = obs["camera1"]
            if args.downscale != 1.0:
                img1_raw = cv2.resize(
                    img1_raw, (0, 0), fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA
                )
            img1 = img1_raw.astype(np.float32) / 255.0
            if "camera2" in obs:
                img2_raw = obs["camera2"]
                if args.downscale != 1.0:
                    img2_raw = cv2.resize(
                        img2_raw, (0, 0), fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA
                    )
                img2 = img2_raw.astype(np.float32) / 255.0
            else:
                img2 = img1
            batch = {
                "observation.images.camera1": img1,
                "observation.images.camera2": img2,
                "observation.state": np.array([obs[f"{m}.pos"] for m in motor_order], dtype=np.float32),
                "task": args.task,
            }

            batch_t = preprocess(batch)
            # Ensure tensors are in (B, C, H, W) and have batch dim.
            for key in ("observation.images.camera1", "observation.images.camera2"):
                t = batch_t[key]
                if isinstance(t, torch.Tensor):
                    if t.ndim == 3 and t.shape[-1] == 3:
                        t = t.permute(2, 0, 1).unsqueeze(0)
                    elif t.ndim == 4 and t.shape[-1] == 3:
                        t = t.permute(0, 3, 1, 2)
                    batch_t[key] = t.to(device)
            state_t = batch_t["observation.state"]
            if isinstance(state_t, torch.Tensor):
                if state_t.ndim == 1:
                    state_t = state_t.unsqueeze(0)
                batch_t["observation.state"] = state_t.to(device)

            if len(action_queue) == 0:
                with torch.no_grad():
                    actions, attn_buffer, meta = policy.predict_action_chunk_with_attn(batch_t)
                # Unnormalize actions
                actions = postprocess(actions)
                # Fill queue
                for a in actions.transpose(0, 1):
                    action_queue.append(a)

                # Build overlays (latest chunk attn)
                action_buf = attn_buffer.get("action", {})
                attn = action_buf.get(args.layer)
                if attn is not None and (chunk_idx % args.update_every == 0):
                    attn = attn[0].detach().cpu().numpy()
                    img_spans = meta.get("img_spans", [])
                    img_grids = meta.get("img_grids", [])
                    lang_start, lang_end = meta.get("lang_range", (0, 0))
                    state_start, state_end = meta.get("state_range", (0, 0))

                    # ratios
                    img_sum = 0.0
                    for (s, e) in img_spans:
                        img_sum += float(attn[s:e].sum())
                    lang_sum = float(attn[lang_start:lang_end].sum())
                    state_sum = float(attn[state_start:state_end].sum())
                    total = img_sum + lang_sum + state_sum + 1e-9
                    ratios = (img_sum / total, lang_sum / total, state_sum / total)

                    # heatmaps
                    for cam_idx, (s, e) in enumerate(img_spans):
                        h, w = img_grids[cam_idx]
                        patch = attn[s:e]
                        if h * w != patch.shape[0]:
                            continue
                        heat = patch.reshape(h, w)
                        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
                        if cam_idx == 0:
                            target_h, target_w = img1_raw.shape[0], img1_raw.shape[1]
                        else:
                            if "camera2" in obs:
                                target_h, target_w = img2_raw.shape[0], img2_raw.shape[1]
                            else:
                                target_h, target_w = img1_raw.shape[0], img1_raw.shape[1]
                        heat = cv2.resize(heat, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        if cam_idx == 0:
                            last_overlay_1 = overlay_heatmap(
                                cv2.cvtColor(img1_raw, cv2.COLOR_RGB2BGR), heat
                            )
                        elif cam_idx == 1:
                            base = img2_raw if "camera2" in obs else img1_raw
                            last_overlay_2 = overlay_heatmap(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), heat)

                    # token bar plot (optional)
                    if args.plot_tokens and ax is not None:
                        tokenizer = policy.model.vlm_with_expert.processor.tokenizer
                        lang_tokens = batch_t[OBS_LANGUAGE_TOKENS][0].detach().cpu().numpy().tolist()
                        lang_mask = batch_t[OBS_LANGUAGE_ATTENTION_MASK][0].detach().cpu().numpy().tolist()
                        tokens = tokenizer.convert_ids_to_tokens(lang_tokens)
                        weights = attn[lang_start:lang_end]
                        specials = set(getattr(tokenizer, "all_special_ids", []))
                        filtered = []
                        for tid, t, w, m in zip(lang_tokens, tokens, weights, lang_mask):
                            if not m or tid in specials:
                                continue
                            filtered.append((normalize_token(t), w))
                        if filtered:
                            tok_labels, tok_w = zip(*filtered)
                            ax.clear()
                            ax.bar(range(len(tok_w)), tok_w)
                            ax.set_xticks(range(len(tok_labels)))
                            ax.set_xticklabels(tok_labels, rotation=90, fontsize=8)
                            ax.set_title("Language token attention (action-step 0)")
                            fig.canvas.draw()
                            fig.canvas.flush_events()

                    # overlay ratio text
                    if last_overlay_1 is not None:
                        text = f"img:{ratios[0]:.2f} lang:{ratios[1]:.2f} state:{ratios[2]:.2f}"
                        cv2.putText(last_overlay_1, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    if last_overlay_2 is not None:
                        text = f"img:{ratios[0]:.2f} lang:{ratios[1]:.2f} state:{ratios[2]:.2f}"
                        cv2.putText(last_overlay_2, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # send action
            if action_queue:
                action = action_queue.popleft()[0].detach().cpu().numpy()
                action_dict = {f"{m}.pos": float(action[i]) for i, m in enumerate(motor_order)}
                robot.send_action(action_dict)

            # show overlays
            if last_overlay_1 is not None:
                if args.show_fps:
                    now = time.time()
                    fps = 1.0 / max(1e-6, (now - last_t))
                    fps_hist.append(fps)
                    last_t = now
                    avg_fps = sum(fps_hist) / len(fps_hist)
                    cv2.putText(last_overlay_1, f"fps:{avg_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("camera1_action_attn", last_overlay_1)
                if record_path:
                    if writer is None:
                        h, w = last_overlay_1.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*args.record_codec)
                        writer = cv2.VideoWriter(record_path, fourcc, args.record_fps, (w, h))
                    writer.write(last_overlay_1)
            if last_overlay_2 is not None and "camera2" in obs:
                cv2.imshow("camera2_action_attn", last_overlay_2)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(args.control_dt)
            chunk_idx += 1

    finally:
        if writer is not None:
            writer.release()
        robot.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
