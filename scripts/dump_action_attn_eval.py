#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import time
from typing import Any
from datetime import datetime, timedelta

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_stats
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import create_transition
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def parse_episodes(spec: str) -> set[int]:
    # Examples: "0-35" or "0,1,2,10-12"
    eps: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            eps.update(range(int(a), int(b) + 1))
        else:
            eps.add(int(part))
    return eps


def parse_int_spec(
    spec: str,
    *,
    name: str,
    max_value: int | None = None,
    allow_all: bool = False,
    allow_minus1: bool = False,
) -> list[int]:
    spec = str(spec).strip().lower()
    if allow_all and spec == "all":
        if max_value is None:
            raise ValueError(f"{name}: 'all' requires max_value")
        return list(range(max_value))

    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        # Treat a-b as range, but keep negative scalar like "-1" as scalar.
        if "-" in part and not part.startswith("-"):
            a_str, b_str = part.split("-", 1)
            a = int(a_str)
            b = int(b_str)
            if a > b:
                raise ValueError(f"{name}: invalid range {part}")
            values.extend(range(a, b + 1))
        else:
            values.append(int(part))

    if not values:
        raise ValueError(f"{name}: empty spec '{spec}'")

    if max_value is not None:
        normalized: list[int] = []
        for v in values:
            if v == -1 and allow_minus1:
                normalized.append(max_value - 1)
            else:
                if v < 0 or v >= max_value:
                    raise ValueError(f"{name}: value {v} out of range [0, {max_value - 1}]")
                normalized.append(v)
        values = normalized

    # Deduplicate while preserving order.
    seen = set()
    out: list[int] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def obs_with_expected_cameras(item: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    """Build observation dict while filling missing cameras with zeros."""
    obs = {OBS_STATE: item[OBS_STATE]}
    sample_img = None
    for k, v in item.items():
        if k.startswith("observation.images."):
            sample_img = v
            break

    for cam in camera_keys:
        if cam in item:
            obs[cam] = item[cam]
        else:
            if sample_img is None:
                raise RuntimeError(f"Missing expected camera '{cam}' and no fallback image found in sample.")
            obs[cam] = np.zeros_like(sample_img)
    return obs


def sanitize_pretrained_config_if_needed(pretrained_path: str | Path) -> None:
    """Fix known malformed fields in local config.json before loading policy."""
    cfg_path = Path(pretrained_path) / "config.json"
    if not cfg_path.exists():
        return

    raw = cfg_path.read_text()
    cfg = json.loads(raw)
    changed = False

    def _resolve_all_layers() -> list[int]:
        n = cfg.get("num_expert_layers", -1)
        if not isinstance(n, int) or n <= 0:
            n = cfg.get("num_vlm_layers", 16)
        return list(range(int(n)))

    for key in ["dump_action_attn_layers", "dump_lang_attn_layers"]:
        val = cfg.get(key)
        if not isinstance(val, str):
            continue
        s = val.strip().lower()
        if s == "all":
            cfg[key] = _resolve_all_layers()
        else:
            cfg[key] = parse_int_spec(s, name=key)
        changed = True

    # Backward-compat key normalization.
    if "dump_action_attn_denoise_steps" in cfg and "dump_action_attn_denoise_step" not in cfg:
        cfg["dump_action_attn_denoise_step"] = cfg["dump_action_attn_denoise_steps"]
        changed = True
    for stale_key in [
        "dump_action_attn_denoise_steps",
        "dump_action_attn_dir",
        "dump_action_attn_heads",
        "dump_lang_attn_heads",
    ]:
        if stale_key in cfg:
            cfg.pop(stale_key, None)
            changed = True

    if changed:
        backup = cfg_path.with_name(cfg_path.name + ".bak_attn_dump_fix")
        if not backup.exists():
            backup.write_text(raw)
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"[fix] sanitized malformed config field(s): {cfg_path}")
        print(f"[fix] backup saved at: {backup}")


def ensure_batch_dim(batch: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    """Ensure model inputs are batched as (B, ...)."""
    for key in [OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, *camera_keys]:
        if key not in batch:
            continue
        val = batch[key]
        if isinstance(val, torch.Tensor):
            # state/lang: 1D -> 2D, image: 3D (C,H,W) -> 4D
            if val.ndim in (1, 3):
                batch[key] = val.unsqueeze(0)
    return batch


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(device)
    return batch


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--dataset_repo_id", required=True)
    # Old arg and compatibility alias.
    ap.add_argument("--pretrained_path", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--episodes", default="0-35")
    # Single-value legacy options.
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--denoise_step", default=None, help="0-based; use -1 for last")
    # New-style options compatible with the user's local command.
    ap.add_argument("--layers", default=None, help="Comma/range list, e.g. 1,3,5,7 or 0-15")
    ap.add_argument("--denoise_steps", default=None, help="'all' or comma/range list")
    ap.add_argument("--action_step", default="0", help="'all' or comma/range list")
    ap.add_argument("--dump_action", action="store_true", help="Compatibility flag (action dump is default).")
    ap.add_argument("--dump_lang", action="store_true", help="Also capture language-to-image attention.")
    ap.add_argument("--lang_layer", type=int, default=15)
    ap.add_argument("--max_frames_per_episode", type=int, default=-1)
    ap.add_argument("--num_frames", type=int, default=None, help="Compatibility alias for max_frames_per_episode")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame within each episode")
    ap.add_argument("--heads", default="mean", choices=["mean"], help="Only head-mean is available in this codebase.")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--log_every_sec", type=float, default=60.0, help="Time-based progress log interval in seconds. <=0 disables.")
    ap.add_argument("--log_time", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    pretrained_path = args.pretrained_path or args.checkpoint
    if pretrained_path is None:
        raise ValueError("One of --pretrained_path or --checkpoint is required.")
    sanitize_pretrained_config_if_needed(pretrained_path)

    episodes = parse_episodes(args.episodes)
    if args.num_frames is not None:
        max_frames = None if args.num_frames < 0 else args.num_frames
    else:
        max_frames = None if args.max_frames_per_episode < 0 else args.max_frames_per_episode
    if args.stride <= 0:
        raise ValueError(f"--stride must be >= 1 (got {args.stride})")

    # Load dataset
    ds = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=dataset_root,
        revision="local",
        force_cache_sync=False,
        download_videos=False,
    )

    # Load policy
    load_t0 = time.time()
    print("[stage] loading policy checkpoint ...")
    policy = SmolVLAPolicy.from_pretrained(pretrained_path)
    policy.config.device = args.device
    policy.to(args.device)
    policy.config.dump_action_attn = True
    policy.config.dump_action_attn_last_denoise_only = False

    if args.layers is not None:
        layers = parse_int_spec(args.layers, name="layers")
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = [15]
    policy.config.dump_action_attn_layers = layers

    denoise_spec = args.denoise_steps if args.denoise_steps is not None else (args.denoise_step if args.denoise_step is not None else "-1")
    denoise_steps = parse_int_spec(
        denoise_spec,
        name="denoise_steps",
        max_value=policy.config.num_steps,
        allow_all=True,
        allow_minus1=True,
    )
    action_steps = parse_int_spec(
        args.action_step,
        name="action_step",
        max_value=policy.config.n_action_steps,
        allow_all=True,
        allow_minus1=False,
    )

    if args.dump_lang:
        policy.config.dump_lang_attn = True
        policy.config.dump_lang_attn_layers = [args.lang_layer]
    else:
        policy.config.dump_lang_attn = False

    stats = load_stats(dataset_root)
    preproc, _ = make_smolvla_pre_post_processors(policy.config, dataset_stats=stats)
    print(f"[stage] policy+preproc ready in {time.time() - load_t0:.1f}s")

    # Use policy's declared camera keys to support models expecting extra cameras.
    expected_camera_keys = sorted(
        [k for k in policy.config.input_features.keys() if k.startswith("observation.images.")]
    )
    if not expected_camera_keys:
        expected_camera_keys = list(ds.meta.camera_keys)

    # Estimate total workload to provide ETA.
    available_eps = len(ds.meta.episodes)
    selected_episodes = sorted(ep for ep in episodes if 0 <= ep < available_eps)
    skipped_episodes = sorted(ep for ep in episodes if ep < 0 or ep >= available_eps)
    if skipped_episodes:
        print(f"[warn] skipping out-of-range episodes: {skipped_episodes}")
    if not selected_episodes:
        raise ValueError("No valid episodes selected after filtering.")

    target_frames_per_episode: dict[int, int] = {}
    for ep in selected_episodes:
        ep_len = int(ds.meta.episodes[ep]["length"])
        frame_window = ep_len if max_frames is None else min(ep_len, max_frames)
        if frame_window <= 0:
            target_frames_per_episode[ep] = 0
        else:
            target_frames_per_episode[ep] = ((frame_window - 1) // args.stride) + 1

    total_target_frames = int(sum(target_frames_per_episode.values()))
    combo_count = len(layers) * len(denoise_steps) * len(action_steps)
    total_target_saves = int(total_target_frames * combo_count)
    print(
        f"[plan] episodes={len(selected_episodes)} target_frames={total_target_frames} "
        f"combos_per_frame={combo_count} target_saves={total_target_saves}"
    )

    # Track per-episode frame counts
    ep_counts: dict[int, int] = {}

    total_processed = 0
    total_saved = 0
    start_time = time.time()
    last_time_log = start_time

    def maybe_log_progress(force: bool = False) -> None:
        nonlocal last_time_log
        now = time.time()
        if not force:
            if args.log_every_sec <= 0:
                return
            if (now - last_time_log) < args.log_every_sec:
                return

        elapsed = now - start_time
        frame_rate = total_processed / elapsed if elapsed > 0 else 0.0
        save_rate = total_saved / elapsed if elapsed > 0 else 0.0

        eta_sec = None
        if total_target_saves > 0 and save_rate > 0:
            eta_sec = (total_target_saves - total_saved) / save_rate
        elif total_target_frames > 0 and frame_rate > 0:
            eta_sec = (total_target_frames - total_processed) / frame_rate

        eta_end = "n/a"
        if eta_sec is not None:
            eta_end = (datetime.now() + timedelta(seconds=eta_sec)).strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"[progress] frames={total_processed}/{total_target_frames} "
            f"saves={total_saved}/{total_target_saves} "
            f"elapsed={format_duration(elapsed)} "
            f"frame_rate={frame_rate:.3f}/s save_rate={save_rate:.3f}/s "
            f"eta={format_duration(eta_sec)} eta_end={eta_end}"
        )
        last_time_log = now

    maybe_log_progress(force=True)

    for idx in range(len(ds)):
        item = ds[idx]
        ep = int(item["episode_index"])
        if ep not in selected_episodes:
            continue

        ep_counts.setdefault(ep, 0)
        if (ep_counts[ep] % args.stride) != 0:
            ep_counts[ep] += 1
            continue
        if max_frames is not None and ep_counts[ep] >= max_frames:
            continue

        # Build observation dict
        obs = obs_with_expected_cameras(item, expected_camera_keys)

        transition = create_transition(
            observation=obs,
            complementary_data={"task": item["task"]},
        )
        processed = preproc._forward(transition)
        batch = preproc.to_output(processed)
        batch = ensure_batch_dim(batch, expected_camera_keys)
        batch = move_batch_to_device(batch, args.device)

        for denoise_step in denoise_steps:
            policy.config.dump_action_attn_denoise_step = denoise_step
            for action_step in action_steps:
                policy.config.dump_action_attn_action_step = action_step

                infer_t0 = time.time()
                _, attn_buffer, meta = policy.predict_action_chunk_with_attn(batch)
                infer_dt = time.time() - infer_t0

                action_buf = attn_buffer.get("action", {})
                missing_layers = [layer for layer in layers if layer not in action_buf]
                if missing_layers:
                    raise RuntimeError(f"attention missing for layers: {missing_layers}")

                lang_attn = None
                if args.dump_lang:
                    lang_buf = attn_buffer.get("lang", {})
                    if args.lang_layer not in lang_buf:
                        raise RuntimeError(f"lang attention for layer {args.lang_layer} not found in buffer")
                    lang_attn = lang_buf[args.lang_layer].detach().cpu().numpy()

                for layer_key in layers:
                    attn = action_buf[layer_key].detach().cpu().numpy()

                    # Save meta
                    meta_out = {
                        **meta,
                        "episode_index": ep,
                        "frame_index": int(item["frame_index"]),
                        "index": int(item["index"]),
                        "task": item["task"],
                        "layer": layer_key,
                        "denoise_step": denoise_step,
                        "action_step": action_step,
                        "lang_tokens": batch[OBS_LANGUAGE_TOKENS].detach().cpu().numpy().tolist(),
                        "lang_mask": batch[OBS_LANGUAGE_ATTENTION_MASK].detach().cpu().numpy().tolist(),
                    }

                    multi_combo = (len(layers) > 1) or (len(denoise_steps) > 1) or (len(action_steps) > 1)
                    if multi_combo:
                        stem = (
                            f"ep{ep:03d}_f{ep_counts[ep]:04d}"
                            f"_l{layer_key:02d}_d{denoise_step:02d}_a{action_step:02d}_action_attn"
                        )
                    else:
                        stem = f"ep{ep:03d}_f{ep_counts[ep]:04d}_action_attn"

                    out_npz = dump_dir / f"{stem}.npz"
                    out_json = dump_dir / f"{stem}_meta.json"
                    if lang_attn is None:
                        np.savez_compressed(out_npz, attn=attn)
                    else:
                        np.savez_compressed(out_npz, attn=attn, lang_attn=lang_attn)
                    out_json.write_text(json.dumps(meta_out))
                    total_saved += 1
                    maybe_log_progress(force=False)

                if args.log_time:
                    print(
                        f"ep={ep:03d} frame={ep_counts[ep]:04d} denoise={denoise_step} action={action_step} "
                        f"infer_time={infer_dt:.3f}s"
                    )

        ep_counts[ep] += 1
        total_processed += 1

        if args.log_every > 0 and (total_processed % args.log_every == 0):
            maybe_log_progress(force=True)

    # Write summary
    elapsed_total = time.time() - start_time
    maybe_log_progress(force=True)
    summary_path = dump_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "episodes": selected_episodes,
                "counts": ep_counts,
                "layers": layers,
                "denoise_steps": denoise_steps,
                "action_steps": action_steps,
                "saved_files": total_saved,
                "target_frames": total_target_frames,
                "target_saves": total_target_saves,
                "elapsed_s": elapsed_total,
            }
        )
    )


if __name__ == "__main__":
    main()
