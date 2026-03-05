#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_stats
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track SmolVLA cross-attention over time.")
    parser.add_argument("--pretrained_path", required=True, help="Path to SmolVLA checkpoint/pretrained_model")
    parser.add_argument("--eval_root", required=True, help="Eval dataset root dir")
    parser.add_argument("--repo_id", required=True, help="Eval dataset repo_id")
    parser.add_argument("--camera_key", default="observation.images.camera2", help="Camera key to visualize")
    parser.add_argument("--out_dir", default="/tmp/attn_track", help="Output directory for frames")
    parser.add_argument("--stride", type=int, default=5, help="Frame stride for visualization")
    parser.add_argument("--max_frames", type=int, default=60, help="Max frames to export per episode")
    parser.add_argument("--last_episodes", type=int, default=1, help="Number of latest episodes to export")
    parser.add_argument(
        "--episodes",
        default="",
        help="Comma-separated episode indices to export (overrides last_episodes)",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip frames already saved on disk")
    parser.add_argument(
        "--focus_token",
        default="",
        help="If set, visualize attention from this language token to image tokens (e.g. 'banana')",
    )
    return parser.parse_args()


def get_episode_frame_indices(data_dir: Path) -> dict[int, list[int]]:
    rows = []
    for f in sorted(data_dir.rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "index"]).to_pylist()
        rows.extend([(r["episode_index"], r["index"]) for r in t])
    if not rows:
        return {}
    df = np.array(rows, dtype=np.int64)
    ep_map: dict[int, list[int]] = {}
    for ep in np.unique(df[:, 0]):
        idxs = df[df[:, 0] == ep][:, 1]
        ep_map[int(ep)] = np.sort(idxs).tolist()
    return ep_map


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=eval_root,
        revision="local",
        force_cache_sync=False,
        download_videos=False,
    )

    # Handle dataset stats across versions
    if hasattr(ds, "stats"):
        dataset_stats = ds.stats
    elif hasattr(ds, "meta") and hasattr(ds.meta, "stats"):
        dataset_stats = ds.meta.stats
    else:
        dataset_stats = load_stats(ds.root)

    policy = SmolVLAPolicy.from_pretrained(args.pretrained_path)
    policy.to(device)
    policy.eval()

    preprocess, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=args.pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    attn_records = []

    def eager_attention_forward_capture(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        seq_q = query_states.shape[1]
        seq_k = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, seq_k, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(batch_size, seq_k, num_key_value_heads * num_key_value_groups, head_dim)

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, seq_k, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(batch_size, seq_k, num_key_value_heads * num_key_value_groups, head_dim)

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)  # B, H, Q, D
        key_states = key_states.transpose(1, 2)      # B, H, K, D

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)

        attn_records.append(
            {"probs": probs.detach().cpu(), "q": int(seq_q), "k": int(seq_k)}
        )

        probs = probs.to(dtype=value_states.dtype)
        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
        return att_output

    policy.model.vlm_with_expert.eager_attention_forward = eager_attention_forward_capture.__get__(
        policy.model.vlm_with_expert, policy.model.vlm_with_expert.__class__
    )

    ep_map = get_episode_frame_indices(eval_root / "data")
    if not ep_map:
        raise SystemExit("No frames found in eval dataset.")

    if args.episodes:
        ep_ids = [int(x) for x in args.episodes.split(",") if x.strip() != ""]
    else:
        # select last N episodes
        ep_ids = sorted(ep_map.keys())[-args.last_episodes :]

    # Find camera index from config order
    cam_keys = list(policy.config.image_features.keys())
    if args.camera_key not in cam_keys:
        raise SystemExit(f"camera_key {args.camera_key} not in policy image features {cam_keys}")
    cam_idx = cam_keys.index(args.camera_key)

    total_saved = 0
    for ep in ep_ids:
        ep_dir = out_dir / f"ep{ep:02d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        frame_indices = ep_map[ep]
        if len(frame_indices) > args.max_frames:
            # Evenly sample across the full episode length.
            idxs = np.linspace(0, len(frame_indices) - 1, args.max_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in idxs]
        else:
            frame_indices = frame_indices[:: max(1, args.stride)]

        for i, idx in enumerate(frame_indices):
            attn_records.clear()
            sample = ds[int(idx)]
            batch = preprocess(sample)
            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

            # Build prefix to get token layout for language focus
            prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_len = prefix_pad_masks.shape[1]
            with torch.no_grad():
                _ = policy.predict_action_chunk(batch)

            if not attn_records:
                continue
            # Prefer prefix self-attn if available (language->image grounding)
            prefix_attn = None
            for rec in reversed(attn_records):
                if rec["q"] == prefix_len and rec["k"] == prefix_len:
                    prefix_attn = rec["probs"]
                    break
            if args.focus_token and prefix_attn is None:
                raise SystemExit("Prefix self-attention not captured; cannot compute token-level attention.")
            probs = prefix_attn if prefix_attn is not None else attn_records[-1]["probs"]

            # Compute image token lengths
            img_token_lens = []
            for img in images:
                img_emb = policy.model.vlm_with_expert.embed_image(img)
                img_token_lens.append(img_emb.shape[1])

            # Account for optional image special tokens
            add_special = policy.config.add_image_special_tokens
            img_spans = []
            cursor = 0
            for n in img_token_lens:
                start = cursor + (1 if add_special else 0)
                end = start + n
                img_spans.append((start, end))
                cursor = end + (1 if add_special else 0)

            # Language span
            lang_start = cursor
            lang_len = lang_tokens.shape[1]
            lang_end = lang_start + lang_len

            # Select attention map
            if args.focus_token:
                tokenizer = policy.model.vlm_with_expert.processor.tokenizer
                token_ids = lang_tokens[0].tolist()
                token_strs = tokenizer.convert_ids_to_tokens(token_ids)
                focus = args.focus_token.lower()
                focus_positions = [
                    lang_start + j
                    for j, s in enumerate(token_strs)
                    if focus in s.lower()
                ]
                if not focus_positions:
                    raise SystemExit(f"Focus token '{args.focus_token}' not found in tokenized prompt: {token_strs}")

                # Average attention from focus token(s) to image tokens
                # probs: [B, H, Q, K]
                att = probs[0]  # [H, Q, K]
                att = att.mean(dim=0)  # [Q, K]
                # image tokens for selected camera
                start, end = img_spans[cam_idx]
                cam_attn = att[focus_positions, start:end].mean(dim=0).numpy()
            else:
                # Average over heads and queries for camera token span
                att = probs[0]  # [H, Q, K]
                att = att.mean(dim=0)  # [Q, K]
                start, end = img_spans[cam_idx]
                cam_attn = att[:, start:end].mean(dim=0).numpy()

            n = len(cam_attn)
            side = int(math.sqrt(n))
            if side * side == n:
                attn_map = cam_attn.reshape(side, side)
            else:
                attn_map = cam_attn.reshape(1, n)

            img = batch[args.camera_key][0]
            img = (img + 1.0) / 2.0
            img = img.permute(1, 2, 0).cpu().numpy()

            attn_map = torch.tensor(attn_map)[None, None, :, :]
            attn_map = torch.nn.functional.interpolate(
                attn_map, size=img.shape[:2], mode="bilinear", align_corners=False
            )[0, 0].numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

            out_path = ep_dir / f"attn_{i:04d}.png"
            if args.skip_existing and out_path.exists():
                continue
            # Save raw image and attention map for ROI scoring.
            img_path = ep_dir / f"img_{i:04d}.png"
            npy_path = ep_dir / f"attn_{i:04d}.npy"
            if not img_path.exists():
                plt.imsave(img_path, img)
            np.save(npy_path, attn_map)

            plt.figure(figsize=(6, 4))
            plt.imshow(img)
            plt.imshow(attn_map, cmap="jet", alpha=0.45)
            plt.axis("off")
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            total_saved += 1

    print(f"saved {total_saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
