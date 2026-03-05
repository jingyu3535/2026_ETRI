#!/usr/bin/env python

"""
Compare SmolVLA cross-attention to object tokens (from SAM2 masks).

This script:
1) loads eval episodes from a local LeRobot dataset,
2) builds object-token masks from SAM2 per-frame masks,
3) captures cross-attention (action query -> prefix keys),
4) compares base vs tuned checkpoints on:
   - object_mass: attention mass on object tokens
   - object_ratio: object_mass / image_mass

Self-attention layers are excluded.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F  # noqa: N812
import numpy as np
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare object attention (base vs tuned) using SAM2 masks.")
    parser.add_argument("--base-path", type=Path, required=True, help="Base checkpoint directory.")
    parser.add_argument("--tuned-path", type=Path, required=True, help="Tuned checkpoint directory.")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/home/etri01/model/eval_task_box_750_0202"),
        help="Eval LeRobot dataset root (contains data/meta/videos).",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="eval_task_box_750_0202",
        help="Dataset repo_id for LeRobotDataset.",
    )
    parser.add_argument(
        "--sam2-mask-root",
        type=Path,
        default=Path("/home/etri01/model/eval_task_box_750_0202/sam2_base-plus"),
        help="SAM2 mask root with camera folders.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        choices=["camera1", "camera2", "both"],
        default="both",
        help="Which camera masks to use for object token mapping.",
    )
    parser.add_argument(
        "--max-samples-per-episode",
        type=int,
        default=4,
        help="Max number of masked frames sampled per episode.",
    )
    parser.add_argument("--num-steps", type=int, default=10, help="Denoising steps per sample.")
    parser.add_argument(
        "--step-mode",
        type=str,
        choices=["all", "last", "specific"],
        default="all",
        help="Which denoising steps to include in aggregation.",
    )
    parser.add_argument(
        "--step-index",
        type=int,
        default=0,
        help="Used only with --step-mode specific. Can be negative (e.g., -1 for last).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed base.")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu.")
    parser.add_argument(
        "--allow-empty-mask",
        action="store_true",
        help="If set, include frames with empty object mask.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("object_attention_compare.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--outcomes-csv",
        type=Path,
        default=Path("/home/etri01/model/eval_task_box_750_0202/eval_outcomes.csv"),
        help="CSV with columns: object,trial,outcome.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but no GPU is available. Falling back to CPU.")
        return "cpu"
    return device


def to_number(value: torch.Tensor) -> float:
    return float(value.detach().mean().item())


def factorize_tokens(num_tokens: int) -> tuple[int, int]:
    side = int(round(num_tokens**0.5))
    if side * side == num_tokens:
        return side, side
    for h in range(side, 0, -1):
        if num_tokens % h == 0:
            return h, num_tokens // h
    return 1, num_tokens


def resize_mask_with_pad(mask_2d: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    Match geometry used by image preprocessing (resize + left/top pad).
    mask_2d: [H, W] float tensor in {0,1}
    returns: [height, width]
    """
    if mask_2d.ndim != 2:
        raise ValueError(f"Expected [H,W] mask, got {tuple(mask_2d.shape)}")
    mask = mask_2d[None, None, :, :]
    cur_h, cur_w = mask.shape[2:]
    ratio = max(cur_w / width, cur_h / height)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)
    resized = F.interpolate(mask, size=(resized_h, resized_w), mode="nearest")
    pad_h = max(0, int(height - resized_h))
    pad_w = max(0, int(width - resized_w))
    padded = F.pad(resized, (pad_w, 0, pad_h, 0), value=0.0)
    return padded[0, 0]


class CrossAttentionRecorder:
    """
    Capture cross-attention stats from eager attention path.
    Records only tensors where:
      key_len == prefix_len and query_len == chunk_size.
    """

    def __init__(
        self,
        vlm_with_expert,
        *,
        image_len: int,
        language_len: int,
        state_len: int,
        chunk_size: int,
        prefix_len: int,
        object_token_mask: torch.Tensor | None,
        allowed_steps: set[int] | None,
    ):
        self.model = vlm_with_expert
        self.image_len = image_len
        self.language_len = language_len
        self.state_len = state_len
        self.chunk_size = chunk_size
        self.prefix_len = prefix_len
        self.object_token_mask = object_token_mask
        self.allowed_steps = allowed_steps
        self.records: list[dict] = []

        self._orig_eager = None
        self._orig_cross = None

    def __enter__(self):
        self._orig_eager = self.model.eager_attention_forward
        self._orig_cross = self.model.forward_cross_attn_layer
        recorder = self

        def eager_capture(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
            num_att_heads = self.num_attention_heads
            num_key_value_heads = self.num_key_value_heads
            num_key_value_groups = num_att_heads // num_key_value_heads
            sequence_length = key_states.shape[1]

            expanded_key_states = key_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            expanded_key_states = expanded_key_states.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )

            expanded_value_states = value_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            expanded_value_states = expanded_value_states.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )

            query_states_float = query_states.to(dtype=torch.float32).transpose(1, 2)
            key_states_float = expanded_key_states.to(dtype=torch.float32).transpose(1, 2)

            att_weights = torch.matmul(query_states_float, key_states_float.transpose(2, 3))
            att_weights *= head_dim**-0.5
            att_weights = att_weights.to(dtype=torch.float32)

            big_neg = torch.finfo(att_weights.dtype).min
            masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
            probs = torch.softmax(masked_att_weights, dim=-1).to(dtype=expanded_value_states.dtype)

            query_len = query_states.shape[1]
            key_len = sequence_length
            if key_len == recorder.prefix_len and query_len == recorder.chunk_size:
                current_step = int(getattr(self, "_capture_step_idx", -1))
                if recorder.allowed_steps is not None and current_step not in recorder.allowed_steps:
                    att_output = torch.matmul(probs, expanded_value_states.permute(0, 2, 1, 3))
                    att_output = att_output.permute(0, 2, 1, 3)
                    att_output = att_output.reshape(
                        batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
                    )
                    return att_output

                probs_mean = probs.to(dtype=torch.float32).mean(dim=(1, 2))

                img_end = recorder.image_len
                lang_end = recorder.image_len + recorder.language_len
                state_end = recorder.image_len + recorder.language_len + recorder.state_len

                image_probs = probs_mean[:, :img_end]
                image_mass = image_probs.sum(dim=1)
                language_mass = probs_mean[:, img_end:lang_end].sum(dim=1)
                state_mass = probs_mean[:, lang_end:state_end].sum(dim=1)

                if recorder.object_token_mask is not None and recorder.object_token_mask.numel() == img_end:
                    obj_mask = recorder.object_token_mask.to(image_probs.device)
                    object_mass = image_probs[:, obj_mask].sum(dim=1)
                    object_ratio = object_mass / torch.clamp(image_mass, min=1e-8)
                else:
                    object_mass = torch.zeros_like(image_mass)
                    object_ratio = torch.zeros_like(image_mass)

                recorder.records.append(
                    {
                        "step": current_step,
                        "layer": int(getattr(self, "_capture_layer_idx", -1)),
                        "image_mass": to_number(image_mass),
                        "language_mass": to_number(language_mass),
                        "state_mass": to_number(state_mass),
                        "object_mass": to_number(object_mass),
                        "object_ratio": to_number(object_ratio),
                    }
                )

            att_output = torch.matmul(probs, expanded_value_states.permute(0, 2, 1, 3))
            att_output = att_output.permute(0, 2, 1, 3)
            att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
            return att_output

        def cross_capture(self, *args, **kwargs):
            if len(args) >= 3:
                self._capture_layer_idx = int(args[2])
            return recorder._orig_cross(*args, **kwargs)

        self.model.eager_attention_forward = MethodType(eager_capture, self.model)
        self.model.forward_cross_attn_layer = MethodType(cross_capture, self.model)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.model.eager_attention_forward = self._orig_eager
        self.model.forward_cross_attn_layer = self._orig_cross
        if hasattr(self.model, "_capture_layer_idx"):
            delattr(self.model, "_capture_layer_idx")
        if hasattr(self.model, "_capture_step_idx"):
            delattr(self.model, "_capture_step_idx")


def load_policy(model_path: Path, config_template, device: str) -> SmolVLAPolicy:
    cfg = copy.deepcopy(config_template)
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(str(model_path), config=cfg)
    policy.eval()
    return policy


def load_preprocessor(tuned_path: Path, device: str) -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(tuned_path),
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": device}},
    )


def get_ep_dirs(mask_root: Path, camera: str) -> dict[int, Path]:
    cam_dir = mask_root / camera
    ep_dirs = {}
    if not cam_dir.exists():
        return ep_dirs
    for d in sorted(cam_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"ep(\d{3})", d.name)
        if m:
            ep_dirs[int(m.group(1))] = d
    return ep_dirs


def parse_frame_id(path: Path) -> int | None:
    m = re.fullmatch(r"frame_(\d{5})\.png", path.name)
    if not m:
        return None
    return int(m.group(1))


def normalize_object_name(text: str) -> str | None:
    low = text.lower()
    if "banana" in low:
        return "banana"
    if "sock" in low:
        return "socks"
    if "strawberry" in low:
        return "strawberry"
    return None


def load_mask_nonempty(mask_path: Path) -> bool:
    if not mask_path.exists():
        return False
    np_arr = np.array(Image.open(mask_path), dtype=np.uint8)
    return bool((np_arr > 0).any())


def build_episode_object_map(dataset: LeRobotDataset) -> dict[int, str]:
    ep_task: dict[int, str] = {}
    for ep_idx in range(dataset.num_episodes):
        ep_meta = dataset.meta.episodes[ep_idx]
        start_idx = int(ep_meta["dataset_from_index"])
        item = dataset[start_idx]
        obj = normalize_object_name(str(item["task"]))
        if obj is None:
            raise ValueError(f"Could not infer object from task text: {item['task']}")
        ep_task[ep_idx] = obj
    return ep_task


def load_episode_outcomes(outcomes_csv: Path, episode_object_map: dict[int, str]) -> dict[int, int | None]:
    if not outcomes_csv.exists():
        return {ep: None for ep in episode_object_map}

    outcome_by_object_trial: dict[str, dict[int, int]] = defaultdict(dict)
    with outcomes_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = normalize_object_name(str(row.get("object", "")))
            if obj is None:
                continue
            trial = int(row["trial"])
            outcome = int(row["outcome"])
            outcome_by_object_trial[obj][trial] = outcome

    episodes_by_object: dict[str, list[int]] = defaultdict(list)
    for ep, obj in episode_object_map.items():
        episodes_by_object[obj].append(ep)
    for obj in episodes_by_object:
        episodes_by_object[obj] = sorted(episodes_by_object[obj])

    ep_outcomes: dict[int, int | None] = {}
    for obj, eps in episodes_by_object.items():
        for trial_idx, ep in enumerate(eps, start=1):
            ep_outcomes[ep] = outcome_by_object_trial.get(obj, {}).get(trial_idx)
    return ep_outcomes


def build_index_map(dataset: LeRobotDataset) -> dict[tuple[int, int], int]:
    index_map: dict[tuple[int, int], int] = {}
    eps = dataset.hf_dataset["episode_index"]
    frs = dataset.hf_dataset["frame_index"]
    for idx, (ep, fr) in enumerate(zip(eps, frs, strict=False)):
        index_map[(int(ep), int(fr))] = idx
    return index_map


def select_samples(
    dataset: LeRobotDataset,
    index_map: dict[tuple[int, int], int],
    episode_outcomes: dict[int, int | None],
    mask_root: Path,
    camera_mode: str,
    max_samples_per_episode: int,
    require_nonempty: bool,
) -> list[dict]:
    if camera_mode == "both":
        cams = ["camera1", "camera2"]
    else:
        cams = [camera_mode]

    ep_frames: dict[int, set[int]] = defaultdict(set)
    ep_dirs_ref = get_ep_dirs(mask_root, cams[0])
    for ep, ep_dir in ep_dirs_ref.items():
        for p in ep_dir.glob("frame_*.png"):
            frame_id = parse_frame_id(p)
            if frame_id is not None:
                ep_frames[ep].add(frame_id)

    if camera_mode == "both":
        ep_dirs_other = get_ep_dirs(mask_root, "camera2")
        for ep, frame_set in list(ep_frames.items()):
            if ep not in ep_dirs_other:
                continue
            other_frames = set()
            for p in ep_dirs_other[ep].glob("frame_*.png"):
                frame_id = parse_frame_id(p)
                if frame_id is not None:
                    other_frames.add(frame_id)
            frame_set.update(other_frames)
            ep_frames[ep] = frame_set

    selected: list[dict] = []

    for ep in sorted(ep_frames.keys()):
        frames = sorted(ep_frames[ep])
        if require_nonempty:
            filtered = []
            for fr in frames:
                mask_nonempty = False
                for cam in cams:
                    p = mask_root / cam / f"ep{ep:03d}" / f"frame_{fr:05d}.png"
                    if load_mask_nonempty(p):
                        mask_nonempty = True
                        break
                if mask_nonempty:
                    filtered.append(fr)
            frames = filtered

        if not frames:
            continue

        if max_samples_per_episode > 0 and len(frames) > max_samples_per_episode:
            n = max_samples_per_episode
            idxs = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
            frames = [frames[i] for i in idxs]

        for fr in frames:
            key = (ep, fr)
            if key not in index_map:
                continue
            abs_idx = index_map[key]
            record = dataset[abs_idx]
            selected.append(
                {
                    "abs_index": int(abs_idx),
                    "episode_index": int(ep),
                    "frame_index": int(fr),
                    "task": str(record["task"]),
                    "outcome": episode_outcomes.get(int(ep)),
                }
            )

    return selected


def load_image_ranges(eval_root: Path, ep: int, frame: int) -> list[dict] | None:
    meta_path = eval_root / "lang_attn_dump_img_all" / f"ep{ep:03d}_f{frame:03d}_lang_attn_meta.json"
    if not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return data.get("image_ranges")


def build_object_token_mask(
    *,
    ep: int,
    frame: int,
    mask_root: Path,
    camera_mode: str,
    present_img_keys: list[str],
    image_len: int,
    image_ranges: list[dict] | None,
    target_size: tuple[int, int],
) -> tuple[torch.Tensor, dict]:
    width, height = target_size
    token_masks: list[torch.Tensor] = []
    debug = {"camera_token_counts": {}, "camera_object_tokens": {}}

    for i, img_key in enumerate(present_img_keys):
        cam_name = img_key.split(".")[-1]
        if image_ranges is not None and i < len(image_ranges):
            num_tokens = int(image_ranges[i]["num_patches"])
        else:
            # fallback: equal split if metadata is unavailable
            num_tokens = image_len // max(1, len(present_img_keys))

        h_tok, w_tok = factorize_tokens(num_tokens)

        use_this_cam = camera_mode == "both" or camera_mode == cam_name
        if not use_this_cam:
            token_mask = torch.zeros(num_tokens, dtype=torch.bool)
            token_masks.append(token_mask)
            debug["camera_token_counts"][cam_name] = num_tokens
            debug["camera_object_tokens"][cam_name] = int(token_mask.sum().item())
            continue

        mask_path = mask_root / cam_name / f"ep{ep:03d}" / f"frame_{frame:05d}.png"
        if not mask_path.exists():
            token_mask = torch.zeros(num_tokens, dtype=torch.bool)
            token_masks.append(token_mask)
            debug["camera_token_counts"][cam_name] = num_tokens
            debug["camera_object_tokens"][cam_name] = int(token_mask.sum().item())
            continue

        mask_np = np.array(Image.open(mask_path), dtype=np.uint8)
        mask_bin = torch.from_numpy((mask_np > 0).astype("float32"))

        padded = resize_mask_with_pad(mask_bin, width=width, height=height)
        pooled = F.adaptive_max_pool2d(padded[None, None], (h_tok, w_tok))[0, 0]
        token_mask = pooled > 0
        token_mask = token_mask.reshape(-1)

        if token_mask.numel() != num_tokens:
            token_mask = token_mask[:num_tokens]
            if token_mask.numel() < num_tokens:
                pad = torch.zeros(num_tokens - token_mask.numel(), dtype=torch.bool)
                token_mask = torch.cat([token_mask, pad], dim=0)

        token_masks.append(token_mask)
        debug["camera_token_counts"][cam_name] = num_tokens
        debug["camera_object_tokens"][cam_name] = int(token_mask.sum().item())

    if token_masks:
        image_object_mask = torch.cat(token_masks, dim=0)
    else:
        image_object_mask = torch.zeros(image_len, dtype=torch.bool)

    if image_object_mask.numel() != image_len:
        image_object_mask = image_object_mask[:image_len]
        if image_object_mask.numel() < image_len:
            pad = torch.zeros(image_len - image_object_mask.numel(), dtype=torch.bool)
            image_object_mask = torch.cat([image_object_mask, pad], dim=0)

    debug["image_len"] = image_len
    debug["object_token_count"] = int(image_object_mask.sum().item())
    return image_object_mask, debug


def run_single_sample(
    policy: SmolVLAPolicy,
    batch: dict,
    *,
    ep: int,
    frame: int,
    eval_root: Path,
    mask_root: Path,
    camera_mode: str,
    sample_seed: int,
    num_steps: int,
    step_mode: str,
    step_index: int,
) -> tuple[list[dict], dict]:
    model = policy.model

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_len = int(prefix_embs.shape[1])
    language_len = int(lang_tokens.shape[1])
    state_len = int(1 if state.ndim == 2 else state.shape[1])
    image_len = prefix_len - language_len - state_len

    present_img_keys = [k for k in policy.config.image_features if k in batch]
    image_ranges = load_image_ranges(eval_root, ep, frame)
    image_object_mask, mask_debug = build_object_token_mask(
        ep=ep,
        frame=frame,
        mask_root=mask_root,
        camera_mode=camera_mode,
        present_img_keys=present_img_keys,
        image_len=image_len,
        image_ranges=image_ranges,
        target_size=tuple(policy.config.resize_imgs_with_padding),
    )

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, past_key_values = model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=policy.config.use_cache,
        fill_kv_cache=True,
    )

    batch_size = state.shape[0]
    actions_shape = (batch_size, policy.config.chunk_size, policy.config.max_action_dim)
    generator = torch.Generator(device=state.device)
    generator.manual_seed(sample_seed)
    noise = torch.randn(actions_shape, device=state.device, generator=generator)
    x_t = noise

    if step_mode == "all":
        allowed_steps = None
    elif step_mode == "last":
        allowed_steps = {num_steps - 1}
    elif step_mode == "specific":
        resolved_step = step_index if step_index >= 0 else num_steps + step_index
        if resolved_step < 0 or resolved_step >= num_steps:
            raise ValueError(
                f"Resolved step index is out of range: {resolved_step}. num_steps={num_steps}, step_index={step_index}"
            )
        allowed_steps = {resolved_step}
    else:
        raise ValueError(f"Unknown step_mode: {step_mode}")

    recorder = CrossAttentionRecorder(
        model.vlm_with_expert,
        image_len=image_len,
        language_len=language_len,
        state_len=state_len,
        chunk_size=policy.config.chunk_size,
        prefix_len=prefix_len,
        object_token_mask=image_object_mask.to(state.device),
        allowed_steps=allowed_steps,
    )

    dt = -1.0 / num_steps
    with recorder:
        for step in range(num_steps):
            model.vlm_with_expert._capture_step_idx = int(step)
            time_value = 1.0 + step * dt
            time_tensor = torch.tensor(time_value, dtype=torch.float32, device=state.device).expand(batch_size)
            v_t = model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

    sample_debug = {
        "image_len": image_len,
        "language_len": language_len,
        "state_len": state_len,
        "object_token_count": int(image_object_mask.sum().item()),
        **mask_debug,
    }
    return recorder.records, sample_debug


def aggregate_records(records: list[dict]) -> dict:
    if not records:
        return {
            "num_records": 0,
            "overall": {
                "image_mass": 0.0,
                "language_mass": 0.0,
                "state_mass": 0.0,
                "object_mass": 0.0,
                "object_ratio": 0.0,
            },
            "by_task": {},
            "by_episode": {},
            "by_outcome": {},
        }

    def mean(vals: list[float]) -> float:
        return float(sum(vals) / len(vals))

    overall = {
        "image_mass": mean([r["image_mass"] for r in records]),
        "language_mass": mean([r["language_mass"] for r in records]),
        "state_mass": mean([r["state_mass"] for r in records]),
        "object_mass": mean([r["object_mass"] for r in records]),
        "object_ratio": mean([r["object_ratio"] for r in records]),
    }

    by_task_raw = defaultdict(list)
    by_episode_raw = defaultdict(list)
    by_outcome_raw = defaultdict(list)
    for r in records:
        by_task_raw[r["task"]].append(r)
        by_episode_raw[r["episode_index"]].append(r)
        outcome_key = str(r["outcome"]) if r.get("outcome") is not None else "unknown"
        by_outcome_raw[outcome_key].append(r)

    by_task = {}
    for task, rs in by_task_raw.items():
        by_task[task] = {
            "count": len(rs),
            "object_mass": mean([x["object_mass"] for x in rs]),
            "object_ratio": mean([x["object_ratio"] for x in rs]),
            "image_mass": mean([x["image_mass"] for x in rs]),
        }

    by_episode = {}
    for ep, rs in sorted(by_episode_raw.items(), key=lambda kv: int(kv[0])):
        by_episode[str(ep)] = {
            "count": len(rs),
            "object_mass": mean([x["object_mass"] for x in rs]),
            "object_ratio": mean([x["object_ratio"] for x in rs]),
        }

    by_outcome = {}
    for outcome, rs in sorted(by_outcome_raw.items(), key=lambda kv: kv[0]):
        by_outcome[str(outcome)] = {
            "count": len(rs),
            "object_mass": mean([x["object_mass"] for x in rs]),
            "object_ratio": mean([x["object_ratio"] for x in rs]),
            "image_mass": mean([x["image_mass"] for x in rs]),
        }

    return {
        "num_records": len(records),
        "overall": overall,
        "by_task": by_task,
        "by_episode": by_episode,
        "by_outcome": by_outcome,
    }


def evaluate_checkpoint(
    *,
    model_name: str,
    model_path: Path,
    config_template,
    preprocessor: PolicyProcessorPipeline,
    dataset: LeRobotDataset,
    samples: list[dict],
    eval_root: Path,
    mask_root: Path,
    camera_mode: str,
    seed: int,
    num_steps: int,
    step_mode: str,
    step_index: int,
    device: str,
) -> dict:
    policy = load_policy(model_path=model_path, config_template=config_template, device=device)
    all_records = []
    sample_debug = []

    with torch.no_grad():
        for s in samples:
            raw_item = dataset[s["abs_index"]]
            batch = preprocessor(raw_item)
            sample_seed = seed + s["abs_index"]

            records, debug = run_single_sample(
                policy=policy,
                batch=batch,
                ep=s["episode_index"],
                frame=s["frame_index"],
                eval_root=eval_root,
                mask_root=mask_root,
                camera_mode=camera_mode,
                sample_seed=sample_seed,
                num_steps=num_steps,
                step_mode=step_mode,
                step_index=step_index,
            )

            for r in records:
                r["sample_index"] = int(s["abs_index"])
                r["episode_index"] = int(s["episode_index"])
                r["frame_index"] = int(s["frame_index"])
                r["task"] = s["task"]
                r["outcome"] = s.get("outcome")
            all_records.extend(records)

            sample_debug.append(
                {
                    "sample_index": int(s["abs_index"]),
                    "episode_index": int(s["episode_index"]),
                    "frame_index": int(s["frame_index"]),
                    "task": s["task"],
                    "outcome": s.get("outcome"),
                    **debug,
                }
            )

    summary = aggregate_records(all_records)
    summary["model_name"] = model_name
    summary["model_path"] = str(model_path)
    summary["samples"] = samples
    summary["sample_debug"] = sample_debug
    return summary


def diff_summaries(base_summary: dict, tuned_summary: dict) -> dict:
    keys = ["image_mass", "language_mass", "state_mass", "object_mass", "object_ratio"]
    return {"overall_delta": {k: tuned_summary["overall"][k] - base_summary["overall"][k] for k in keys}}


def main():
    args = parse_args()
    device = resolve_device(args.device)
    require_nonempty = not args.allow_empty_mask

    tuned_cfg = PreTrainedConfig.from_pretrained(str(args.tuned_path))
    tuned_cfg.device = device

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=str(args.eval_root),
        download_videos=False,
    )
    episode_object_map = build_episode_object_map(dataset)
    episode_outcomes = load_episode_outcomes(args.outcomes_csv, episode_object_map)
    index_map = build_index_map(dataset)
    samples = select_samples(
        dataset=dataset,
        index_map=index_map,
        episode_outcomes=episode_outcomes,
        mask_root=args.sam2_mask_root,
        camera_mode=args.camera,
        max_samples_per_episode=args.max_samples_per_episode,
        require_nonempty=require_nonempty,
    )
    if not samples:
        raise RuntimeError("No samples selected. Check mask root / camera setting.")

    print(f"Selected samples: {len(samples)}")

    preprocessor = load_preprocessor(tuned_path=args.tuned_path, device=device)

    base_summary = evaluate_checkpoint(
        model_name="base",
        model_path=args.base_path,
        config_template=tuned_cfg,
        preprocessor=preprocessor,
        dataset=dataset,
        samples=samples,
        eval_root=args.eval_root,
        mask_root=args.sam2_mask_root,
        camera_mode=args.camera,
        seed=args.seed,
        num_steps=args.num_steps,
        step_mode=args.step_mode,
        step_index=args.step_index,
        device=device,
    )
    tuned_summary = evaluate_checkpoint(
        model_name="tuned",
        model_path=args.tuned_path,
        config_template=tuned_cfg,
        preprocessor=preprocessor,
        dataset=dataset,
        samples=samples,
        eval_root=args.eval_root,
        mask_root=args.sam2_mask_root,
        camera_mode=args.camera,
        seed=args.seed,
        num_steps=args.num_steps,
        step_mode=args.step_mode,
        step_index=args.step_index,
        device=device,
    )
    delta = diff_summaries(base_summary, tuned_summary)

    result = {
        "settings": {
            "base_path": str(args.base_path),
            "tuned_path": str(args.tuned_path),
            "eval_root": str(args.eval_root),
            "dataset_repo_id": args.dataset_repo_id,
            "sam2_mask_root": str(args.sam2_mask_root),
            "outcomes_csv": str(args.outcomes_csv),
            "camera": args.camera,
            "max_samples_per_episode": args.max_samples_per_episode,
            "num_steps": args.num_steps,
            "step_mode": args.step_mode,
            "step_index": args.step_index,
            "seed": args.seed,
            "device": device,
            "require_nonempty_mask": require_nonempty,
            "note": "Self-attention excluded. Reports cross-attention from action query to object/image/language/state.",
        },
        "base": base_summary,
        "tuned": tuned_summary,
        "delta": delta,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        "[base] "
        f"object_mass={base_summary['overall']['object_mass']:.4f} "
        f"object_ratio={base_summary['overall']['object_ratio']:.4f}"
    )
    print(
        "[tuned] "
        f"object_mass={tuned_summary['overall']['object_mass']:.4f} "
        f"object_ratio={tuned_summary['overall']['object_ratio']:.4f}"
    )
    print(
        "[delta] "
        f"object_mass={delta['overall_delta']['object_mass']:+.4f} "
        f"object_ratio={delta['overall_delta']['object_ratio']:+.4f}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
