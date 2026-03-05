#!/usr/bin/env python

"""
Compare SmolVLA cross-attention allocation from action queries to prefix tokens.

This script compares two checkpoints (typically base vs finetuned) on the same
samples and reports how much cross-attention mass goes to:
  - image tokens
  - language tokens
  - state tokens

Self-attention layers are intentionally excluded.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from types import MethodType

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SmolVLA cross-attention mass by token type.")
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Base checkpoint directory (e.g., local cache snapshot of lerobot/smolvla_base).",
    )
    parser.add_argument(
        "--tuned-path",
        type=Path,
        required=True,
        help="Finetuned checkpoint directory (e.g., .../checkpoints/500000/pretrained_model).",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="task_box_100",
        help="LeRobot dataset repo_id for local dataset loading.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/etri01/.cache/huggingface/lerobot/etri01/task_box_100"),
        help="Local dataset directory path.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated absolute frame indices from dataset.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on, e.g., cuda or cpu.")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of denoising steps to run for attention capture.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed for deterministic per-sample noise.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cross_attention_compare.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def parse_indices(indices_text: str) -> list[int]:
    values = []
    for x in indices_text.split(","):
        x = x.strip()
        if not x:
            continue
        values.append(int(x))
    if not values:
        raise ValueError("No valid sample indices provided.")
    return values


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but no GPU is available. Falling back to CPU.")
        return "cpu"
    return device


def to_number(value: torch.Tensor) -> float:
    return float(value.detach().mean().item())


class CrossAttentionRecorder:
    """
    Patch SmolVLMWithExpertModel.eager_attention_forward to collect cross-attention
    mass by prefix segment. Only captures tensors with key_len == prefix_len and
    query_len == chunk_size.
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
    ):
        self.model = vlm_with_expert
        self.image_len = image_len
        self.language_len = language_len
        self.state_len = state_len
        self.chunk_size = chunk_size
        self.prefix_len = prefix_len
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
            probs = torch.softmax(masked_att_weights, dim=-1)
            probs = probs.to(dtype=expanded_value_states.dtype)

            query_len = query_states.shape[1]
            key_len = sequence_length
            if key_len == recorder.prefix_len and query_len == recorder.chunk_size:
                probs_mean = probs.to(dtype=torch.float32).mean(dim=(1, 2))

                img_end = recorder.image_len
                lang_end = recorder.image_len + recorder.language_len
                state_end = recorder.image_len + recorder.language_len + recorder.state_len

                image_mass = probs_mean[:, :img_end].sum(dim=1)
                language_mass = probs_mean[:, img_end:lang_end].sum(dim=1)
                state_mass = probs_mean[:, lang_end:state_end].sum(dim=1)

                recorder.records.append(
                    {
                        "step": int(getattr(self, "_capture_step_idx", -1)),
                        "layer": int(getattr(self, "_capture_layer_idx", -1)),
                        "image_mass": to_number(image_mass),
                        "language_mass": to_number(language_mass),
                        "state_mass": to_number(state_mass),
                    }
                )

            att_output = torch.matmul(probs, expanded_value_states.permute(0, 2, 1, 3))
            att_output = att_output.permute(0, 2, 1, 3)
            att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
            return att_output

        def cross_capture(self, *args, **kwargs):
            # args index 2 is layer_idx in forward_cross_attn_layer signature.
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


def aggregate_records(records: list[dict]) -> dict:
    if not records:
        return {
            "num_records": 0,
            "overall": {"image_mass": 0.0, "language_mass": 0.0, "state_mass": 0.0},
            "by_layer": {},
            "by_step": {},
        }

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values))

    overall = {
        "image_mass": mean([r["image_mass"] for r in records]),
        "language_mass": mean([r["language_mass"] for r in records]),
        "state_mass": mean([r["state_mass"] for r in records]),
    }

    by_layer_raw = defaultdict(list)
    by_step_raw = defaultdict(list)
    for record in records:
        by_layer_raw[record["layer"]].append(record)
        by_step_raw[record["step"]].append(record)

    by_layer = {}
    for layer, layer_records in sorted(by_layer_raw.items(), key=lambda kv: kv[0]):
        by_layer[str(layer)] = {
            "count": len(layer_records),
            "image_mass": mean([r["image_mass"] for r in layer_records]),
            "language_mass": mean([r["language_mass"] for r in layer_records]),
            "state_mass": mean([r["state_mass"] for r in layer_records]),
        }

    by_step = {}
    for step, step_records in sorted(by_step_raw.items(), key=lambda kv: kv[0]):
        by_step[str(step)] = {
            "count": len(step_records),
            "image_mass": mean([r["image_mass"] for r in step_records]),
            "language_mass": mean([r["language_mass"] for r in step_records]),
            "state_mass": mean([r["state_mass"] for r in step_records]),
        }

    return {
        "num_records": len(records),
        "overall": overall,
        "by_layer": by_layer,
        "by_step": by_step,
    }


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


def run_single_sample(
    policy: SmolVLAPolicy,
    batch: dict,
    *,
    sample_seed: int,
    num_steps: int,
) -> list[dict]:
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
    if image_len < 0:
        raise ValueError(
            f"Invalid segment lengths: prefix_len={prefix_len}, language_len={language_len}, state_len={state_len}"
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

    recorder = CrossAttentionRecorder(
        model.vlm_with_expert,
        image_len=image_len,
        language_len=language_len,
        state_len=state_len,
        chunk_size=policy.config.chunk_size,
        prefix_len=prefix_len,
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

    return recorder.records


def evaluate_checkpoint(
    *,
    model_name: str,
    model_path: Path,
    config_template,
    preprocessor: PolicyProcessorPipeline,
    dataset: LeRobotDataset,
    sample_indices: list[int],
    seed: int,
    num_steps: int,
    device: str,
) -> dict:
    policy = load_policy(model_path=model_path, config_template=config_template, device=device)
    all_records = []

    with torch.no_grad():
        for sample_idx in sample_indices:
            raw_item = dataset[sample_idx]
            batch = preprocessor(raw_item)
            sample_seed = seed + sample_idx
            sample_records = run_single_sample(
                policy=policy,
                batch=batch,
                sample_seed=sample_seed,
                num_steps=num_steps,
            )
            for record in sample_records:
                record["sample_index"] = int(sample_idx)
            all_records.extend(sample_records)

    summary = aggregate_records(all_records)
    summary["model_name"] = model_name
    summary["model_path"] = str(model_path)
    summary["sample_indices"] = sample_indices
    return summary


def diff_summaries(base_summary: dict, tuned_summary: dict) -> dict:
    def delta(key: str) -> float:
        return tuned_summary["overall"][key] - base_summary["overall"][key]

    return {
        "overall_delta": {
            "image_mass": delta("image_mass"),
            "language_mass": delta("language_mass"),
            "state_mass": delta("state_mass"),
        }
    }


def print_brief(name: str, summary: dict):
    overall = summary["overall"]
    print(
        f"[{name}] records={summary['num_records']} "
        f"image={overall['image_mass']:.4f} "
        f"language={overall['language_mass']:.4f} "
        f"state={overall['state_mass']:.4f}"
    )


def main():
    args = parse_args()
    sample_indices = parse_indices(args.sample_indices)
    device = resolve_device(args.device)

    tuned_cfg = PreTrainedConfig.from_pretrained(str(args.tuned_path))
    tuned_cfg.device = device

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=str(args.dataset_root),
        download_videos=False,
    )
    preprocessor = load_preprocessor(tuned_path=args.tuned_path, device=device)

    base_summary = evaluate_checkpoint(
        model_name="base",
        model_path=args.base_path,
        config_template=tuned_cfg,
        preprocessor=preprocessor,
        dataset=dataset,
        sample_indices=sample_indices,
        seed=args.seed,
        num_steps=args.num_steps,
        device=device,
    )
    tuned_summary = evaluate_checkpoint(
        model_name="tuned",
        model_path=args.tuned_path,
        config_template=tuned_cfg,
        preprocessor=preprocessor,
        dataset=dataset,
        sample_indices=sample_indices,
        seed=args.seed,
        num_steps=args.num_steps,
        device=device,
    )
    delta_summary = diff_summaries(base_summary, tuned_summary)

    result = {
        "settings": {
            "base_path": str(args.base_path),
            "tuned_path": str(args.tuned_path),
            "dataset_repo_id": args.dataset_repo_id,
            "dataset_root": str(args.dataset_root),
            "sample_indices": sample_indices,
            "device": device,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "note": "Self-attention excluded. Captures only cross-attention where key_len == prefix_len.",
        },
        "base": base_summary,
        "tuned": tuned_summary,
        "delta": delta_summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print_brief("base", base_summary)
    print_brief("tuned", tuned_summary)
    print(
        "[delta] "
        f"image={delta_summary['overall_delta']['image_mass']:+.4f} "
        f"language={delta_summary['overall_delta']['language_mass']:+.4f} "
        f"state={delta_summary['overall_delta']['state_mass']:+.4f}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
