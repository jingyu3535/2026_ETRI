#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from PIL import Image


TAG_RE = re.compile(r"ep(\d+)_f(\d+)_action_attn\.npz$")
GROUP_ORDER = ["P_A", "A_A", "B_A", "B_B"]
CAMERA_ORDER = ["camera1", "camera2"]


@dataclass(frozen=True)
class GroupSpec:
    name: str
    dump_dir: Path
    mask_root: Path


@dataclass
class RunningStats:
    n: int = 0
    sum_image_mass: float = 0.0
    sum_image_ratio: float = 0.0
    sum_object_mass: float = 0.0
    sum_object_ratio: float = 0.0
    sum_total_mass: float = 0.0

    def update(
        self,
        image_mass: float,
        image_ratio: float,
        object_mass: float,
        object_ratio: float,
        total_mass: float,
    ) -> None:
        self.n += 1
        self.sum_image_mass += image_mass
        self.sum_image_ratio += image_ratio
        self.sum_object_mass += object_mass
        self.sum_object_ratio += object_ratio
        self.sum_total_mass += total_mass

    def to_means(self) -> dict[str, float | int]:
        if self.n <= 0:
            return {
                "n_valid_frames": 0,
                "image_mass_mean": float("nan"),
                "image_ratio_mean": float("nan"),
                "object_mass_mean": float("nan"),
                "object_ratio_mean": float("nan"),
                "total_mass_mean": float("nan"),
            }
        inv_n = 1.0 / float(self.n)
        return {
            "n_valid_frames": self.n,
            "image_mass_mean": self.sum_image_mass * inv_n,
            "image_ratio_mean": self.sum_image_ratio * inv_n,
            "object_mass_mean": self.sum_object_mass * inv_n,
            "object_ratio_mean": self.sum_object_ratio * inv_n,
            "total_mass_mean": self.sum_total_mass * inv_n,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-aggregate P@A/A@A/B@A/B@B camera means from raw dump files with explicit soft-coverage provenance."
    )
    p.add_argument("--p_dump_dir", default="/home/etri01/paper/eval/eval_task_box_750_P/action_attn_dump_img/_raw_base")
    p.add_argument("--a_dump_dir", default="/home/etri01/paper/eval/eval_task_box_750_A/action_attn_dump_img_all")
    p.add_argument("--ba_dump_dir", default="/home/etri01/paper/eval/eval_task_box_750_B/action_attn_dump_img/_raw_tuned")
    p.add_argument(
        "--bb_dump_dir",
        default="/home/etri01/paper/eval/eval_task_box_1050_B/action_attn_dump_img/_raw_tuned",
    )
    p.add_argument("--pa_mask_root", default="/home/etri01/paper/eval/eval_task_box_750_A/sam2_large")
    p.add_argument("--aa_mask_root", default="/home/etri01/paper/eval/eval_task_box_750_A/sam2_large")
    p.add_argument("--ba_mask_root", default="/home/etri01/paper/eval/eval_task_box_750_A/sam2_large")
    p.add_argument("--bb_mask_root", default="/home/etri01/paper/eval/eval_task_box_1050_B/sam2_large")
    p.add_argument("--mask_mode", choices=["soft", "hard50", "hard0"], default="soft")
    p.add_argument("--mask_pad_size", type=int, default=512)
    p.add_argument("--layer_selection", choices=["odd", "all"], default="odd")
    p.add_argument("--progress_every", type=int, default=2000)
    p.add_argument("--out_dir", default="results/paper/tables")
    return p.parse_args()


def parse_tag(name: str) -> tuple[int, int] | None:
    m = TAG_RE.fullmatch(name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def resize_mask_with_pad(mask_2d: np.ndarray, size: int) -> np.ndarray:
    h, w = mask_2d.shape
    ratio = max(w / size, h / size)
    rh, rw = int(h / ratio), int(w / ratio)
    rh = max(1, rh)
    rw = max(1, rw)
    resized = np.array(Image.fromarray(mask_2d).resize((rw, rh), resample=Image.NEAREST), dtype=np.uint8)
    padded = np.zeros((size, size), dtype=np.uint8)
    ph, pw = max(0, size - rh), max(0, size - rw)
    padded[ph : ph + rh, pw : pw + rw] = resized
    return padded


def build_mask_coverage(
    mask_path: Path,
    token_side: int,
    mask_mode: str,
    pad_size: int,
) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    mask = (np.array(Image.open(mask_path), dtype=np.uint8) > 0).astype(np.uint8)
    if int(mask.max()) <= 0:
        return None

    if pad_size > 0:
        mask = resize_mask_with_pad(mask, pad_size)
    h, w = mask.shape
    if h != w:
        return None
    if h % token_side != 0:
        return None

    block = h // token_side
    cov = (
        mask.reshape(token_side, block, token_side, block)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )

    if mask_mode == "soft":
        out = cov
    elif mask_mode == "hard50":
        out = (cov >= 0.5).astype(np.float32)
    elif mask_mode == "hard0":
        out = (cov > 0.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")

    if float(out.max()) <= 0.0:
        return None
    return out


def select_layer_axis(attn: np.ndarray, layers: np.ndarray, selection: str) -> np.ndarray:
    if selection == "all":
        return attn
    keep = (layers.astype(int) % 2) == 1
    if int(keep.sum()) <= 0:
        return attn
    if attn.ndim == 5:
        return attn[:, keep, :, :, :]
    if attn.ndim == 6:
        return attn[:, keep, :, :, :, :]
    raise ValueError(f"Unsupported attention rank: {attn.ndim}")


def reduce_prefix(attn: np.ndarray) -> np.ndarray:
    if attn.ndim == 5:
        # [step, layer, batch, action, seq]
        return attn.mean(axis=(0, 1, 2, 3), dtype=np.float64)
    if attn.ndim == 6:
        # [step, layer, batch, head, action, seq]
        return attn.mean(axis=(0, 1, 2, 3, 4), dtype=np.float64)
    raise ValueError(f"Unsupported attention rank: {attn.ndim}")


def resolve_spans(meta_path: Path) -> dict[str, tuple[int, int, int]]:
    meta = json.loads(meta_path.read_text())
    image_ranges = meta["image_ranges"]
    if len(image_ranges) < 2:
        raise RuntimeError(f"Need two image ranges for dual-camera aggregation: {meta_path}")
    out: dict[str, tuple[int, int, int]] = {}
    for cam, idx in [("camera1", 0), ("camera2", 1)]:
        start = int(image_ranges[idx]["start"])
        end = int(image_ranges[idx]["end"])
        tok = int(end - start)
        side = int(math.sqrt(tok))
        if side * side != tok:
            raise RuntimeError(f"Non-square token span for {cam} in {meta_path}: {tok}")
        out[cam] = (start, end, side)
    return out


def pct_delta(base: float, target: float) -> float:
    if not np.isfinite(base) or not np.isfinite(target) or abs(base) < 1e-12:
        return float("nan")
    return (target - base) * 100.0 / base


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        GroupSpec("P_A", Path(args.p_dump_dir), Path(args.pa_mask_root)),
        GroupSpec("A_A", Path(args.a_dump_dir), Path(args.aa_mask_root)),
        GroupSpec("B_A", Path(args.ba_dump_dir), Path(args.ba_mask_root)),
        GroupSpec("B_B", Path(args.bb_dump_dir), Path(args.bb_mask_root)),
    ]

    for spec in specs:
        if not spec.dump_dir.exists():
            raise FileNotFoundError(f"Missing dump dir: {spec.dump_dir}")
        if not spec.mask_root.exists():
            raise FileNotFoundError(f"Missing mask root: {spec.mask_root}")

    generated_at_utc = datetime.now(timezone.utc).isoformat()

    long_path = out_dir / "raw_dump_direct_reaggregated_long.csv"
    long_fields = [
        "group",
        "camera",
        "episode",
        "frame0",
        "image_mass",
        "total_mass",
        "image_ratio",
        "object_mass",
        "object_ratio",
        "mask_mode",
        "mask_pad_size",
        "layer_selection",
        "dump_dir",
        "mask_root",
    ]
    with long_path.open("w", newline="") as f_long:
        long_writer = csv.DictWriter(f_long, fieldnames=long_fields)
        long_writer.writeheader()

        # (mask_root, camera, ep, frame0, token_side, mask_mode, pad) -> coverage or False
        cov_cache: Dict[tuple[str, str, int, int, int, str, int], np.ndarray | bool] = {}
        stats: dict[tuple[str, str], RunningStats] = {}
        total_files_by_group: dict[str, int] = {}

        for spec in specs:
            files = sorted(spec.dump_dir.glob("*_action_attn.npz"))
            total_files_by_group[spec.name] = len(files)
            if not files:
                raise RuntimeError(f"No dump npz files found: {spec.dump_dir}")

            spans = resolve_spans(files[0].with_name(files[0].name.replace("_action_attn.npz", "_action_attn_meta.json")))
            print(
                f"[group] {spec.name} files={len(files)} dump_dir={spec.dump_dir} mask_root={spec.mask_root}",
                flush=True,
            )

            for i, npz_path in enumerate(files, start=1):
                tag = parse_tag(npz_path.name)
                if tag is None:
                    continue
                ep, frame0 = tag
                with np.load(npz_path, allow_pickle=False) as d:
                    attn = d["attn"].astype(np.float32)
                    layers = d["layers"].astype(int)
                attn = select_layer_axis(attn, layers, args.layer_selection)
                prefix = reduce_prefix(attn)
                total_mass = float(prefix.sum(dtype=np.float64))

                for camera in CAMERA_ORDER:
                    start, end, side = spans[camera]
                    key = (
                        str(spec.mask_root),
                        camera,
                        ep,
                        frame0,
                        side,
                        args.mask_mode,
                        int(args.mask_pad_size),
                    )
                    if key in cov_cache:
                        cached = cov_cache[key]
                        coverage = None if cached is False else cached
                    else:
                        mask_path = spec.mask_root / camera / f"ep{ep:03d}" / f"frame_{frame0 + 1:05d}.png"
                        coverage = build_mask_coverage(
                            mask_path=mask_path,
                            token_side=side,
                            mask_mode=args.mask_mode,
                            pad_size=int(args.mask_pad_size),
                        )
                        cov_cache[key] = False if coverage is None else coverage

                    if coverage is None:
                        continue

                    img_vec = prefix[start:end]
                    image_mass = float(img_vec.sum(dtype=np.float64))
                    if not np.isfinite(image_mass) or image_mass <= 0.0:
                        continue
                    object_mass = float((img_vec.reshape(side, side) * coverage).sum(dtype=np.float64))
                    image_ratio = float(image_mass / total_mass) if total_mass > 0 else float("nan")
                    object_ratio = float(object_mass / image_mass)
                    if not np.isfinite(object_ratio) or not np.isfinite(image_ratio):
                        continue

                    stat_key = (spec.name, camera)
                    if stat_key not in stats:
                        stats[stat_key] = RunningStats()
                    stats[stat_key].update(
                        image_mass=image_mass,
                        image_ratio=image_ratio,
                        object_mass=object_mass,
                        object_ratio=object_ratio,
                        total_mass=total_mass,
                    )

                    long_writer.writerow(
                        {
                            "group": spec.name,
                            "camera": camera,
                            "episode": ep,
                            "frame0": frame0,
                            "image_mass": image_mass,
                            "total_mass": total_mass,
                            "image_ratio": image_ratio,
                            "object_mass": object_mass,
                            "object_ratio": object_ratio,
                            "mask_mode": args.mask_mode,
                            "mask_pad_size": int(args.mask_pad_size),
                            "layer_selection": args.layer_selection,
                            "dump_dir": str(spec.dump_dir),
                            "mask_root": str(spec.mask_root),
                        }
                    )

                if i % max(1, int(args.progress_every)) == 0:
                    print(f"[group] {spec.name} processed {i}/{len(files)}", flush=True)

    # Aggregate table rows.
    all_rows: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        for camera in CAMERA_ORDER:
            st = stats.get((group, camera), RunningStats())
            means = st.to_means()
            all_rows.append(
                {
                    "group": group,
                    "camera": camera,
                    "n_valid_frames": int(means["n_valid_frames"]),
                    "image_mass_mean": float(means["image_mass_mean"]),
                    "image_ratio_mean": float(means["image_ratio_mean"]),
                    "object_mass_mean": float(means["object_mass_mean"]),
                    "object_ratio_mean": float(means["object_ratio_mean"]),
                    "total_mass_mean": float(means["total_mass_mean"]),
                    "mask_mode": args.mask_mode,
                    "mask_pad_size": int(args.mask_pad_size),
                    "layer_selection": args.layer_selection,
                    "generated_at_utc": generated_at_utc,
                }
            )

    all_path = out_dir / "raw_dump_direct_all_groups_camera_means.csv"
    all_fields = list(all_rows[0].keys())
    write_csv(all_path, all_rows, all_fields)

    # Split per-group camera mean files for compatibility.
    for group in GROUP_ORDER:
        rows = [r for r in all_rows if str(r["group"]) == group]
        short_rows = [
            {
                "group": r["group"],
                "camera": r["camera"],
                "n_valid_frames": r["n_valid_frames"],
                "image_mass_mean": r["image_mass_mean"],
                "object_ratio_mean": r["object_ratio_mean"],
                "mask_mode": r["mask_mode"],
                "mask_pad_size": r["mask_pad_size"],
                "layer_selection": r["layer_selection"],
                "generated_at_utc": r["generated_at_utc"],
            }
            for r in rows
        ]
        write_csv(
            out_dir / f"raw_dump_direct_{group}_camera_means.csv",
            short_rows,
            list(short_rows[0].keys()),
        )

    # Build PA/AA/BA/BB compare table.
    mean_map = {(str(r["group"]), str(r["camera"])): r for r in all_rows}
    compare_rows: list[dict[str, object]] = []
    for camera in CAMERA_ORDER:
        p = mean_map[("P_A", camera)]
        a = mean_map[("A_A", camera)]
        ba = mean_map[("B_A", camera)]
        bb = mean_map[("B_B", camera)]
        compare_rows.append(
            {
                "camera": camera,
                "P@A_img": p["image_mass_mean"],
                "A@A_img": a["image_mass_mean"],
                "B@A_img": ba["image_mass_mean"],
                "B@B_img": bb["image_mass_mean"],
                "AA/PA_img_%": pct_delta(float(p["image_mass_mean"]), float(a["image_mass_mean"])),
                "BA/AA_img_%": pct_delta(float(a["image_mass_mean"]), float(ba["image_mass_mean"])),
                "BB/BA_img_%": pct_delta(float(ba["image_mass_mean"]), float(bb["image_mass_mean"])),
                "P@A_obj": p["object_ratio_mean"],
                "A@A_obj": a["object_ratio_mean"],
                "B@A_obj": ba["object_ratio_mean"],
                "B@B_obj": bb["object_ratio_mean"],
                "AA/PA_obj_%": pct_delta(float(p["object_ratio_mean"]), float(a["object_ratio_mean"])),
                "BA/AA_obj_%": pct_delta(float(a["object_ratio_mean"]), float(ba["object_ratio_mean"])),
                "BB/BA_obj_%": pct_delta(float(ba["object_ratio_mean"]), float(bb["object_ratio_mean"])),
                "mask_mode": args.mask_mode,
                "mask_pad_size": int(args.mask_pad_size),
                "layer_selection": args.layer_selection,
                "generated_at_utc": generated_at_utc,
            }
        )
    compare_path = out_dir / "raw_dump_direct_PA_AA_BA_BB_compare.csv"
    write_csv(compare_path, compare_rows, list(compare_rows[0].keys()))

    # Validation report.
    checks = {
        "all_group_camera_rows_present": len(all_rows) == (len(GROUP_ORDER) * len(CAMERA_ORDER)),
        "all_n_valid_positive": all(int(r["n_valid_frames"]) > 0 for r in all_rows),
        "all_total_mass_finite": all(np.isfinite(float(r["total_mass_mean"])) for r in all_rows),
    }

    pct_match = True
    for row in compare_rows:
        for base_k, target_k, delta_k in [
            ("P@A_img", "A@A_img", "AA/PA_img_%"),
            ("A@A_img", "B@A_img", "BA/AA_img_%"),
            ("B@A_img", "B@B_img", "BB/BA_img_%"),
            ("P@A_obj", "A@A_obj", "AA/PA_obj_%"),
            ("A@A_obj", "B@A_obj", "BA/AA_obj_%"),
            ("B@A_obj", "B@B_obj", "BB/BA_obj_%"),
        ]:
            expected = pct_delta(float(row[base_k]), float(row[target_k]))
            got = float(row[delta_k])
            if not (np.isfinite(expected) and np.isfinite(got) and abs(expected - got) <= 1e-9):
                pct_match = False
                break
    checks["pct_delta_recompute_match"] = pct_match
    checks["all_passed"] = all(bool(v) for v in checks.values())

    report = {
        "generated_at_utc": generated_at_utc,
        "config": {
            "mask_mode": args.mask_mode,
            "mask_pad_size": int(args.mask_pad_size),
            "layer_selection": args.layer_selection,
            "group_order": GROUP_ORDER,
            "camera_order": CAMERA_ORDER,
            "paths": {
                "P_A_dump_dir": str(Path(args.p_dump_dir)),
                "A_A_dump_dir": str(Path(args.a_dump_dir)),
                "B_A_dump_dir": str(Path(args.ba_dump_dir)),
                "B_B_dump_dir": str(Path(args.bb_dump_dir)),
                "P_A_mask_root": str(Path(args.pa_mask_root)),
                "A_A_mask_root": str(Path(args.aa_mask_root)),
                "B_A_mask_root": str(Path(args.ba_mask_root)),
                "B_B_mask_root": str(Path(args.bb_mask_root)),
            },
        },
        "total_files_by_group": total_files_by_group,
        "checks": checks,
        "outputs": {
            "long_csv": str(long_path),
            "all_groups_csv": str(all_path),
            "compare_csv": str(compare_path),
        },
    }
    report_json_path = out_dir / "raw_dump_direct_reaggregation_report.json"
    report_json_path.write_text(json.dumps(report, indent=2))

    md_lines = [
        "# Raw Dump Direct Re-aggregation Report",
        "",
        f"- Generated at (UTC): `{generated_at_utc}`",
        f"- Mask mode: `{args.mask_mode}`",
        f"- Mask pad size: `{int(args.mask_pad_size)}`",
        f"- Layer selection: `{args.layer_selection}`",
        "",
        "## Group file counts",
        "",
        "| Group | Dump files |",
        "| --- | ---: |",
    ]
    for g in GROUP_ORDER:
        md_lines.append(f"| {g} | {int(total_files_by_group.get(g, 0))} |")
    md_lines += [
        "",
        "## Validation checks",
        "",
    ]
    for k, v in checks.items():
        md_lines.append(f"- `{k}`: `{bool(v)}`")
    md_lines += [
        "",
        "## Output files",
        "",
        f"- `{long_path}`",
        f"- `{all_path}`",
        f"- `{compare_path}`",
        f"- `{report_json_path}`",
    ]
    report_md_path = out_dir / "raw_dump_direct_reaggregation_report.md"
    report_md_path.write_text("\n".join(md_lines) + "\n")

    print(f"[done] wrote {long_path}", flush=True)
    print(f"[done] wrote {all_path}", flush=True)
    print(f"[done] wrote {compare_path}", flush=True)
    print(f"[done] wrote {report_json_path}", flush=True)
    print(f"[done] wrote {report_md_path}", flush=True)


if __name__ == "__main__":
    main()
