#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    plt = None


GROUP_ORDER = ["P_A", "A_A", "B_A", "C_A", "D_A"]
METRICS = ["object_ratio", "image_ratio"]
DUMP_DIRS = {
    "P_A": "/home/etri01/논문/eval/eval_task_box_750_P/action_attn_dump_img/_raw_base",
    "A_A": "/home/etri01/논문/eval/eval_task_box_750_A/action_attn_dump_img_all",
    "B_A": "/home/etri01/논문/eval/eval_task_box_750_B/action_attn_dump_img/_raw_tuned",
    "C_A": "/home/etri01/논문/eval/eval_task_box_750_C/action_attn_dump_img/_raw_tuned",
    "D_A": "/home/etri01/논문/eval/eval_task_box_750_D/action_attn_dump_img/_raw_tuned",
}


def _ordered_groups(groups: list[str]) -> list[str]:
    ordered = [g for g in GROUP_ORDER if g in groups]
    rest = sorted([g for g in groups if g not in ordered])
    return ordered + rest


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _build_metric_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows: list[pd.DataFrame] = []
    layer_rows: list[pd.DataFrame] = []

    for metric in METRICS:
        usable = df.dropna(subset=[metric]).copy()

        global_stat = (
            usable.groupby("group")[metric]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
            .rename(columns={"group": "group"})
        )
        global_stat.insert(0, "metric", metric)
        global_rows.append(global_stat)

        layer_stat = (
            usable.groupby(["group", "layer"])[metric]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )
        layer_stat.insert(0, "metric", metric)
        layer_rows.append(layer_stat)

    global_df = pd.concat(global_rows, ignore_index=True)
    layer_df = pd.concat(layer_rows, ignore_index=True)

    global_df["group"] = pd.Categorical(global_df["group"], categories=_ordered_groups(global_df["group"].unique().tolist()), ordered=True)
    global_df = global_df.sort_values(["metric", "group"]).reset_index(drop=True)

    layer_df["group"] = pd.Categorical(layer_df["group"], categories=_ordered_groups(layer_df["group"].unique().tolist()), ordered=True)
    layer_df = layer_df.sort_values(["metric", "group", "layer"]).reset_index(drop=True)

    return global_df, layer_df


def _build_outcome_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aa = df[df["group"] == "A_A"].copy()

    ep_base = aa[["episode", "object", "trial", "outcome"]].drop_duplicates()
    counts = ep_base.groupby(["object", "outcome"]).size().reset_index(name="n_episodes")
    counts = counts.sort_values(["object", "outcome"]).reset_index(drop=True)

    rows: list[pd.DataFrame] = []
    for metric in METRICS:
        stat = (
            aa.dropna(subset=[metric])
            .groupby(["outcome"])[metric]
            .agg(mean="mean", std="std", median="median", n="count")
            .reset_index()
        )
        stat.insert(0, "metric", metric)
        rows.append(stat)
    metric_summary = pd.concat(rows, ignore_index=True)
    metric_summary = metric_summary.sort_values(["metric", "outcome"]).reset_index(drop=True)

    return counts, metric_summary


def _scan_dump_inventory() -> pd.DataFrame:
    rows: list[dict] = []

    for group, path_str in DUMP_DIRS.items():
        p = Path(path_str)
        npz_files = sorted(p.glob("*_action_attn.npz")) if p.exists() else []
        meta_files = sorted(p.glob("*_action_attn_meta.json")) if p.exists() else []

        row = {
            "group": group,
            "dump_dir": str(p),
            "exists": p.exists(),
            "npz_count": len(npz_files),
            "meta_count": len(meta_files),
            "sample_attn_shape": "",
            "sample_layers": "",
            "sample_steps": "",
            "sample_dump_action_step": "",
            "sample_dump_heads": "",
        }

        if npz_files:
            with np.load(npz_files[0]) as data:
                if "attn" in data:
                    row["sample_attn_shape"] = str(tuple(int(x) for x in data["attn"].shape))
                if "layers" in data:
                    row["sample_layers"] = ",".join(str(int(x)) for x in data["layers"].tolist())
                if "steps" in data:
                    row["sample_steps"] = ",".join(str(int(x)) for x in data["steps"].tolist())

        if meta_files:
            meta = json.loads(meta_files[0].read_text())
            row["sample_dump_action_step"] = str(meta.get("dump_action_step", ""))
            row["sample_dump_heads"] = str(meta.get("dump_heads", ""))

        rows.append(row)

    out = pd.DataFrame(rows)
    out["group"] = pd.Categorical(out["group"], categories=_ordered_groups(out["group"].tolist()), ordered=True)
    out = out.sort_values("group").reset_index(drop=True)
    return out


def _build_mask_quality(metrics_root: Path) -> pd.DataFrame:
    summary_path = metrics_root / "server_style_a_a_summary.json"
    if not summary_path.exists():
        return pd.DataFrame(
            [
                {
                    "group": "A_A",
                    "total_frames": np.nan,
                    "nonempty_frames": np.nan,
                    "empty_frames": np.nan,
                    "empty_ratio": np.nan,
                    "source": "missing server_style_a_a_summary.json",
                }
            ]
        )

    summary = json.loads(summary_path.read_text())
    total = int(summary.get("total_frames", 0))
    nonempty = int(summary.get("nonempty_frames", 0))
    empty = max(total - nonempty, 0)
    ratio = (empty / total) if total > 0 else np.nan

    return pd.DataFrame(
        [
            {
                "group": "A_A",
                "total_frames": total,
                "nonempty_frames": nonempty,
                "empty_frames": empty,
                "empty_ratio": ratio,
                "source": str(summary_path),
            }
        ]
    )


def _plot_layer_curve(layer_df: pd.DataFrame, metric: str, out_png: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    d = layer_df[layer_df["metric"] == metric].copy()
    if d.empty:
        return

    groups = _ordered_groups(d["group"].astype(str).unique().tolist())
    plt.figure(figsize=(7.2, 4.5))
    for g in groups:
        gdf = d[d["group"].astype(str) == g].sort_values("layer")
        if gdf.empty:
            continue
        plt.plot(gdf["layer"], gdf["mean"], marker="o", linewidth=1.8, markersize=4.5, label=g)

    plt.xlabel("Layer")
    plt.ylabel(metric)
    plt.title(f"Layer-wise {metric} by group")
    plt.grid(alpha=0.25, linewidth=0.7)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _plot_outcome_bar(outcome_metric_df: pd.DataFrame, metric: str, out_png: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    d = outcome_metric_df[outcome_metric_df["metric"] == metric].copy()
    if d.empty:
        return

    d = d.sort_values("outcome")
    x = np.arange(len(d))
    y = d["mean"].to_numpy(dtype=float)
    yerr = d["std"].fillna(0.0).to_numpy(dtype=float)

    plt.figure(figsize=(6.2, 4.2))
    plt.bar(x, y, yerr=yerr, capsize=4, width=0.6)
    plt.xticks(x, [f"Outcome {int(o)}" for o in d["outcome"].tolist()])
    plt.ylabel(metric)
    plt.title(f"A_A outcome summary ({metric})")
    plt.grid(axis="y", alpha=0.25, linewidth=0.7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.write_bytes(src.read_bytes())
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact paper summary tables/figures.")
    parser.add_argument(
        "--metrics_root",
        default="/home/etri01/논문/eval/hetmap/metrics",
        help="Root directory containing attention ratio/statistics files.",
    )
    parser.add_argument(
        "--out_root",
        default="/home/etri01/projects/lerobot/results/paper",
        help="Output root (tables/ and figures/ are created under this path).",
    )
    args = parser.parse_args()

    metrics_root = Path(args.metrics_root)
    out_root = Path(args.out_root)
    out_tables = out_root / "tables"
    out_figures = out_root / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    ratios_csv = metrics_root / "ratios_all_eps_long.csv"
    ratio_df = _safe_read_csv(ratios_csv)

    global_df, layer_df = _build_metric_tables(ratio_df)
    global_df.to_csv(out_tables / "abcd_global_metric_summary.csv", index=False)
    layer_df.to_csv(out_tables / "abcd_layer_metric_summary.csv", index=False)

    outcome_counts_df, outcome_metric_df = _build_outcome_tables(ratio_df)
    outcome_counts_df.to_csv(out_tables / "aa_outcome_counts.csv", index=False)
    outcome_metric_df.to_csv(out_tables / "aa_outcome_metric_summary.csv", index=False)

    inv_df = _scan_dump_inventory()
    inv_df.to_csv(out_tables / "dump_inventory_summary.csv", index=False)

    mask_quality_df = _build_mask_quality(metrics_root)
    mask_quality_df.to_csv(out_tables / "aa_mask_quality_summary.csv", index=False)

    sig_src = metrics_root / "outcome_soft_analysis" / "A_A" / "overall_significance_with_ci.csv"
    if sig_src.exists():
        sig_df = pd.read_csv(sig_src)
        sig_df.to_csv(out_tables / "aa_outcome_significance_overall.csv", index=False)

    _plot_layer_curve(layer_df, "object_ratio", out_figures / "abcd_layer_object_ratio.png")
    _plot_layer_curve(layer_df, "image_ratio", out_figures / "abcd_layer_image_ratio.png")
    _plot_outcome_bar(outcome_metric_df, "object_ratio", out_figures / "aa_outcome_object_ratio.png")
    _plot_outcome_bar(outcome_metric_df, "image_ratio", out_figures / "aa_outcome_image_ratio.png")

    copied = []
    ref_plot_src = metrics_root / "ratio_36ep_mean" / "plots" / "object_ratio_4panel_7x8.png"
    if _copy_if_exists(ref_plot_src, out_figures / "reference_object_ratio_4panel_7x8.png"):
        copied.append(str(ref_plot_src))
    ref_plot_src = metrics_root / "ratio_36ep_mean" / "plots" / "image_ratio_4panel_7x8.png"
    if _copy_if_exists(ref_plot_src, out_figures / "reference_image_ratio_4panel_7x8.png"):
        copied.append(str(ref_plot_src))

    summary = {
        "metrics_root": str(metrics_root),
        "ratios_rows": int(len(ratio_df)),
        "groups_in_ratio": sorted(ratio_df["group"].dropna().unique().tolist()),
        "layers_in_ratio": sorted(int(x) for x in ratio_df["layer"].dropna().unique().tolist()),
        "matplotlib_available": HAS_MATPLOTLIB,
        "tables": sorted(p.name for p in out_tables.glob("*.csv")),
        "figures": sorted(p.name for p in out_figures.glob("*.png")),
        "copied_reference_plots": copied,
    }
    (out_tables / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[done] exported paper summary assets")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
