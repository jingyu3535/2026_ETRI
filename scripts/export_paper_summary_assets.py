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


DUAL_GROUP_ORDER = ["A_A", "B_A", "C_A", "D_A", "B_B"]
TOP_GROUP_ORDER = ["P_A", "A_A", "B_A", "C_A"]
DUAL_METRICS = ["image_mass", "object_mass_est", "image_ratio", "object_ratio"]
PAIR_SPECS = [
    ("AA_to_BA", "A_A", "B_A"),
    ("BA_to_BB", "B_A", "B_B"),
    ("AA_to_CA", "A_A", "C_A"),
    ("AA_to_DA", "A_A", "D_A"),
]

COLOR_GROUP = {
    "P_A": "#4C78A8",
    "A_A": "#54A24B",
    "B_A": "#E45756",
    "C_A": "#B279A2",
    "D_A": "#72B7B2",
    "B_B": "#F58518",
}

COLOR_METRIC = {
    "image_mass": "#4C78A8",
    "object_ratio": "#E45756",
}


def _ordered(values: list[str], order: list[str]) -> list[str]:
    front = [x for x in order if x in values]
    tail = sorted([x for x in values if x not in front])
    return front + tail


def _pct(base: float, target: float) -> float:
    if not np.isfinite(base) or abs(base) < 1e-12:
        return np.nan
    return (target - base) / base * 100.0


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _with_object_mass_est(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _to_float(out, ["image_mass", "object_ratio"])
    out["object_mass_est"] = out["image_mass"] * out["object_ratio"]
    return out


def _summarize_dual(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "image_mass_mean": ("image_mass", "mean"),
        "image_mass_std": ("image_mass", "std"),
        "object_mass_est_mean": ("object_mass_est", "mean"),
        "object_mass_est_std": ("object_mass_est", "std"),
        "image_ratio_mean": ("image_ratio", "mean"),
        "image_ratio_std": ("image_ratio", "std"),
        "object_ratio_mean": ("object_ratio", "mean"),
        "object_ratio_std": ("object_ratio", "std"),
        "n": ("group", "size"),
    }
    out = (
        df.groupby(["group", "camera"], dropna=False)
        .agg(**agg)
        .reset_index()
        .sort_values(["camera", "group"])  # for stable export
        .reset_index(drop=True)
    )
    out["group"] = pd.Categorical(out["group"], categories=_ordered(out["group"].astype(str).tolist(), DUAL_GROUP_ORDER), ordered=True)
    out = out.sort_values(["camera", "group"]).reset_index(drop=True)
    return out


def _pair_delta_long(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    cameras = sorted(summary_df["camera"].dropna().astype(str).unique().tolist())

    for comp, base_g, target_g in PAIR_SPECS:
        for cam in cameras:
            b = summary_df[(summary_df["group"].astype(str) == base_g) & (summary_df["camera"].astype(str) == cam)]
            t = summary_df[(summary_df["group"].astype(str) == target_g) & (summary_df["camera"].astype(str) == cam)]
            if b.empty or t.empty:
                continue
            n_base = int(b.iloc[0]["n"])
            n_target = int(t.iloc[0]["n"])
            for metric in DUAL_METRICS:
                base_mean = float(b.iloc[0][f"{metric}_mean"])
                target_mean = float(t.iloc[0][f"{metric}_mean"])
                rows.append(
                    {
                        "comparison": comp,
                        "camera": cam,
                        "metric": metric,
                        "base_group": base_g,
                        "target_group": target_g,
                        "base_mean": base_mean,
                        "target_mean": target_mean,
                        "delta_abs": target_mean - base_mean,
                        "delta_pct": _pct(base_mean, target_mean),
                        "n_base": n_base,
                        "n_target": n_target,
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["comparison", "camera", "metric"]).reset_index(drop=True)
    return out


def _pair_delta_wide(pair_long: pd.DataFrame, comparison: str) -> pd.DataFrame:
    d = pair_long[pair_long["comparison"] == comparison].copy()
    if d.empty:
        return d

    idx_cols = ["comparison", "camera", "base_group", "target_group", "n_base", "n_target"]
    val_cols = ["base_mean", "target_mean", "delta_abs", "delta_pct"]
    wide = d.pivot_table(index=idx_cols, columns="metric", values=val_cols, aggfunc="first")

    def _flat(col: tuple[str, str]) -> str:
        return f"{col[0]}_{col[1]}"

    wide.columns = [_flat(c) for c in wide.columns]
    wide = wide.reset_index()
    return wide


def _top_group_summary(ratio_df: pd.DataFrame) -> pd.DataFrame:
    d = ratio_df.copy()
    d = d[d["camera"].astype(str) == "camera2"].copy()
    d = d[d["group"].astype(str).isin(TOP_GROUP_ORDER)].copy()

    d["image_mass"] = pd.to_numeric(d.get("image_sum"), errors="coerce")
    d["object_mass"] = pd.to_numeric(d.get("object_sum"), errors="coerce")
    d["image_ratio"] = pd.to_numeric(d.get("image_ratio"), errors="coerce")
    d["object_ratio"] = pd.to_numeric(d.get("object_ratio"), errors="coerce")

    out = (
        d.groupby("group", dropna=False)[["image_mass", "object_mass", "image_ratio", "object_ratio"]]
        .mean()
        .reset_index()
    )
    out["group"] = pd.Categorical(out["group"], categories=TOP_GROUP_ORDER, ordered=True)
    out = out.sort_values("group").reset_index(drop=True)
    return out


def _top_transition_table(top_summary: pd.DataFrame) -> pd.DataFrame:
    vals: dict[str, dict[str, float]] = {}
    for _, r in top_summary.iterrows():
        vals[str(r["group"])] = {
            "image_mass": float(r["image_mass"]),
            "object_mass": float(r["object_mass"]),
            "image_ratio": float(r["image_ratio"]),
            "object_ratio": float(r["object_ratio"]),
        }

    rows = []
    for metric in ["image_mass", "object_mass", "image_ratio", "object_ratio"]:
        p = vals.get("P_A", {}).get(metric, np.nan)
        a = vals.get("A_A", {}).get(metric, np.nan)
        b = vals.get("B_A", {}).get(metric, np.nan)
        c = vals.get("C_A", {}).get(metric, np.nan)
        rows.append(
            {
                "metric": metric,
                "P_A": p,
                "A_A": a,
                "B_A": b,
                "C_A": c,
                "AA_vs_PA_pct": _pct(p, a),
                "BA_vs_AA_pct": _pct(a, b),
                "CA_vs_BA_pct": _pct(b, c),
            }
        )
    return pd.DataFrame(rows)


def _top_outcome_matrix(ratio_df: pd.DataFrame, b_outcome_df: pd.DataFrame) -> pd.DataFrame:
    d = ratio_df.copy()
    d = d[(d["camera"].astype(str) == "camera2") & (d["group"].astype(str).isin(["P_A", "A_A", "B_A"]))]

    d["image_mass"] = pd.to_numeric(d.get("image_sum"), errors="coerce")
    d["object_mass"] = pd.to_numeric(d.get("object_sum"), errors="coerce")
    d["object_ratio"] = pd.to_numeric(d.get("object_ratio"), errors="coerce")

    base = (
        d.groupby(["outcome", "group"], dropna=False)[["image_mass", "object_mass", "object_ratio"]]
        .mean()
        .reset_index()
    )

    b2 = b_outcome_df.copy()
    b2 = b2[b2["camera"].astype(str) == "camera2"].copy()
    b2 = _to_float(b2, ["outcome", "mean_image_mass", "mean_object_mass", "mean_object_ratio"])

    rows = []
    outcomes = sorted(set(base["outcome"].dropna().astype(int).tolist()) | set(b2["outcome"].dropna().astype(int).tolist()))
    for outcome in outcomes:
        sub = base[base["outcome"] == outcome]

        def _val(g: str, col: str) -> float:
            q = sub[sub["group"].astype(str) == g]
            if q.empty:
                return np.nan
            return float(q.iloc[0][col])

        bb = b2[b2["outcome"] == outcome]
        bb_img = float(bb.iloc[0]["mean_image_mass"]) if not bb.empty else np.nan
        bb_objm = float(bb.iloc[0]["mean_object_mass"]) if not bb.empty else np.nan
        bb_objr = float(bb.iloc[0]["mean_object_ratio"]) if not bb.empty else np.nan

        for metric, bb_val, src_col in [
            ("image_mass", bb_img, "image_mass"),
            ("object_mass", bb_objm, "object_mass"),
            ("object_ratio", bb_objr, "object_ratio"),
        ]:
            p = _val("P_A", src_col)
            a = _val("A_A", src_col)
            b = _val("B_A", src_col)
            rows.append(
                {
                    "metric": metric,
                    "outcome": int(outcome),
                    "P_A": p,
                    "A_A": a,
                    "B_A": b,
                    "B_B": bb_val,
                    "AA_vs_PA_pct": _pct(p, a),
                    "BA_vs_AA_pct": _pct(a, b),
                    "BB_vs_BA_pct": _pct(b, bb_val),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["metric", "outcome"]).reset_index(drop=True)
    return out


def _scan_dump_inventory(eval_root: Path) -> pd.DataFrame:
    dump_dirs = {
        "P_A": eval_root / "eval_task_box_750_P/action_attn_dump_img/_raw_base",
        "A_A": eval_root / "eval_task_box_750_A/action_attn_dump_img_all",
        "B_A": eval_root / "eval_task_box_750_B/action_attn_dump_img/_raw_tuned",
        "C_A": eval_root / "eval_task_box_750_C/action_attn_dump_img/_raw_tuned",
        "D_A": eval_root / "eval_task_box_750_D/action_attn_dump_img/_raw_tuned",
    }

    rows: list[dict] = []
    for g in ["P_A", "A_A", "B_A", "C_A", "D_A"]:
        p = dump_dirs[g]
        npz_files = sorted(p.glob("*_action_attn.npz")) if p.exists() else []
        meta_files = sorted(p.glob("*_action_attn_meta.json")) if p.exists() else []
        row = {
            "group": g,
            "dump_dir": str(p),
            "exists": bool(p.exists()),
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
            meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
            row["sample_dump_action_step"] = str(meta.get("dump_action_step", ""))
            row["sample_dump_heads"] = str(meta.get("dump_heads", ""))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["group"] = pd.Categorical(out["group"], categories=["P_A", "A_A", "B_A", "C_A", "D_A"], ordered=True)
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

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
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


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.write_bytes(src.read_bytes())
    return True


def _paper_rc() -> dict:
    # Match common VLA paper style: clean white background, thin gray grid, restrained colors.
    return {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#fbfbfb",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.95,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }


def _save_figure(fig, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=280, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _annotate_bar_values(ax, bars, values: list[float]) -> None:
    for bar, v in zip(bars, values, strict=True):
        if not np.isfinite(v):
            continue
        y_pad = 0.8
        if v >= 0:
            y = v + y_pad
            va = "bottom"
        else:
            y = v - y_pad
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{v:+.1f}%",
            ha="center",
            va=va,
            fontsize=9.5,
            color="#222222",
            zorder=5,
        )


def _comparison_title(comparison: str) -> str:
    mapping = {
        "AA_to_BA": "Fixed-Frame Model Effect (A@A -> B@A)",
        "BA_to_BB": "On-Policy Shift (B@A -> B@B)",
        "AA_to_CA": "Ablation Effect (A@A -> C@A)",
        "AA_to_DA": "Dose Effect (A@A -> D@A)",
    }
    return mapping.get(comparison, comparison)


def _plot_pair_delta(pair_long: pd.DataFrame, comparison: str, out_png: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    d = pair_long[pair_long["comparison"] == comparison].copy()
    if d.empty:
        return

    metrics = ["image_mass", "object_ratio"]
    cameras = ["camera1", "camera2"]
    cam_label = {"camera1": "Wrist", "camera2": "Top"}

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(8.6, 4.9), constrained_layout=True)
        x = np.arange(len(cameras), dtype=float)
        width = 0.34

        # subtle highlight for the first cluster, inspired by common VLA grouped-bar layouts
        ax.axvspan(-0.5, 0.5, color="#efefef", zorder=0)

        all_vals: list[float] = []
        for i, metric in enumerate(metrics):
            vals: list[float] = []
            for cam in cameras:
                q = d[(d["camera"] == cam) & (d["metric"] == metric)]
                vals.append(float(q.iloc[0]["delta_pct"]) if not q.empty else np.nan)
            all_vals.extend([v for v in vals if np.isfinite(v)])
            pos = x + (i - 0.5) * width
            bars = ax.bar(
                pos,
                vals,
                width=width * 0.93,
                label=metric.replace("_", " "),
                color=COLOR_METRIC[metric],
                edgecolor="#3f3f3f",
                linewidth=0.8,
                zorder=3,
            )
            _annotate_bar_values(ax, bars, vals)

        if all_vals:
            y_min = float(np.nanmin(all_vals))
            y_max = float(np.nanmax(all_vals))
            y_pad = max(2.4, 0.18 * max(1e-6, y_max - y_min))
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.axhline(0.0, color="#3a3a3a", linewidth=1.0, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([cam_label[c] for c in cameras])
        ax.set_ylabel("Relative Change (%)")
        ax.set_title(_comparison_title(comparison))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        _save_figure(fig, out_png)


def _plot_ablation_delta(pair_long: pd.DataFrame, out_png: Path, metric: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    d = pair_long[pair_long["comparison"].isin(["AA_to_CA", "AA_to_DA"]) & (pair_long["metric"] == metric)].copy()
    if d.empty:
        return

    cameras = ["camera1", "camera2"]
    comps = ["AA_to_CA", "AA_to_DA"]
    comp_label = {"AA_to_CA": "C@A - A@A", "AA_to_DA": "D@A - A@A"}
    cam_color = {"camera1": "#4C78A8", "camera2": "#E45756"}

    with plt.rc_context(_paper_rc()):
        x = np.arange(len(comps), dtype=float)
        w = 0.34
        fig, ax = plt.subplots(figsize=(7.8, 4.9), constrained_layout=True)
        ax.axvspan(-0.5, 0.5, color="#efefef", zorder=0)

        all_vals: list[float] = []
        for i, cam in enumerate(cameras):
            vals: list[float] = []
            for c in comps:
                q = d[(d["camera"] == cam) & (d["comparison"] == c)]
                vals.append(float(q.iloc[0]["delta_pct"]) if not q.empty else np.nan)
            all_vals.extend([v for v in vals if np.isfinite(v)])
            offs = x + (i - 0.5) * w
            bars = ax.bar(
                offs,
                vals,
                width=w * 0.93,
                label=("Wrist" if cam == "camera1" else "Top"),
                color=cam_color[cam],
                edgecolor="#3f3f3f",
                linewidth=0.8,
                zorder=3,
            )
            _annotate_bar_values(ax, bars, vals)

        if all_vals:
            y_min = float(np.nanmin(all_vals))
            y_max = float(np.nanmax(all_vals))
            y_pad = max(2.2, 0.18 * max(1e-6, y_max - y_min))
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.axhline(0.0, color="#3a3a3a", linewidth=1.0, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([comp_label[c] for c in comps])
        ax.set_ylabel("Relative Change (%)")
        ax.set_title(f"Preliminary Ablation / Dose Delta ({metric})")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15))
        _save_figure(fig, out_png)


def _plot_layer_profile(abcd_df: pd.DataFrame, metric: str, out_png: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    d = abcd_df.copy()
    d = d[d["layer"].isin([1, 3, 5, 7, 9, 11, 13, 15])].copy()
    d = d[d["group"].isin(["A_A", "B_A", "C_A", "D_A"])].copy()
    if d.empty:
        return

    d = _to_float(d, [metric])
    agg = (
        d.groupby(["camera", "group", "layer"], dropna=False)[metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    agg["se"] = agg["std"] / np.sqrt(np.maximum(agg["n"], 1))

    cameras = ["camera1", "camera2"]
    groups = ["A_A", "B_A", "C_A", "D_A"]
    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), constrained_layout=True)
        for ax, cam in zip(axes, cameras, strict=True):
            cur = agg[agg["camera"] == cam]
            for g in groups:
                gdf = cur[cur["group"] == g].sort_values("layer")
                if gdf.empty:
                    continue
                x = gdf["layer"].to_numpy(dtype=float)
                y = gdf["mean"].to_numpy(dtype=float)
                se = gdf["se"].fillna(0.0).to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2.0,
                    markersize=4.2,
                    color=COLOR_GROUP[g],
                    label=g,
                    zorder=3,
                )
                ax.fill_between(x, y - se, y + se, color=COLOR_GROUP[g], alpha=0.13, linewidth=0.0, zorder=2)

            ax.set_title("Wrist camera" if cam == "camera1" else "Top camera")
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric)
            ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
            ax.grid(axis="y")
            ax.grid(axis="x", visible=False)

        merged: dict[str, object] = {}
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l, strict=True):
                if ll not in merged:
                    merged[ll] = hh
        fig.legend(
            list(merged.values()),
            list(merged.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=4,
            frameon=False,
        )
        fig.suptitle(f"Layer Profile ({metric})", y=1.11)
        _save_figure(fig, out_png)


def _plot_top_group(top_summary: pd.DataFrame, metric: str, out_png: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    if top_summary.empty:
        return

    d = top_summary.copy()
    d = d[d["group"].isin(TOP_GROUP_ORDER)].copy()
    d["group"] = pd.Categorical(d["group"], categories=TOP_GROUP_ORDER, ordered=True)
    d = d.sort_values("group")

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
        x = np.arange(len(d), dtype=float)
        y = d[metric].to_numpy(dtype=float)
        bar_colors = [COLOR_GROUP[g] for g in d["group"].astype(str).tolist()]

        bars = ax.bar(
            x,
            y,
            width=0.62,
            color=bar_colors,
            edgecolor="#3f3f3f",
            linewidth=0.8,
            zorder=3,
        )
        ax.axvspan(-0.5, 0.5, color="#efefef", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(d["group"].astype(str).tolist())
        ax.set_ylabel(metric)
        ax.set_title(f"Top-Camera Group Means ({metric})")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        for bar, v in zip(bars, y, strict=True):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    v + max(1e-4, 0.015 * float(np.nanmax(y))),
                    f"{v:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.2,
                )
        _save_figure(fig, out_png)


def _hypothesis_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hypothesis_id": "H1",
                "hypothesis": "Model change alone (A@A->B@A) reduces object grounding on fixed A frames.",
                "primary_comparison": "AA_to_BA",
                "primary_metrics": "object_ratio,image_mass",
                "primary_tables": "T1_fixed_frame_AA_to_BA.csv",
                "primary_figures": "G1_fixed_frame_AA_to_BA_pct.png",
                "status": "analyzed",
            },
            {
                "hypothesis_id": "H2",
                "hypothesis": "On-policy B rollouts (B@A->B@B) induce view-allocation shift and further grounding drop.",
                "primary_comparison": "BA_to_BB",
                "primary_metrics": "image_mass,object_ratio",
                "primary_tables": "T2_onpolicy_BA_to_BB.csv",
                "primary_figures": "G2_onpolicy_BA_to_BB_pct.png",
                "status": "analyzed",
            },
            {
                "hypothesis_id": "H3",
                "hypothesis": "Wrist-camera removal (C) should mitigate shortcut-related degradation.",
                "primary_comparison": "AA_to_CA",
                "primary_metrics": "object_ratio,image_mass",
                "primary_tables": "T3_prelim_AA_to_CA_DA.csv",
                "primary_figures": "G3_prelim_ablation_object_ratio.png",
                "status": "preliminary",
            },
            {
                "hypothesis_id": "H4",
                "hypothesis": "Dose effect is nonlinear; +45 (D) differs from +300 (B).",
                "primary_comparison": "AA_to_DA",
                "primary_metrics": "object_ratio,image_mass",
                "primary_tables": "T3_prelim_AA_to_CA_DA.csv",
                "primary_figures": "G4_prelim_dose_image_mass.png",
                "status": "preliminary",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready hypothesis tables and figures.")
    parser.add_argument(
        "--paper_eval_root",
        default="/home/etri01/paper/eval",
        help="Root directory containing eval outputs (timeseries, hetmap, dump dirs).",
    )
    parser.add_argument(
        "--out_root",
        default="/home/etri01/projects/lerobot/results/paper",
        help="Output root for tables/figures.",
    )
    args = parser.parse_args()

    eval_root = Path(args.paper_eval_root)
    out_root = Path(args.out_root)
    out_tables = out_root / "tables"
    out_figures = out_root / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    paths = {
        "abcd_raw": eval_root / "timeseries/abcd_progress_layer/tables/raw_frame_metrics_layer_progress_abcd.csv",
        "aa_bb_long": eval_root / "timeseries/new_aa_vs_bb/tables/frame_metrics_long_aa_vs_bb.csv",
        "ratio_long": eval_root / "hetmap/metrics/ratios_all_eps_long.csv",
        "b_outcome": eval_root
        / "eval_task_box_1050_B/action_attn_dump_img/object_ratio_tables_b_only/object_ratio_b_by_outcome_summary.csv",
        "a_outcomes": eval_root / "eval_task_box_750_A/eval_outcomes.csv",
        "metrics_root": eval_root / "hetmap/metrics",
    }

    abcd_raw = _safe_read_csv(paths["abcd_raw"])
    aa_bb_long = _safe_read_csv(paths["aa_bb_long"])
    ratio_long = _safe_read_csv(paths["ratio_long"])
    b_outcome = _safe_read_csv(paths["b_outcome"])
    a_outcomes = _safe_read_csv(paths["a_outcomes"])

    abcd_raw = _to_float(abcd_raw, ["layer", "image_mass", "image_ratio", "object_ratio"])
    aa_bb_long = _to_float(aa_bb_long, ["image_mass", "image_ratio", "object_ratio"])

    abcd_global = _with_object_mass_est(abcd_raw[abcd_raw["layer"] == -1].copy())
    bb_global = _with_object_mass_est(aa_bb_long[aa_bb_long["group"].astype(str) == "B_B"].copy())

    dual_for_summary = pd.concat([abcd_global, bb_global], ignore_index=True)
    dual_summary = _summarize_dual(dual_for_summary)
    pair_long = _pair_delta_long(dual_summary)

    # Tables
    _hypothesis_table().to_csv(out_tables / "hypothesis_test_map.csv", index=False)
    dual_summary.to_csv(out_tables / "T0_dual_camera_group_means.csv", index=False)

    _pair_delta_wide(pair_long, "AA_to_BA").to_csv(out_tables / "T1_fixed_frame_AA_to_BA.csv", index=False)
    _pair_delta_wide(pair_long, "BA_to_BB").to_csv(out_tables / "T2_onpolicy_BA_to_BB.csv", index=False)
    pd.concat(
        [_pair_delta_wide(pair_long, "AA_to_CA"), _pair_delta_wide(pair_long, "AA_to_DA")],
        ignore_index=True,
    ).to_csv(out_tables / "T3_prelim_AA_to_CA_DA.csv", index=False)

    top_summary = _top_group_summary(ratio_long)
    top_summary.to_csv(out_tables / "T4_topcam_group_means_PA_AA_BA_CA.csv", index=False)
    _top_transition_table(top_summary).to_csv(out_tables / "T4_topcam_transition_matrix_PA_AA_BA_CA.csv", index=False)
    _top_outcome_matrix(ratio_long, b_outcome).to_csv(out_tables / "T5_topcam_outcome_matrix_PA_AA_BA_BB.csv", index=False)

    b_outcome.to_csv(out_tables / "T6_b_only_outcome_summary.csv", index=False)

    out_counts = a_outcomes.groupby("outcome", dropna=False).size().reset_index(name="n_episodes").sort_values("outcome")
    out_counts.to_csv(out_tables / "aa_outcome_counts.csv", index=False)

    _scan_dump_inventory(eval_root).to_csv(out_tables / "dump_inventory_summary.csv", index=False)
    _build_mask_quality(paths["metrics_root"]).to_csv(out_tables / "aa_mask_quality_summary.csv", index=False)
    pair_long.to_csv(out_tables / "pair_delta_long_all.csv", index=False)

    sig_src = paths["metrics_root"] / "outcome_soft_analysis/A_A/overall_significance_with_ci.csv"
    if sig_src.exists():
        _safe_read_csv(sig_src).to_csv(out_tables / "aa_outcome_significance_overall.csv", index=False)

    # Figures
    _plot_pair_delta(pair_long, "AA_to_BA", out_figures / "G1_fixed_frame_AA_to_BA_pct.png")
    _plot_pair_delta(pair_long, "BA_to_BB", out_figures / "G2_onpolicy_BA_to_BB_pct.png")
    _plot_ablation_delta(pair_long, out_figures / "G3_prelim_ablation_object_ratio.png", metric="object_ratio")
    _plot_ablation_delta(pair_long, out_figures / "G4_prelim_dose_image_mass.png", metric="image_mass")
    _plot_layer_profile(abcd_raw, "object_ratio", out_figures / "G5_layer_profile_object_ratio_abcd.png")
    _plot_layer_profile(abcd_raw, "image_mass", out_figures / "G6_layer_profile_image_mass_abcd.png")
    _plot_top_group(top_summary, "object_ratio", out_figures / "G7_topcam_object_ratio_groups.png")
    _plot_top_group(top_summary, "image_ratio", out_figures / "G8_topcam_image_ratio_groups.png")

    copied = []
    ref_obj = paths["metrics_root"] / "ratio_36ep_mean/plots/object_ratio_4panel_7x8.png"
    if _copy_if_exists(ref_obj, out_figures / "reference_object_ratio_4panel_7x8.png"):
        copied.append(str(ref_obj))
    ref_img = paths["metrics_root"] / "ratio_36ep_mean/plots/image_ratio_4panel_7x8.png"
    if _copy_if_exists(ref_img, out_figures / "reference_image_ratio_4panel_7x8.png"):
        copied.append(str(ref_img))

    summary = {
        "paper_eval_root": str(eval_root),
        "matplotlib_available": HAS_MATPLOTLIB,
        "tables": sorted(p.name for p in out_tables.glob("*.csv")),
        "figures": sorted(p.name for p in out_figures.glob("*.png")),
        "copied_reference_plots": copied,
        "source_files": {k: str(v) for k, v in paths.items()},
    }
    (out_tables / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[done] exported paper result assets")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
