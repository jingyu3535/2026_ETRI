#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "A_A": "#4C78A8",
    "B_A": "#F58518",
    "B_B": "#E45756",
    "C_A": "#72B7B2",
    "D_A": "#54A24B",
    "camera1": "#D94A4A",
    "camera2": "#2F5BEA",
}

B_MANUAL_OUTCOME_SEQ = {
    "banana": "221222/333232",
    "socks": "111112/112112",
    "strawberry": "223122/222222",
}

P_A_FALLBACK = {
    "camera1": {"image_ratio": 0.286046, "object_ratio": 0.240756},
    "camera2": {"image_ratio": 0.294939, "object_ratio": 0.084992},
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.1,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "axes.grid": False,
            "grid.alpha": 1.0,
            "grid.linewidth": 0.8,
            "grid.color": "#D7DCE3",
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _reset_dir(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def _mkdirs(root: Path) -> tuple[Path, Path]:
    t = root / "tables"
    f = root / "figures"
    t.mkdir(parents=True, exist_ok=True)
    f.mkdir(parents=True, exist_ok=True)
    return t, f


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _save(fig: plt.Figure, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=320, facecolor="white", edgecolor="white")
    plt.close(fig)


def _count_outcomes_from_seq(seq: str) -> dict[int, int]:
    digits = [int(ch) for ch in seq.replace("/", "").strip() if ch.isdigit()]
    return {
        1: sum(1 for d in digits if d == 1),
        2: sum(1 for d in digits if d == 2),
        3: sum(1 for d in digits if d == 3),
        0: len(digits),
    }


def _to_wide_progress(df: pd.DataFrame) -> pd.DataFrame:
    if {"metric", "mean", "se"}.issubset(set(df.columns)):
        wide = (
            df.pivot_table(
                index=["group", "camera", "progress", "progress_pct"],
                columns="metric",
                values=["mean", "se"],
                aggfunc="first",
            )
            .reset_index()
            .copy()
        )
        wide.columns = [
            "_".join([c for c in col if str(c) != ""]).strip("_")
            if isinstance(col, tuple)
            else str(col)
            for col in wide.columns
        ]
        rename = {
            "mean_image_ratio": "image_ratio_mean",
            "se_image_ratio": "image_ratio_se",
            "mean_object_ratio": "object_ratio_mean",
            "se_object_ratio": "object_ratio_se",
            "mean_image_mass": "image_mass_mean",
            "se_image_mass": "image_mass_se",
        }
        for k, v in rename.items():
            if k in wide.columns:
                wide = wide.rename(columns={k: v})
        return wide
    return df


def _build_stage_table_from_t2(table2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cam in ["camera1", "camera2"]:
        aa_ba = table2[(table2["comparison"] == "A@A -> B@A") & (table2["camera"] == cam)]
        ba_bb = table2[(table2["comparison"] == "B@A -> B@B") & (table2["camera"] == cam)]
        if aa_ba.empty or ba_bb.empty:
            continue
        aa_ba = aa_ba.iloc[0]
        ba_bb = ba_bb.iloc[0]
        rows.extend(
            [
                {
                    "camera": cam,
                    "stage": "A@A",
                    "image_ratio": float(aa_ba["base_image_ratio"]),
                    "object_ratio": float(aa_ba["base_object_ratio"]),
                    "object_mass": float(aa_ba["base_object_mass"]),
                },
                {
                    "camera": cam,
                    "stage": "B@A",
                    "image_ratio": float(aa_ba["target_image_ratio"]),
                    "object_ratio": float(aa_ba["target_object_ratio"]),
                    "object_mass": float(aa_ba["target_object_mass"]),
                },
                {
                    "camera": cam,
                    "stage": "B@B",
                    "image_ratio": float(ba_bb["target_image_ratio"]),
                    "object_ratio": float(ba_bb["target_object_ratio"]),
                    "object_mass": float(ba_bb["target_object_mass"]),
                },
            ]
        )
    stage = pd.DataFrame(rows)
    stage["stage"] = pd.Categorical(stage["stage"], categories=["A@A", "B@A", "B@B"], ordered=True)
    return stage.sort_values(["camera", "stage"])


def _build_condition_table_with_pa(stage: pd.DataFrame, results_table_root: Path) -> pd.DataFrame:
    pa_path = results_table_root / "raw_dump_direct_PA_AA_BA_BB_compare.csv"
    pa_map = {}

    if pa_path.exists():
        pa = pd.read_csv(pa_path).set_index("camera")
        for cam in ["camera1", "camera2"]:
            if cam in pa.index:
                pa_map[cam] = {
                    "image_ratio": float(pa.loc[cam, "P@A_img"]),
                    "object_ratio": float(pa.loc[cam, "P@A_obj"]),
                }

    for cam in ["camera1", "camera2"]:
        if cam not in pa_map:
            pa_map[cam] = P_A_FALLBACK[cam]

    rows = []
    for cam in ["camera1", "camera2"]:
        rows.append(
            {
                "camera": cam,
                "condition": "P@A",
                "image_ratio": pa_map[cam]["image_ratio"],
                "object_ratio": pa_map[cam]["object_ratio"],
                "source": "raw_dump_direct_PA_AA_BA_BB_compare.csv" if pa_path.exists() else "fallback_manual",
            }
        )
        sub = stage[stage["camera"] == cam]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "camera": cam,
                    "condition": str(r["stage"]),
                    "image_ratio": float(r["image_ratio"]),
                    "object_ratio": float(r["object_ratio"]),
                    "source": "T2_main_transition",
                }
            )

    out = pd.DataFrame(rows)
    out["condition"] = pd.Categorical(out["condition"], categories=["P@A", "A@A", "B@A", "B@B"], ordered=True)
    return out.sort_values(["camera", "condition"])


def build_tables(eval_root: Path, results_table_root: Path, out_tables: Path) -> dict[str, pd.DataFrame]:
    t1 = pd.read_csv(results_table_root / "T1_fixed_frame_AA_to_BA.csv")
    t2 = pd.read_csv(results_table_root / "T2_onpolicy_BA_to_BB.csv")
    t3 = pd.read_csv(results_table_root / "T3_prelim_AA_to_CA_DA.csv")

    table1 = pd.DataFrame(
        [
            {"model": "A", "train_dataset": "task_box_750", "episodes": 750, "camera_setup": "front+top", "analysis_view": "A@A"},
            {"model": "B", "train_dataset": "task_box_1050", "episodes": 1050, "camera_setup": "front+top", "analysis_view": "B@A, B@B"},
            {"model": "C", "train_dataset": "task_box_1050_toponly", "episodes": 1050, "camera_setup": "top-only", "analysis_view": "C@A"},
            {"model": "D", "train_dataset": "task_box_795", "episodes": 795, "camera_setup": "front+top", "analysis_view": "D@A"},
        ]
    )
    table1.to_csv(out_tables / "Table_01_model_design_abcd.csv", index=False)

    t12 = pd.concat([t1, t2], ignore_index=True)
    t12["comparison_label"] = t12["comparison"].map({"AA_to_BA": "A@A -> B@A", "BA_to_BB": "B@A -> B@B"})
    table2 = t12[
        [
            "comparison_label",
            "camera",
            "base_group",
            "target_group",
            "base_mean_image_ratio",
            "target_mean_image_ratio",
            "delta_pct_image_ratio",
            "base_mean_object_ratio",
            "target_mean_object_ratio",
            "delta_pct_object_ratio",
            "base_mean_object_mass_est",
            "target_mean_object_mass_est",
            "delta_pct_object_mass_est",
        ]
    ].rename(
        columns={
            "comparison_label": "comparison",
            "base_mean_image_ratio": "base_image_ratio",
            "target_mean_image_ratio": "target_image_ratio",
            "delta_pct_image_ratio": "delta_image_ratio_pct",
            "base_mean_object_ratio": "base_object_ratio",
            "target_mean_object_ratio": "target_object_ratio",
            "delta_pct_object_ratio": "delta_object_ratio_pct",
            "base_mean_object_mass_est": "base_object_mass",
            "target_mean_object_mass_est": "target_object_mass",
            "delta_pct_object_mass_est": "delta_object_mass_pct",
        }
    )
    table2.to_csv(out_tables / "Table_02_main_transition_metrics.csv", index=False)

    table3 = t3[
        [
            "comparison",
            "camera",
            "base_group",
            "target_group",
            "base_mean_image_ratio",
            "target_mean_image_ratio",
            "delta_pct_image_ratio",
            "base_mean_object_ratio",
            "target_mean_object_ratio",
            "delta_pct_object_ratio",
        ]
    ].rename(
        columns={
            "base_mean_image_ratio": "base_image_ratio",
            "target_mean_image_ratio": "target_image_ratio",
            "delta_pct_image_ratio": "delta_image_ratio_pct",
            "base_mean_object_ratio": "base_object_ratio",
            "target_mean_object_ratio": "target_object_ratio",
            "delta_pct_object_ratio": "delta_object_ratio_pct",
        }
    )
    table3.to_csv(out_tables / "Table_03_preliminary_cd_metrics.csv", index=False)

    a_outcomes = pd.read_csv(eval_root / "eval_task_box_750_A" / "eval_outcomes.csv")
    by_obj = (
        a_outcomes.groupby(["object", "outcome"], dropna=False)
        .size()
        .reset_index(name="n")
        .pivot(index="object", columns="outcome", values="n")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    for c in [1, 2, 3]:
        if c not in by_obj.columns:
            by_obj[c] = 0
    by_obj = by_obj[["object", 1, 2, 3]]
    by_obj.columns = ["object", "success_n", "partial_n", "fail_n"]
    by_obj["total_n"] = by_obj[["success_n", "partial_n", "fail_n"]].sum(axis=1)
    by_obj["success_rate_pct"] = 100.0 * by_obj["success_n"] / by_obj["total_n"].clip(lower=1)
    by_obj.to_csv(out_tables / "Table_04_A_36ep_outcome_by_object.csv", index=False)

    b_manual_rows = []
    for obj in ["banana", "socks", "strawberry"]:
        cnt = _count_outcomes_from_seq(B_MANUAL_OUTCOME_SEQ[obj])
        b_manual_rows.append(
            {
                "object": obj,
                "outcome_sequence": B_MANUAL_OUTCOME_SEQ[obj],
                "success_n": cnt[1],
                "partial_n": cnt[2],
                "fail_n": cnt[3],
                "total_n": cnt[0],
                "success_rate_pct": 100.0 * cnt[1] / max(cnt[0], 1),
                "source": "manual_input_from_user",
            }
        )
    b_manual = pd.DataFrame(b_manual_rows)
    b_manual.to_csv(out_tables / "Table_06_B_36ep_outcome_by_object_manual.csv", index=False)

    ab = pd.concat(
        [
            by_obj.assign(model="A")[["model", "object", "success_n", "partial_n", "fail_n", "total_n", "success_rate_pct"]],
            b_manual.assign(model="B")[["model", "object", "success_n", "partial_n", "fail_n", "total_n", "success_rate_pct"]],
        ],
        ignore_index=True,
    )
    ab.to_csv(out_tables / "Table_07_A_vs_B_36ep_outcome_by_object.csv", index=False)

    b_ep_level = eval_root / "eval_task_box_1050_B" / "eval_outcomes.csv"
    b_proxy = eval_root / "eval_task_box_1050_B" / "action_attn_dump_img" / "object_ratio_tables_b_only" / "object_ratio_b_by_outcome_summary.csv"
    table5 = pd.DataFrame(
        [
            {
                "model": "A",
                "episode_level_outcome_source": str(eval_root / "eval_task_box_750_A" / "eval_outcomes.csv"),
                "available": (eval_root / "eval_task_box_750_A" / "eval_outcomes.csv").exists(),
                "note": "Used for 36-episode object-level outcome table.",
            },
            {
                "model": "B",
                "episode_level_outcome_source": str(b_ep_level),
                "available": b_ep_level.exists(),
                "note": "Episode-level B outcome csv not found in current filesystem.",
            },
            {
                "model": "B (manual)",
                "episode_level_outcome_source": "Table_06_B_36ep_outcome_by_object_manual.csv",
                "available": True,
                "note": "Manually provided 36-episode object-level outcomes (fallback source).",
            },
            {
                "model": "B (proxy)",
                "episode_level_outcome_source": str(b_proxy),
                "available": b_proxy.exists(),
                "note": "Outcome-stratified attention summary only; not object-level episode outcomes.",
            },
        ]
    )
    table5.to_csv(out_tables / "Table_05_outcome_source_inventory.csv", index=False)

    table8 = _build_stage_table_from_t2(table2)
    table8.to_csv(out_tables / "Table_08_stage_series_AA_BA_BB.csv", index=False)

    table9 = _build_condition_table_with_pa(table8, results_table_root)
    table9.to_csv(out_tables / "Table_09_condition_series_PA_AA_BA_BB.csv", index=False)

    return {"table8": table8, "table9": table9}


def _plot_condition_lines(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, conds: list[str]) -> None:
    x = np.arange(len(conds))
    for cam in ["camera1", "camera2"]:
        sub = df[df["camera"] == cam].set_index("condition").reindex(conds)
        y = sub[metric].to_numpy(dtype=float)
        if np.all(np.isnan(y)):
            continue
        ax.plot(
            x,
            y,
            color=PALETTE[cam],
            linewidth=3.0,
            marker="o",
            markersize=8.5,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=cam,
            zorder=3,
        )
    ax.set_xticks(x, conds, fontsize=11)
    ax.set_title(title, loc="left", pad=10, fontsize=12, fontweight="semibold")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    vals = df[metric].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size > 0:
        pad = (vals.max() - vals.min()) * 0.20 if vals.max() > vals.min() else max(vals.max() * 0.2, 0.02)
        ax.set_ylim(vals.min() - pad * 0.35, vals.max() + pad * 0.30)


def fig_03_main_transition_lines(table8: pd.DataFrame, out: Path) -> None:
    cond = table8.rename(columns={"stage": "condition"}).copy()
    conds = ["A@A", "B@A", "B@B"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharex=True)
    _plot_condition_lines(axes[0], cond, "image_ratio", "Image Ratio", conds)
    _plot_condition_lines(axes[1], cond, "object_ratio", "Object Ratio", conds)
    axes[0].set_ylabel("ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Main Transition (A@A -> B@A -> B@B)", y=1.10, fontsize=14, fontweight="semibold")
    _save(fig, out / "Fig_03_main_transition_lines.png")


def fig_04_condition_trend_pa(table9: pd.DataFrame, out: Path) -> None:
    conds = ["P@A", "A@A", "B@A", "B@B"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharex=True)
    _plot_condition_lines(axes[0], table9, "image_ratio", "Image Ratio", conds)
    _plot_condition_lines(axes[1], table9, "object_ratio", "Object Ratio", conds)
    axes[0].set_ylabel("ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Condition Trend with Pretrained Baseline (P@A)", y=1.10, fontsize=14, fontweight="semibold")
    _save(fig, out / "Fig_04_condition_trend_PA_AA_BA_BB.png")


def _plot_progress_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    camera: str,
    groups: list[str],
    labels: dict[str, str],
) -> None:
    for g in groups:
        sub = df[(df["group"] == g) & (df["camera"] == camera)].sort_values("progress_pct")
        if sub.empty:
            continue
        x = sub["progress_pct"].to_numpy(dtype=float)
        y = sub[f"{metric}_mean"].to_numpy(dtype=float)
        se = sub[f"{metric}_se"].to_numpy(dtype=float)
        ax.plot(x, y, color=PALETTE[g], linewidth=2.6, label=labels.get(g, g))
        ax.fill_between(x, y - se, y + se, color=PALETTE[g], alpha=0.16, linewidth=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Progress (%)")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def fig_05_progress_cd(eval_root: Path, out: Path) -> None:
    abcd = _to_wide_progress(pd.read_csv(eval_root / "timeseries/abcd_progress_layer/tables/lineplot_mean_se_abcd.csv"))
    sub = abcd[(abcd["group"].isin(["A_A", "C_A", "D_A"])) & (abcd["camera"] == "camera2")]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharex=True)
    labels = {"A_A": "A@A", "C_A": "C@A", "D_A": "D@A"}
    groups = ["A_A", "C_A", "D_A"]

    for ax, metric in zip(axes, ["image_ratio", "object_ratio"], strict=True):
        for g in groups:
            gsub = sub[sub["group"] == g].sort_values("progress_pct")
            x = gsub["progress_pct"].to_numpy(dtype=float)
            y = gsub[f"{metric}_mean"].to_numpy(dtype=float)
            se = gsub[f"{metric}_se"].to_numpy(dtype=float)
            ax.plot(x, y, color=PALETTE[g], linewidth=2.6, label=labels[g])
            ax.fill_between(x, y - se, y + se, color=PALETTE[g], alpha=0.16, linewidth=0)
        ax.set_title(metric, loc="left", pad=10, fontsize=12, fontweight="semibold")
        ax.set_xlabel("Progress (%)")
        ax.set_ylabel(metric)
        ax.set_xlim(0, 100)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)

    handles, labels_ = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Top-Camera Progress (Preliminary): A@A vs C@A vs D@A", y=1.11, fontsize=14, fontweight="semibold")
    _save(fig, out / "Fig_05_progress_timeseries_AA_CA_DA_topcam.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Build table/figure assets for A->B->C/D storyline.")
    p.add_argument("--eval_root", default="/home/etri01/paper/eval")
    p.add_argument("--results_table_root", default="/home/etri01/projects/lerobot/results/paper/tables")
    p.add_argument("--out_root", default="/home/etri01/paper/figure/abcd_story")
    p.add_argument("--repo_out_root", default="/home/etri01/projects/lerobot/results/paper/abcd_story")
    args = p.parse_args()

    _style()
    eval_root = Path(args.eval_root)
    results_table_root = Path(args.results_table_root)
    out_root = Path(args.out_root)
    repo_out_root = Path(args.repo_out_root)

    _reset_dir(out_root)
    out_tables, out_figures = _mkdirs(out_root)

    tables = build_tables(eval_root, results_table_root, out_tables)
    fig_03_main_transition_lines(tables["table8"], out_figures)
    fig_04_condition_trend_pa(tables["table9"], out_figures)
    fig_05_progress_cd(eval_root, out_figures)

    _copy_tree(out_root, repo_out_root)
    print(f"[done] wrote assets to: {out_root}")
    print(f"[done] mirrored assets to: {repo_out_root}")


if __name__ == "__main__":
    main()
