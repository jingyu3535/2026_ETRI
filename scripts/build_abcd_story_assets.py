#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


PALETTE = {
    "A_A": "#4C78A8",
    "B_A": "#F58518",
    "B_B": "#E45756",
    "C_A": "#72B7B2",
    "D_A": "#54A24B",
    "camera1": "#B279A2",
    "camera2": "#4C78A8",
    "outcome_1": "#54A24B",
    "outcome_2": "#F2CF5B",
    "outcome_3": "#E45756",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


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


def build_tables(
    eval_root: Path,
    results_table_root: Path,
    out_tables: Path,
) -> dict[str, pd.DataFrame]:
    t1 = pd.read_csv(results_table_root / "T1_fixed_frame_AA_to_BA.csv")
    t2 = pd.read_csv(results_table_root / "T2_onpolicy_BA_to_BB.csv")
    t3 = pd.read_csv(results_table_root / "T3_prelim_AA_to_CA_DA.csv")

    table1 = pd.DataFrame(
        [
            {
                "model": "A",
                "train_dataset": "task_box_750",
                "episodes": 750,
                "camera_setup": "front+top (camera1+camera2)",
                "role": "baseline fine-tuned model",
                "analysis_view": "A@A",
            },
            {
                "model": "B",
                "train_dataset": "task_box_1050",
                "episodes": 1050,
                "camera_setup": "front+top (camera1+camera2)",
                "role": "A + near-phase +300 episodes",
                "analysis_view": "B@A, B@B",
            },
            {
                "model": "C",
                "train_dataset": "task_box_1050_toponly",
                "episodes": 1050,
                "camera_setup": "top-only (wrist removed)",
                "role": "wrist-shortcut ablation",
                "analysis_view": "C@A",
            },
            {
                "model": "D",
                "train_dataset": "task_box_795",
                "episodes": 795,
                "camera_setup": "front+top (camera1+camera2)",
                "role": "dose-control (+45 near-phase episodes)",
                "analysis_view": "D@A",
            },
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
    ].copy()
    table2 = table2.rename(
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
                "model": "B (proxy)",
                "episode_level_outcome_source": str(b_proxy),
                "available": b_proxy.exists(),
                "note": "Outcome-stratified attention summary exists, but not object-level episode outcomes.",
            },
        ]
    )
    table5.to_csv(out_tables / "Table_05_outcome_source_inventory.csv", index=False)

    return {
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "table4": by_obj,
        "table5": table5,
    }


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig_01_flow(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")

    boxes = [
        ("A", "Model A\n(task_box_750)", 0.06, 0.35, PALETTE["A_A"]),
        ("B", "Model B\n(+300 near-phase)", 0.31, 0.35, PALETTE["B_A"]),
        ("C", "Model C\n(top-only ablation)", 0.60, 0.58, PALETTE["C_A"]),
        ("D", "Model D\n(+45 dose control)", 0.60, 0.12, PALETTE["D_A"]),
    ]
    w, h = 0.22, 0.28

    for _, label, x, y, color in boxes:
        patch = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.2,
            edgecolor=color,
            facecolor="#FFFFFF",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, color="#222222")

    arrow_kw = dict(arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color="#444444")
    ax.add_patch(patches.FancyArrowPatch((0.28, 0.49), (0.31, 0.49), **arrow_kw))
    ax.add_patch(patches.FancyArrowPatch((0.53, 0.49), (0.60, 0.69), **arrow_kw))
    ax.add_patch(patches.FancyArrowPatch((0.53, 0.49), (0.60, 0.26), **arrow_kw))

    ax.text(0.435, 0.56, "extra near-phase data", ha="center", va="center", fontsize=10, color="#555555")
    ax.text(0.86, 0.69, "C@A\n(wrist shortcut ablation)", ha="left", va="center", fontsize=10, color="#444444")
    ax.text(0.86, 0.26, "D@A\n(dose nonlinearity check)", ha="left", va="center", fontsize=10, color="#444444")
    ax.set_title("Experiment Flow: A -> B -> {C, D}", fontsize=14, pad=8)

    _save(fig, out / "Fig_01_experiment_flow_abcd.png")


def fig_02_a_outcomes(table4: pd.DataFrame, out: Path) -> None:
    df = table4.copy()
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.2, 4.5))

    b1 = ax.bar(x, df["success_n"], color=PALETTE["outcome_1"], label="Success (outcome=1)")
    b2 = ax.bar(x, df["partial_n"], bottom=df["success_n"], color=PALETTE["outcome_2"], label="Partial (outcome=2)")
    b3 = ax.bar(
        x,
        df["fail_n"],
        bottom=df["success_n"] + df["partial_n"],
        color=PALETTE["outcome_3"],
        label="Fail (outcome=3)",
    )

    for i, r in enumerate(df["success_rate_pct"]):
        ax.text(i, df.loc[i, "total_n"] + 0.35, f"{r:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x, df["object"].tolist())
    ax.set_ylabel("Episodes (n)")
    ax.set_ylim(0, max(df["total_n"]) + 2.5)
    ax.set_title("A Model: 36-Episode Outcome by Object")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.20))

    _save(fig, out / "Fig_02_A_36ep_outcome_by_object.png")


def fig_03_transition(table2: pd.DataFrame, out: Path) -> None:
    transitions = ["A@A -> B@A", "B@A -> B@B"]
    cams = ["camera1", "camera2"]
    width = 0.34
    x = np.arange(len(transitions))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharex=True)
    metrics = [
        ("delta_image_ratio_pct", "Image Ratio Delta (%)"),
        ("delta_object_ratio_pct", "Object Ratio Delta (%)"),
    ]

    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for i, cam in enumerate(cams):
            vals = []
            for tr in transitions:
                v = table2.loc[(table2["comparison"] == tr) & (table2["camera"] == cam), metric]
                vals.append(float(v.iloc[0]) if not v.empty else np.nan)
            xpos = x + (i - 0.5) * width
            bars = ax.bar(xpos, vals, width=width, color=PALETTE[cam], label=cam)
            for b in bars:
                y = b.get_height()
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    y + (1.1 if y >= 0 else -1.8),
                    f"{y:+.1f}",
                    ha="center",
                    va="bottom" if y >= 0 else "top",
                    fontsize=9,
                )
        ax.axhline(0.0, color="#555555", linewidth=1.0)
        ax.set_title(title)
        ax.set_xticks(x, transitions)
        ax.set_ylabel("Delta (%)")

    axes[1].legend(loc="upper right")
    fig.suptitle("Main Transition Effects by Camera", y=1.04, fontsize=14)
    _save(fig, out / "Fig_03_main_transition_deltas.png")


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
        ax.plot(x, y, color=PALETTE[g], linewidth=2.0, label=labels.get(g, g))
        ax.fill_between(x, y - se, y + se, color=PALETTE[g], alpha=0.16, linewidth=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Progress (%)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{camera}")


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


def fig_04_progress_main(eval_root: Path, out: Path) -> None:
    abcd = _to_wide_progress(pd.read_csv(eval_root / "timeseries/abcd_progress_layer/tables/lineplot_mean_se_abcd.csv"))
    aabb = pd.read_csv(eval_root / "timeseries/new_aa_vs_bb/tables/aggregate_mean_se_aa_vs_bb.csv")

    use_abcd = abcd[abcd["group"].isin(["A_A", "B_A"])]
    use_aabb = aabb[aabb["group"].isin(["B_B"])]
    data = pd.concat([use_abcd, use_aabb], ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True)
    labels = {"A_A": "A@A", "B_A": "B@A", "B_B": "B@B"}
    groups = ["A_A", "B_A", "B_B"]

    for c_idx, cam in enumerate(["camera1", "camera2"]):
        _plot_progress_panel(axes[0, c_idx], data, "image_ratio", cam, groups, labels)
        _plot_progress_panel(axes[1, c_idx], data, "object_ratio", cam, groups, labels)

    axes[0, 0].legend(loc="upper left")
    axes[0, 0].set_title("camera1 (wrist)")
    axes[0, 1].set_title("camera2 (top)")
    axes[0, 0].set_ylabel("image_ratio")
    axes[1, 0].set_ylabel("object_ratio")

    fig.suptitle("Progress Time-Series: A@A vs B@A vs B@B", y=1.01, fontsize=14)
    _save(fig, out / "Fig_04_progress_timeseries_AA_BA_BB.png")


def fig_05_progress_cd(eval_root: Path, out: Path) -> None:
    abcd = _to_wide_progress(pd.read_csv(eval_root / "timeseries/abcd_progress_layer/tables/lineplot_mean_se_abcd.csv"))
    sub = abcd[(abcd["group"].isin(["A_A", "C_A", "D_A"])) & (abcd["camera"] == "camera2")]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), sharex=True)
    labels = {"A_A": "A@A", "C_A": "C@A", "D_A": "D@A"}
    groups = ["A_A", "C_A", "D_A"]

    for ax, metric in zip(axes, ["image_ratio", "object_ratio"], strict=True):
        for g in groups:
            gsub = sub[sub["group"] == g].sort_values("progress_pct")
            x = gsub["progress_pct"].to_numpy(dtype=float)
            y = gsub[f"{metric}_mean"].to_numpy(dtype=float)
            se = gsub[f"{metric}_se"].to_numpy(dtype=float)
            ax.plot(x, y, color=PALETTE[g], linewidth=2.0, label=labels[g])
            ax.fill_between(x, y - se, y + se, color=PALETTE[g], alpha=0.16, linewidth=0)
        ax.set_title(metric)
        ax.set_xlabel("Progress (%)")
        ax.set_ylabel(metric)
        ax.set_xlim(0, 100)

    axes[0].legend(loc="upper right")
    fig.suptitle("Top-Camera Progress (Preliminary): A@A vs C@A vs D@A", y=1.03, fontsize=14)
    _save(fig, out / "Fig_05_progress_timeseries_AA_CA_DA_topcam.png")


def fig_06_peak_windows(eval_root: Path, out: Path) -> None:
    df = pd.read_csv(eval_root / "timeseries/new_aa_vs_bb/tables/peak_summary_90pct_window_aa_vs_bb.csv")
    df = df[df["metric"] == "object_ratio"].copy()
    df["group_label"] = df["group"].map({"A_A": "A@A", "B_B": "B@B"}).fillna(df["group"])
    df["camera_label"] = df["camera"].map({"camera1": "wrist", "camera2": "top"})

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    y = np.arange(len(df))
    colors = [PALETTE[g] for g in df["group"].tolist()]
    ax.barh(y, df["window_width_pct"], color=colors, alpha=0.85)

    for i, row in df.iterrows():
        ax.text(
            row["window_width_pct"] + 1.0,
            int(i),
            f"{row['window_start_pct']:.0f}-{row['window_end_pct']:.0f}%",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y, [f"{r['group_label']} / {r['camera_label']}" for _, r in df.iterrows()])
    ax.set_xlabel("90%-peak window width (%)")
    ax.set_title("Object-Ratio Peak Window Width (A@A vs B@B)")
    _save(fig, out / "Fig_06_peak_window_object_ratio_AA_BB.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Build table/figure assets for A->B->C/D paper storyline.")
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

    out_tables, out_figures = _mkdirs(out_root)
    tables = build_tables(eval_root, results_table_root, out_tables)

    fig_01_flow(out_figures)
    fig_02_a_outcomes(tables["table4"], out_figures)
    fig_03_transition(tables["table2"], out_figures)
    fig_04_progress_main(eval_root, out_figures)
    fig_05_progress_cd(eval_root, out_figures)
    fig_06_peak_windows(eval_root, out_figures)

    _copy_tree(out_root, repo_out_root)
    print(f"[done] wrote assets to: {out_root}")
    print(f"[done] mirrored assets to: {repo_out_root}")


if __name__ == "__main__":
    main()
