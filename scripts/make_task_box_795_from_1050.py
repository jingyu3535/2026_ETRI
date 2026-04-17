#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _load_episode_meta_df(dataset_root: Path) -> pd.DataFrame:
    ep_dir = dataset_root / "meta" / "episodes" / "chunk-000"
    files = sorted(ep_dir.glob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet found: {ep_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values("episode_index")


def _task_name(x) -> str:
    if isinstance(x, list) and len(x) > 0:
        return str(x[0])
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/internship/data/etri01")
    ap.add_argument("--src_750", default="task_box_750")
    ap.add_argument("--src_1050", default="task_box_1050")
    ap.add_argument("--tmp_45", default="task_box_45_from1050")
    ap.add_argument("--dst_795", default="task_box_795")
    ap.add_argument("--added_start", type=int, default=750)
    ap.add_argument("--tail_per_task", type=int, default=15)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    src750_root = root / args.src_750
    src1050_root = root / args.src_1050
    tmp45_root = root / args.tmp_45
    dst795_root = root / args.dst_795

    if args.overwrite:
        for p in [tmp45_root, dst795_root]:
            if p.exists():
                shutil.rmtree(p)
                print(f"[rm] {p}")
    else:
        for p in [tmp45_root, dst795_root]:
            if p.exists():
                raise FileExistsError(f"{p} already exists. Use --overwrite to recreate.")

    # 1) 1050 메타에서 추가 300(ep>=750) 중 task별 마지막 15개 선택
    df1050 = _load_episode_meta_df(src1050_root).copy()
    df1050["task"] = df1050["tasks"].apply(_task_name)
    added = df1050[df1050["episode_index"] >= args.added_start].copy()
    selected = (
        added.groupby("task", group_keys=False)
        .tail(args.tail_per_task)
        .sort_values("episode_index")
        .reset_index(drop=True)
    )
    selected_eps = selected["episode_index"].astype(int).tolist()

    print(f"[select] added episodes: {len(added)}")
    print(f"[select] selected episodes: {len(selected_eps)}")
    print(selected.groupby("task")["episode_index"].agg(["min", "max", "count"]))

    # 2) 1050 -> selected 45만 남긴 임시셋 생성
    ds1050 = LeRobotDataset(args.src_1050, root=src1050_root)
    remove_from_1050 = [i for i in range(ds1050.meta.total_episodes) if i not in set(selected_eps)]
    ds45 = delete_episodes(
        dataset=ds1050,
        episode_indices=remove_from_1050,
        output_dir=tmp45_root,
        repo_id=args.tmp_45,
    )
    print(f"[tmp45] episodes={ds45.meta.total_episodes}, frames={ds45.meta.total_frames}")

    # 3) 750 + 45 merge -> 795
    ds750 = LeRobotDataset(args.src_750, root=src750_root)
    ds45 = LeRobotDataset(args.tmp_45, root=tmp45_root)
    ds795 = merge_datasets(
        datasets=[ds750, ds45],
        output_repo_id=args.dst_795,
        output_dir=dst795_root,
    )
    print(f"[dst795] episodes={ds795.meta.total_episodes}, frames={ds795.meta.total_frames}")

    # 4) 검증 출력
    df795 = _load_episode_meta_df(dst795_root).copy()
    df795["task"] = df795["tasks"].apply(_task_name)
    tail = df795[df795["episode_index"] >= args.added_start]
    print(f"[check] tail episodes in dst(>= {args.added_start}): {len(tail)}")
    print(tail.groupby("task")["episode_index"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
