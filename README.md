# 2026_ETRI Paper Release

This repository is the code release for paper experiments built on LeRobot with local SmolVLA-related changes.

## Quick start for paper organization
- Playbook index: `docs/paper_release/README.md`
- Ordered workflow: `docs/paper_release/01_EXPERIMENT_WORKFLOW.md`
- Release checklist: `docs/paper_release/02_RELEASE_CHECKLIST.md`
- Result package spec: `docs/paper_release/05_RESULTS_PACKAGE.md`
- Generated assets root: `results/paper/`

## Base
- Upstream: `huggingface/lerobot`
- Base commit: `15724826`
- Release branch: `paper_release`

## Repository scope
- Included: `src/`, `scripts/`, packaging/dependency files
- Excluded: upstream docs/examples/tests/CI and temporary backup files

## 1) Environment
This work used a split pipeline:
- Local robot host for data collection and robot-side inference (`SO101 leader + follower`)
- Remote server for training and attention dump analysis

### Local robot host
- Hostname: `etri01`
- OS: `Ubuntu 22.04.5 LTS`
- Python: `3.10.19` (conda env `lerobot`)
- PyTorch/CUDA: `2.7.0+cu118 / 11.8`
- GPU: `NVIDIA GeForce GTX TITAN X` (12209 MiB, driver `470.256.02`)

### Remote training server
- Hostname: `user`
- OS: `Ubuntu 24.04.3 LTS (x86_64)`
- Python: `3.10.19` (conda env `lerobot`)
- PyTorch/CUDA: `2.6.0+cu124 / 12.4`
- GPU: `NVIDIA H200 NVL` (143771 MiB, driver `570.195.03`)

## 2) Installation
We follow the official LeRobot installation guide:
- https://huggingface.co/docs/lerobot/installation

For paper reproduction, use the exact environment style below.

```bash
# 1) Conda env (Python 3.10)
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install -y -c conda-forge ffmpeg

# 2) Clone and pin base code
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 15724826

# 3) Install dependencies for this project
# Training/analysis:
pip install -e ".[smolvla]"

# Local SO101 robot control (Feetech motors):
pip install -e ".[smolvla,feetech]"
```

## 3) Robot Setup (SO101)
Run these on the local robot host before data recording.

```bash
conda activate lerobot

# 1) Identify motor USB ports (run once per arm by unplug/replug)
lerobot-find-port

# 2) Identify available cameras
lerobot-find-cameras

# 3) Calibrate leader and follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_leader
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_follower
```

Optional sanity check before recording:

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_leader \
  --display_data=true
```

## 4) Dataset collection and transfer
Canonical collection command (blue-box stage):

```bash
# [Local host] optional OpenCV backend override (used in part of runs)
export OPENCV_VIDEOIO_PRIORITY_FFMPEG=0
export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0

# SO101 teleoperation recording
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: '/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0', width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: '/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN0001-video-index0', width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_leader \
  --display_data=true \
  --dataset.repo_id=etri01/blue_box_data \
  --resume=true \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=<N> \
  --dataset.reset_time_s=0 \
  --dataset.single_task="pick the <object> and put it in the blue box" \
  --manual_advance=true
```

Collection details:
- Language prompt template: `pick the <object> and put it in the <box> box`
- Prompt used in transparent stage: `pick the <object> and put it in the transparent box`
- Prompt used in blue stage: `pick the <object> and put it in the blue box`

Camera naming:
- Paper text: `wrist-view`, `top-view`
- Stored keys: `observation.images.front` (wrist-view), `observation.images.top` (top-view)

Dataset growth used by analysis models:
- Initial transparent-box data: `300` = banana `100`, socks `100`, strawberry `100`
- To build `750`, blue-box data added: `450` = banana `134`, socks `150`, strawberry `166`
- `750` total = transparent `300` + blue `450`
- To build `1050`, additional blue-box data added: `300` = banana `100`, socks `100`, strawberry `100`
- Final `1050` totals = transparent `300` + blue `750` (banana `234`, socks `250`, strawberry `266`)

Model-to-dataset mapping:
- `smolVLA_task_box_750` -> `750` episodes (`task_box_750`)
- `smolVLA_task_box_1050` -> `1050` episodes
- `smolVLA_task_box_1050_toponly` -> same `1050` episodes with top-view-only input (`front` removed by `rename_map`)
- `smolVLA_task_box_795` -> `795` episodes (`task_box_795`, D condition: +45 dose)

Transfer (local collection host -> remote training server):
- Data transfer was performed manually in Termius (SFTP/SCP workflow) from local host (`etri01`) to server (`internship@user`), then placed under `/home/internship/data/etri01/`.

## 5) Training runs used in paper
Runs reported in this repository:
- `smolVLA_task_box_750` (750 episodes)
- `smolVLA_task_box_1050` (1050 episodes)
- `smolVLA_task_box_1050_toponly` (1050 episodes, top camera only)
- `smolVLA_task_box_795` (795 episodes, dose-control D condition)

All four runs use:
- `steps=500000`
- `batch_size=32`
- `seed=1000`
- `policy.optimizer_lr=5e-5`
- `policy.scheduler_warmup_steps=15000`
- `policy.scheduler_decay_steps=500000`

Run matrix:

| Run | dataset.repo_id | dataset.root | rename_map | output_dir |
| --- | --- | --- | --- | --- |
| `smolVLA_task_box_750` | `task_box_750` | `/home/internship/data/etri01/task_box_750` | `front->camera1`, `top->camera2` | `/home/internship/model/smolVLA_task_box_750` |
| `smolVLA_task_box_1050` | `task_box_1050` | `/home/internship/data/etri01/task_box_1050` | `front->camera1`, `top->camera2` | `/home/internship/model/smolVLA_task_box_1050` |
| `smolVLA_task_box_1050_toponly` | `task_box_1050_toponly` | `/home/internship/data/etri01/task_box_1050_toponly` | `top->camera1` | `/home/internship/model/smolVLA_task_box_1050_toponly` |
| `smolVLA_task_box_795` | `task_box_795` | `/home/internship/data/etri01/task_box_795` | `front->camera1`, `top->camera2` | `/home/internship/model/smolVLA_task_box_795` |

Compatibility note:
- Original server run for `smolVLA_task_box_750` used legacy dataset id/path `task_box_100`.
- This README normalizes it to `task_box_750` to match episode count and model naming.
- If reproducing directly from the original server snapshot, replace `task_box_750` with `task_box_100`.

Run timeline (known dates):
- `smolVLA_task_box_750`: server checkpoint timestamps show `020000` at `2026-01-23` and `500000` at `2026-01-25`.
- `smolVLA_task_box_1050`: training log starts on `2026-02-13`, with `020000` checkpoint at `2026-02-13` and `500000` at `2026-02-15`.
- `smolVLA_task_box_1050_toponly`: server checkpoint timestamps show `020000` at `2026-02-25` and `500000` at `2026-02-27`.
- `smolVLA_task_box_795`: local archived `500000` checkpoint timestamp is `2026-03-11`.

500k training command template:
```bash
cd /home/internship/projects/lerobot
conda activate lerobot
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<DATASET_REPO_ID> \
  --dataset.root=<DATASET_ROOT> \
  --policy.repo_id=<POLICY_REPO_ID> \
  --batch_size=32 \
  --steps=500000 \
  --output_dir=<OUTPUT_DIR> \
  --policy.device=cuda \
  --wandb.enable=false \
  --dataset.video_backend=pyav \
  --rename_map='<RENAME_MAP_JSON>' \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=15000 \
  --policy.scheduler_decay_steps=500000 \
  --policy.push_to_hub=false
```

Run-specific values:

| Run | dataset.repo_id | dataset.root | policy.repo_id | output_dir | rename_map |
| --- | --- | --- | --- | --- | --- |
| `smolVLA_task_box_750` | `task_box_750` | `/home/internship/data/etri01/task_box_750` | `internship/temp_model_750` | `/home/internship/model/smolVLA_task_box_750` | `{"observation.images.front":"observation.images.camera1","observation.images.top":"observation.images.camera2"}` |
| `smolVLA_task_box_1050` | `task_box_1050` | `/home/internship/data/etri01/task_box_1050` | `internship/temp_model_1050` | `/home/internship/model/smolVLA_task_box_1050` | `{"observation.images.front":"observation.images.camera1","observation.images.top":"observation.images.camera2"}` |
| `smolVLA_task_box_1050_toponly` | `task_box_1050_toponly` | `/home/internship/data/etri01/task_box_1050_toponly` | `internship/temp_model_1050_toponly` | `/home/internship/model/smolVLA_task_box_1050_toponly` | `{"observation.images.top":"observation.images.camera1"}` |
| `smolVLA_task_box_795` | `task_box_795` | `/home/internship/data/etri01/task_box_795` | `internship/temp_model_500000_795` | `/home/internship/model/smolVLA_task_box_795` | `{"observation.images.front":"observation.images.camera1","observation.images.top":"observation.images.camera2"}` |

Training changes vs defaults (for 500k runs):

| Item | Default | Used |
| --- | --- | --- |
| `batch_size` | `8` | `32` |
| `steps` | `100000` | `500000` |
| `policy.optimizer_lr` | `1e-4` | `5e-5` |
| `policy.scheduler_warmup_steps` | `1000` | `15000` |
| `policy.scheduler_decay_steps` | `30000` | `500000` |
| `rename_map` | empty | set per run |

Reasoning summary:
- Lower `lr` and longer `warmup` were used to stabilize long-horizon fine-tuning.
- `scheduler_decay_steps=500000` keeps effective learning updates through late training instead of early decay saturation.
- `batch_size=32` improves update stability and throughput under available GPU memory.
- `steps=500000` was used for longer optimization than the 100k default.
- `rename_map` is required because dataset image keys (`front/top`) are mapped to policy keys (`camera1/2`).
- `policy.push_to_hub=false` avoids HF auth/push failures while preserving local checkpoints.

## 6) Evaluation and attention dump
Evaluation protocol used for analysis:
- Reference frame source is fixed to A rollout (`A frame`): `/home/etri01/paper/eval/eval_task_box_750_A`
- Cross-model comparison is performed on the same frame source (`X@A`): `P@A`, `A@A`, `B@A`, `C@A`, `D@A`
- Object placement layouts: two versions (`6` episodes + `6` episodes)

Evaluation directory convention used in this release:
- Frame source (`A`): `/home/etri01/paper/eval/eval_task_box_750_A` (`data/meta/videos` 포함)
- Dump outputs on A frames:
  - `A@A`: `/home/etri01/paper/eval/eval_task_box_750_A/action_attn_dump_img_all`
  - `P@A`: `/home/etri01/paper/eval/eval_task_box_750_P/action_attn_dump_img/_raw_base`
  - `B@A`: `/home/etri01/paper/eval/eval_task_box_750_B/action_attn_dump_img/_raw_tuned`
  - `C@A`: `/home/etri01/paper/eval/eval_task_box_750_C/action_attn_dump_img/_raw_tuned`
  - `D@A`: `/home/etri01/paper/eval/eval_task_box_750_D/action_attn_dump_img/_raw_tuned`

Checkpoint list for evaluation:

| Run | checkpoint path |
| --- | --- |
| `smolVLA_task_box_750` | `/home/etri01/paper/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model` |
| `smolVLA_task_box_1050` | `/home/etri01/paper/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model` |
| `smolVLA_task_box_1050_toponly` | `/home/etri01/paper/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model` |
| `smolVLA_task_box_795` | `/home/etri01/paper/model/smolVLA_task_box_795/checkpoints/500000/pretrained_model` |

```bash
# Step 1) A frame 생성 (한 번만 수행)
RUN_TAG_FRAME=smolVLA_task_box_750
CKPT_FRAME=/home/etri01/paper/model/${RUN_TAG_FRAME}/checkpoints/500000/pretrained_model
FRAME_TAG=eval_task_box_750_A

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: '/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0', width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: '/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN0001-video-index0', width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_leader \
  --display_data=true \
  --dataset.single_task="pick the <object> and put it in the blue box" \
  --dataset.repo_id=etri01/${FRAME_TAG} \
  --dataset.root=/home/etri01/paper/eval/${FRAME_TAG} \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=40 \
  --dataset.reset_time_s=10 \
  --dataset.num_episodes=12 \
  --resume=false \
  --policy.type=smolvla \
  --policy.pretrained_path=${CKPT_FRAME}

# Step 2) 고정된 A frame에 대해 모델 X dump (예: B@A)
RUN_TAG_DUMP=smolVLA_task_box_1050
DUMP_TAG=eval_task_box_750_B
DUMP_DIR=/home/etri01/paper/eval/${DUMP_TAG}/action_attn_dump_img/_raw_tuned
PYTHONPATH=/home/etri01/projects/lerobot/src \
  /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/dump_action_attn_eval.py \
  --dataset_root /home/etri01/paper/eval/${FRAME_TAG} \
  --dataset_repo_id etri01/${FRAME_TAG} \
  --checkpoint /home/etri01/paper/model/${RUN_TAG_DUMP}/checkpoints/500000/pretrained_model \
  --dump_dir ${DUMP_DIR} \
  --episodes 0-11 \
  --layers 1,5,9,13,15 \
  --denoise_steps 0,5,9 \
  --action_step all \
  --num_frames -1 \
  --stride 1 \
  --log_every 50 \
  --log_time
```

Camera note:
- This repository hardcodes camera mapping as `front=camera1=Innomaker`, `top=camera2=Sonix` on host `etri01`.
- Single-camera (`camera2` only) evaluation is a separate diagnostic setting, not the default full evaluation.

Action-to-image cross-attention dump implementation (this fork):
- Config flags (`src/lerobot/policies/smolvla/configuration_smolvla.py`):
  - `dump_action_attn`, `dump_action_attn_layers`, `dump_action_attn_action_step`
  - `dump_action_attn_last_denoise_only`, `dump_action_attn_denoise_step`
- Inference path:
  - `predict_action_chunk_with_attn()` -> `sample_actions(..., return_attn=True)`
  - Denoise loop toggles capture with `set_action_attn_capture(...)` per selected denoise step/layer/action-step
  - Captured buffer is read via `pop_action_attn_buffer()`
- Attention capture point:
  - `smolvlm_with_expert.py:eager_attention_forward()`
  - Stores expert cross-attention `probs` at selected action query index
  - Head dimension is averaged in-model (`mean` only in this codebase)
- Saved files (`scripts/dump_action_attn_eval.py`):
  - Per combination output: `epXXX_fXXXX_lYY_dZZ_aAA_action_attn.npz`
  - Optional language attention dump when `--dump_lang` is enabled
  - Paired metadata: `*_meta.json`
  - Run summary: `summary.json` (episodes/layers/denoise_steps/action_steps/counts)

Saved tensor/meta schema:
- `.npz`
  - `attn`: shape `[B, K]` (`B` batch, usually `1`; `K` prefix token length)
  - `lang_attn` (optional with `--dump_lang`)
- `*_meta.json`
  - `episode_index`, `frame_index`, `task`, `layer`, `denoise_step`, `action_step`
  - `img_spans`, `img_token_lens`, `img_grids`, `lang_range`, `state_range`, `camera_keys`

Notes:
- This dump stores action-expert cross-attention (action query -> prefix tokens).
- `image` token ranges are identified via `img_spans` in metadata.
- CLI accepts `--heads`, but this fork supports `mean` only.

SAM2-large mask pipeline used for `eval_task_box_1050_B`:
- Dataset root: `/home/etri01/model/eval_task_box_1050_B`
- Camera convention: `camera1=front`, `camera2=top`
- Output masks: `/home/etri01/model/eval_task_box_1050_B/sam2_large`
- Target split: episodes `0-35`, cameras `camera1,camera2`

Step 1. Create 5-bin seed frames per episode (0/25/50/75/100%)
```bash
ROOT=/home/etri01/model/eval_task_box_1050_B

PYTHONPATH=src /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/tools/make_seed_frames_5bins.py \
  --frames_root ${ROOT}/frames_by_ep \
  --out_root ${ROOT}/seg_seed_frames \
  --episodes 0-35 \
  --cameras camera1,camera2
```

Step 2. Manual seed mask labeling (`seg_label_tool.py`)
```bash
ROOT=/home/etri01/model/eval_task_box_1050_B

# camera1
PYTHONPATH=src /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/seg_label_tool.py \
  --input_dir ${ROOT}/seg_seed_frames/camera1 \
  --output_dir ${ROOT}/seg_seed_masks/camera1

# camera2
PYTHONPATH=src /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/seg_label_tool.py \
  --input_dir ${ROOT}/seg_seed_frames/camera2 \
  --output_dir ${ROOT}/seg_seed_masks/camera2
```

Label-tool hotkeys:
- Left click: add point
- Right click: remove last point
- `a`: finalize current polygon
- `r`: reset current work
- `s`: save filled mask and move next
- `n`: save empty mask and move next
- `q`: quit

Step 3. Optional correction for empty first-seed masks
- For episodes where the first seed mask is empty but the object appears later, add one extra seed frame near first object appearance and label it manually.
- Extra seed frames are stored under `/home/etri01/model/eval_task_box_1050_B/seg_seed_frames/camera1_extra_first_object` and merged into the existing camera1 seed-mask directory.

```bash
ROOT=/home/etri01/model/eval_task_box_1050_B

PYTHONPATH=src /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/seg_label_tool.py \
  --input_dir ${ROOT}/seg_seed_frames/camera1_extra_first_object \
  --output_dir ${ROOT}/seg_seed_masks/camera1
```

Step 4. Run SAM2-large VOS over full episodes
```bash
ROOT=/home/etri01/model/eval_task_box_1050_B
EP=$(seq -s, 0 35)

PYTHONPATH=src /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/sam2_vos_test.py \
  --video_dir ${ROOT}/video_by_ep \
  --seed_frames_dir ${ROOT}/seg_seed_frames \
  --seed_masks_dir ${ROOT}/seg_seed_masks \
  --out_dir ${ROOT}/sam2_large \
  --episodes "$EP" \
  --cameras camera1,camera2 \
  --model_cfg /tmp/segment-anything-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint /tmp/segment-anything-2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --save_stride 1 \
  --fill_before_prompt \
  --out_index_base 1 \
  --offload_video_to_cpu \
  --offload_state_to_cpu
```

Step 5. Generate full-frame overlay checks
```bash
ROOT=/home/etri01/model/eval_task_box_1050_B

PYTHONPATH=src /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/make_sam2_overlay_check.py \
  --frames_root ${ROOT}/frames_by_ep \
  --masks_root ${ROOT}/sam2_large \
  --out_root ${ROOT}/sam2_overlay_check \
  --episodes 0-35 \
  --cameras camera1,camera2 \
  --alpha 0.20 \
  --draw_contour \
  --contour_thickness 1
```

Step 6. Final consistency check (frame/mask filename set must match)
```bash
ROOT=/home/etri01/model/eval_task_box_1050_B

PYTHONPATH=src /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/tools/check_frame_mask_match.py \
  --frames_root ${ROOT}/frames_by_ep \
  --masks_root ${ROOT}/sam2_large \
  --episodes 0-35 \
  --cameras camera1,camera2
```

### Metric Definition and Aggregation (used in Sections 7/8)
Primary source code:
- `scripts/make_attention_ratio_table.py`
- `scripts/export_paper_summary_assets.py`

Per-frame definitions (action-to-prefix cross-attention):
- `image_mass = sum(attn over selected camera image tokens)`
- `total_mass = sum(attn over full prefix tokens)`
- `image_ratio = image_mass / total_mass`
- `object_mass = sum(attn_token * coverage_token)`
- `object_ratio = object_mass / image_mass`

Soft coverage definition (SAM2-based):
- Default mode: `mask_mode=soft`, `mask_pad_size=512`
- Binary SAM2 mask is resized/padded to `512x512`, then downsampled to token grid (`8x8` for topcam setting).
- `coverage_token` is continuous in `[0, 1]` and used as a weight for `object_mass`.

Missing/empty mask handling:
- If mask file is missing or mask is all-zero, `object_mass/object_ratio` are stored as `NaN` for that row.
- `image_mass/image_ratio` are still computed from attention.

Aggregation rule used for summary tables/figures:
- Packed dumps (`A_A/B_A/C_A/D_A/B_B`): average attention over selected denoise steps and action queries.
- `P_A` base dumps: aggregate per-step dumps, then average to the same prefix-level form.
- Group means in results are computed from these row-level metrics.

Reporting convention used in this release:
- Main text metrics: `image_ratio` (allocation) + `object_ratio` (grounding).
- Support metric: `object_mass` (absolute object-referenced attention).
- `image_mass` is retained in CSVs for compatibility; in this pipeline it is numerically near-identical to `image_ratio` because `total_mass` is approximately `1` after attention softmax normalization.

## 7) Hypotheses and claim order
This paper release organizes claims by hypothesis, then maps each hypothesis to concrete comparison tables/figures.
All numbers in this section follow the metric/aggregation definition above (`soft` coverage, `mask_pad_size=512`).

H1 (fixed-frame model effect):
- Claim: `A@A -> B@A` already weakens object grounding on the same `A` frames.
- Primary comparisons: `AA_to_BA`
- Main files: `results/paper/tables/T1_fixed_frame_AA_to_BA.csv`, `results/paper/figures/G1_fixed_frame_AA_to_BA_pct.png`
- Additional analysis recommended: report confidence interval by episode bootstrap on the same comparison.

H2 (on-policy state shift):
- Claim: `B@A -> B@B` induces attention redistribution (wrist up / top down) and additional grounding drop.
- Primary comparisons: `BA_to_BB`
- Main files: `results/paper/tables/T2_onpolicy_BA_to_BB.csv`, `results/paper/figures/G2_onpolicy_BA_to_BB_pct.png`
- Additional analysis recommended: separate early/mid/late rollout windows to localize when shift starts.

H3 (camera shortcut ablation):
- Claim: removing wrist camera (`C`) should mitigate shortcut dependence.
- Primary comparisons: `AA_to_CA`
- Main files: `results/paper/tables/T3_prelim_AA_to_CA_DA.csv`, `results/paper/figures/G3_prelim_ablation_object_ratio.png`
- Status: `preliminary`
- Additional analysis recommended: rerun `C@A` with fully unified dump settings and matched stems.

H4 (dose nonlinearity):
- Claim: +45 dose (`D`) is not equivalent to +300 dose (`B`); degradation depends on dose intensity.
- Primary comparisons: `AA_to_DA` (+ compare against `AA_to_BA`)
- Main files: `results/paper/tables/T3_prelim_AA_to_CA_DA.csv`, `results/paper/figures/G4_prelim_dose_image_mass.png`
- Status: `preliminary`
- Additional analysis recommended: fit dose-response curve with intermediate doses and report threshold uncertainty.

## 8) Results package (tables + figures)
This section now follows the rebuilt paper flow in order.

Current draft files:
- `results/paper/abcd_story/RESULTS_PACKAGE.md`
- `results/paper/abcd_story/SECTION8_FLOW_DRAFT.md`

### 8.1 Model A baseline outcome (36 episodes)

Sources:
- `results/paper/abcd_story/tables/Table_01_A_36ep_outcomes.csv`
- `results/paper/abcd_story/figures/TableFig_01_A_36ep_outcomes.png`

| object | 1(success) | 2(attempt fail) | 3(no approach) |
|---|---:|---:|---:|
| banana | 4 | 7 | 1 |
| socks | 11 | 1 | 0 |
| strawberry | 0 | 9 | 3 |

Interpretation:
- Model A often reaches the object, but pick quality is weak, especially on strawberry.

### 8.2 P@A vs A@A attention (Model A analysis)

Re-aggregation source (soft coverage verification):
- command: `python scripts/reaggregate_raw_dump_direct_tables.py --out_dir results/paper/tables`
- compare table: `results/paper/tables/raw_dump_direct_PA_AA_BA_BB_compare.csv`
- validation report: `results/paper/tables/raw_dump_direct_reaggregation_report.json` (`all_passed=true` expected)

Image Mass:
- `results/paper/abcd_story/tables/Table_02_PA_AA_image_mass_by_camera.csv`
- `results/paper/abcd_story/figures/TableFig_02_PA_AA_image_mass_by_camera.png`

| camera | P@A | A@A | Δ(%) |
|---|---:|---:|---:|
| camera1 | 0.291 | 0.231 | -20.3% |
| camera2 | 0.300 | 0.397 | +32.2% |

Object Ratio:
- `results/paper/abcd_story/tables/Table_03_PA_AA_object_ratio_by_camera.csv`
- `results/paper/abcd_story/figures/TableFig_03_PA_AA_object_ratio_by_camera.png`

| camera | P@A | A@A | Δ(%) |
|---|---:|---:|---:|
| camera1 | 0.142 | 0.126 | -11.2% |
| camera2 | 0.023 | 0.033 | +47.0% |

Interpretation:
- In `object_ratio`, top camera (`camera2`) increases (`+47.0%`) while wrist camera (`camera1`) decreases (`-11.2%`) from pretrained to A@A.
- Despite this shift, Model A still struggles in precise picking (Table 8.1, strawberry `0/12`), so wrist-dependent fine manipulation is not reliably solved.

### 8.3 Why we moved to Model B (after tables)

From the two tables above, Model A shows a consistent pattern:
- behavior: approach is often possible, but fine-grained picking is unstable (especially strawberry),
- attention: wrist-camera object-referenced attention decreases vs pretrained (`camera1` drop in both metrics).

Supplementary check (outcome-stratified, not primary evidence):
- `results/paper/abcd_story/tables/Table_11_PA_AA_outcome_compare_camera1.csv`
- `results/paper/abcd_story/tables/Table_12_PA_AA_outcome_compare_camera2.csv`
- These outcome-split tables are used as a supporting note only: they do not show a strong additional separation between outcomes beyond the main `P@A -> A@A` camera shift (and outcome `3` has small sample size, `n=4` episodes).

To target this gap, we collected additional close-range data focused on the **pick->place** segment and trained **Model B**:
- added data: `+300 episodes` (`banana 100`, `socks 100`, `strawberry 100`),
- training set transition: `task_box_750 -> task_box_1050`.

### 8.4 Model B outcome (36 episodes, manual labels)

Source:
- `results/paper/abcd_story/tables/Table_04_B_36ep_outcomes.csv`

| object | 1(success) | 2(attempt fail) | 3(no approach) |
|---|---:|---:|---:|
| banana | 1 | 7 | 4 |
| socks | 9 | 3 | 0 |
| strawberry | 1 | 10 | 1 |

Interpretation:
- Strawberry pick appears at least once (`0 -> 1`), and close-range fine motion qualitatively improves.
- But approach-stage object recognition errors increase (e.g., banana command but socks in wrist view gets selected), so overall robustness drops.

### 8.5 B on-policy attention shift (B@A -> B@B)

Image Mass:
- `results/paper/abcd_story/tables/Table_05_BA_BB_image_mass_by_camera.csv`

| camera | B@A | B@B | Δ(%) |
|---|---:|---:|---:|
| camera1 | 0.223 | 0.362 | +62.5% |
| camera2 | 0.405 | 0.276 | -32.0% |

Object Ratio:
- `results/paper/abcd_story/tables/Table_06_BA_BB_object_ratio_by_camera.csv`

| camera | B@A | B@B | Δ(%) |
|---|---:|---:|---:|
| camera1 | 0.121 | 0.116 | -4.3% |
| camera2 | 0.029 | 0.017 | -40.5% |

Interpretation:
- On-policy B@B redistributes image attention toward wrist camera (`camera1` up, `camera2` down).
- However, `object_ratio` falls in both cameras, consistent with degraded grounding and lower real-task reliability.

### 8.6 Training-log reporting policy

- Current section above does **not** include raw training logs.
- Recommended for paper: include one compact log summary table per model (run id, dataset, steps, LR, final loss, selected checkpoint, eval success), and keep full logs as appendix/artifact links.

## 9) Local code changes summary
Core modified files:
- `src/lerobot/datasets/dataset_tools.py`
- `src/lerobot/datasets/lerobot_dataset.py`
- `src/lerobot/policies/smolvla/configuration_smolvla.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
- `src/lerobot/robots/so_follower/so_follower.py`
- `src/lerobot/scripts/lerobot_train.py`

See also `PAPER_RELEASE.md`.

## 10) Security note
Keep credentials in environment variables only. Do not hardcode tokens or keys.

## 11) Provenance
Evidence and source line references are documented in `REPRO_EVIDENCE.md`.
