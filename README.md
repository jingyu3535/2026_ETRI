# 2026_ETRI Paper Release

This repository is the code release for paper experiments built on LeRobot with local SmolVLA-related changes.

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

# Verify and set camera-device mapping for this machine/session.
# Example check:
# ls -l /dev/v4l/by-id
FRONT_CAM_PATH=<FRONT_CAM_PATH>
TOP_CAM_PATH=<TOP_CAM_PATH>

# SO101 teleoperation recording
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: '${FRONT_CAM_PATH}', width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: '${TOP_CAM_PATH}', width: 640, height: 480, fps: 30}}" \
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

Transfer (local collection host -> remote training server):
- Data transfer was performed manually in Termius (SFTP/SCP workflow) from local host (`etri01`) to server (`internship@user`), then placed under `/home/internship/data/etri01/`.

## 5) Training runs used in paper
Runs reported in this repository:
- `smolVLA_task_box_750` (750 episodes)
- `smolVLA_task_box_1050` (1050 episodes)
- `smolVLA_task_box_1050_toponly` (1050 episodes, top camera only)

All three runs use:
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

Compatibility note:
- Original server run for `smolVLA_task_box_750` used legacy dataset id/path `task_box_100`.
- This README normalizes it to `task_box_750` to match episode count and model naming.
- If reproducing directly from the original server snapshot, replace `task_box_750` with `task_box_100`.

Run timeline (known dates):
- `smolVLA_task_box_750`: server checkpoint timestamps show `020000` at `2026-01-23` and `500000` at `2026-01-25`.
- `smolVLA_task_box_1050`: training log starts on `2026-02-13`, with `020000` checkpoint at `2026-02-13` and `500000` at `2026-02-15`.
- `smolVLA_task_box_1050_toponly`: server checkpoint timestamps show `020000` at `2026-02-25` and `500000` at `2026-02-27`.

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
- Evaluated checkpoints: `smolVLA_task_box_750`, `smolVLA_task_box_1050`, `smolVLA_task_box_1050_toponly`
- Robot-side on-policy evaluation data: `12` episodes per checkpoint, `36` episodes total
- Object placement layouts: two versions (`6` episodes + `6` episodes) per checkpoint

Checkpoint list for evaluation:

| Run | checkpoint path |
| --- | --- |
| `smolVLA_task_box_750` | `/home/etri01/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model` |
| `smolVLA_task_box_1050` | `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model` |
| `smolVLA_task_box_1050_toponly` | `/home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model` |

```bash
# [Local host] robot-side on-policy evaluation template (run once per checkpoint)
# Set these per run:
RUN_TAG=smolVLA_task_box_1050
CKPT=/home/etri01/model/${RUN_TAG}/checkpoints/500000/pretrained_model
FRONT_CAM_PATH=<FRONT_CAM_PATH>
TOP_CAM_PATH=<TOP_CAM_PATH>

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: '${FRONT_CAM_PATH}', width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: '${TOP_CAM_PATH}', width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_leader \
  --display_data=true \
  --dataset.single_task="pick the <object> and put it in the blue box" \
  --dataset.repo_id=etri01/eval_${RUN_TAG}_12ep \
  --dataset.root=/home/etri01/model/eval_${RUN_TAG}_12ep \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=40 \
  --dataset.reset_time_s=10 \
  --dataset.num_episodes=12 \
  --resume=false \
  --policy.type=smolvla \
  --policy.pretrained_path=${CKPT}

# [Server] attention dump example (single run with 12 episodes)
PYTHONPATH=/home/internship/projects/lerobot/src \
  /home/internship/miniforge3/envs/lerobot/bin/python \
  /home/internship/projects/lerobot/scripts/dump_action_attn_eval.py \
  --dataset_root /home/internship/model/eval_${RUN_TAG}_12ep \
  --dataset_repo_id etri01/eval_${RUN_TAG}_12ep \
  --checkpoint /home/internship/model/${RUN_TAG}/checkpoints/500000/pretrained_model \
  --dump_dir /home/internship/model/eval_${RUN_TAG}_12ep/action_attn_dump_img \
  --episodes 0-11 \
  --layers 1,5,9,13,15 \
  --denoise_steps 0,5,9 \
  --action_step all \
  --num_frames -1 \
  --stride 1 \
  --log_every 50 \
  --log_time

# If all three runs are merged into one eval dataset (36 episodes total), use:
# --episodes 0-35
```

Camera note:
- `front/top` are role labels in the command, not fixed hardware-vendor names.
- Device mapping can differ by host/session; always verify on the robot host before running (`ls -l /dev/v4l/by-id`).
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

SAM2-large mask pipeline used for `eval_task_box_1050`:
- Dataset root: `/home/etri01/model/eval_task_box_1050`
- Camera convention: `camera1=front`, `camera2=top` (aligned to 750_A analysis convention)
- Output masks: `/home/etri01/model/eval_task_box_1050/sam2_large`
- Target split: episodes `0-35`, cameras `camera1,camera2`

1. Create 5-bin seed frames per episode (0%, 25%, 50%, 75%, 100%)
```bash
/home/etri01/miniforge3/envs/lerobot/bin/python - <<'PY'
from pathlib import Path
import shutil

root = Path("/home/etri01/model/eval_task_box_1050")
for cam in ["camera1", "camera2"]:
    in_root = root / "frames_by_ep" / cam
    out_root = root / "seg_seed_frames" / cam
    out_root.mkdir(parents=True, exist_ok=True)

    for ep_dir in sorted(in_root.glob("ep*")):
        frames = sorted(ep_dir.glob("frame_*.png"))
        if not frames:
            continue
        n = len(frames)
        idxs = [0, round((n - 1) * 0.25), round((n - 1) * 0.5), round((n - 1) * 0.75), n - 1]
        idxs = sorted(set(idxs))
        for i in idxs:
            src = frames[i]
            dst = out_root / f"{ep_dir.name}_{src.name}"
            shutil.copy2(src, dst)
        print(f"[done] {cam}/{ep_dir.name} -> {len(idxs)} seeds")
PY
```

2. Manual seed mask labeling (`seg_label_tool.py`)
```bash
# camera1
PYTHONPATH=src /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/seg_label_tool.py \
  --input_dir /home/etri01/model/eval_task_box_1050/seg_seed_frames/camera1 \
  --output_dir /home/etri01/model/eval_task_box_1050/seg_seed_masks/camera1

# camera2
PYTHONPATH=src /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/seg_label_tool.py \
  --input_dir /home/etri01/model/eval_task_box_1050/seg_seed_frames/camera2 \
  --output_dir /home/etri01/model/eval_task_box_1050/seg_seed_masks/camera2
```

Label-tool hotkeys:
- Left click: add point
- Right click: remove last point
- `a`: finalize current polygon
- `r`: reset current work
- `s`: save filled mask and move next
- `n`: save empty mask and move next
- `q`: quit

3. Optional correction for empty first-seed masks
- For episodes where the first seed mask is empty but the object appears later, add one extra seed frame near first object appearance and label it manually.
- Extra seed frames are stored under `/home/etri01/model/eval_task_box_1050/seg_seed_frames/camera1_extra_first_object` and merged into the existing camera1 seed-mask directory.

```bash
PYTHONPATH=src /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/seg_label_tool.py \
  --input_dir /home/etri01/model/eval_task_box_1050/seg_seed_frames/camera1_extra_first_object \
  --output_dir /home/etri01/model/eval_task_box_1050/seg_seed_masks/camera1
```

4. Run SAM2-large VOS over full episodes
```bash
EP=$(seq -s, 0 35)

PYTHONPATH=src /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/sam2_vos_test.py \
  --video_dir /home/etri01/model/eval_task_box_1050/video_by_ep \
  --seed_frames_dir /home/etri01/model/eval_task_box_1050/seg_seed_frames \
  --seed_masks_dir /home/etri01/model/eval_task_box_1050/seg_seed_masks \
  --out_dir /home/etri01/model/eval_task_box_1050/sam2_large \
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

5. Generate full-frame overlay checks
```bash
PYTHONPATH=src /home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/make_sam2_overlay_check.py \
  --frames_root /home/etri01/model/eval_task_box_1050/frames_by_ep \
  --masks_root /home/etri01/model/eval_task_box_1050/sam2_large \
  --out_root /home/etri01/model/eval_task_box_1050/sam2_overlay_check \
  --episodes 0-35 \
  --cameras camera1,camera2 \
  --alpha 0.20 \
  --draw_contour \
  --contour_thickness 1
```

6. Final consistency check (frame/mask filename set must match)
```bash
/home/etri01/miniforge3/envs/lerobot/bin/python - <<'PY'
from pathlib import Path
root = Path("/home/etri01/model/eval_task_box_1050")
ok = True
for cam in ["camera1", "camera2"]:
    for ep in range(36):
        epn = f"ep{ep:03d}"
        f = set(p.name for p in (root / "frames_by_ep" / cam / epn).glob("frame_*.png"))
        m = set(p.name for p in (root / "sam2_large" / cam / epn).glob("frame_*.png"))
        if f != m:
            ok = False
            print("[mismatch]", cam, epn, "missing", len(f - m), "extra", len(m - f))
print("ALL_OK =", ok)
PY
```

## 7) Expected results
Runtime summary example (`task_box_1050` log):
- `cfg.steps=500000`
- `dataset.num_frames=421143`
- `dataset.num_episodes=1050`
- `effective batch size=32`
- Final task metrics and success rates: `TODO`

## 8) Local code changes summary
Core modified files:
- `src/lerobot/datasets/dataset_tools.py`
- `src/lerobot/datasets/lerobot_dataset.py`
- `src/lerobot/policies/smolvla/configuration_smolvla.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
- `src/lerobot/robots/so_follower/so_follower.py`
- `src/lerobot/scripts/lerobot_train.py`

See also `PAPER_RELEASE.md`.

## 9) Security note
Keep credentials in environment variables only. Do not hardcode tokens or keys.

## 10) Provenance
Evidence and source line references are documented in `REPRO_EVIDENCE.md`.
