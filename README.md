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

# SO101 teleoperation recording
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: '/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN0001-video-index0', width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: '/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0', width: 640, height: 480, fps: 30}}" \
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
- `smolVLA_task_box_750` -> `750` episodes
- `smolVLA_task_box_1050` -> `1050` episodes
- `smolVLA_task_box_1050_toponly` -> same `1050` episodes with top-view-only input (`front` removed by `rename_map`)

Transfer (local collection host -> remote training server):
- Data transfer was performed manually in Termius (SFTP/SCP workflow) from local host (`etri01`) to server (`internship@user`), then placed under `/home/internship/data/etri01/`.

## 5) Training runs used in paper
Primary reported run:
- `smolVLA_task_box_1050` (main run)

Additional comparison runs:
- `smolVLA_task_box_750` (fewer episodes)
- `smolVLA_task_box_1050_toponly` (top camera only)

All three runs use:
- `steps=500000`
- `batch_size=32`
- `seed=1000`
- `policy.optimizer_lr=5e-5`
- `policy.scheduler_warmup_steps=15000`
- `policy.scheduler_decay_steps=500000`

Run matrix (from `checkpoints/500000/pretrained_model/train_config.json`):

| Run | dataset.repo_id | dataset.root | rename_map | output_dir |
| --- | --- | --- | --- | --- |
| `smolVLA_task_box_750` | `task_box_100` | `/home/internship/data/etri01/task_box_100` | `front->camera1`, `top->camera2` | `/home/internship/model/smolVLA_task_box_750` |
| `smolVLA_task_box_1050` | `task_box_1050` | `/home/internship/data/etri01/task_box_1050` | `front->camera1`, `top->camera2` | `/home/internship/model/smolVLA_task_box_1050` |
| `smolVLA_task_box_1050_toponly` | `task_box_1050_toponly` | `/home/internship/data/etri01/task_box_1050_toponly` | `top->camera1` | `/home/internship/model/smolVLA_task_box_1050_toponly` |

Run timeline (known dates):
- `smolVLA_task_box_750`: server checkpoint timestamps show `020000` at `2026-01-23` and `500000` at `2026-01-25`.
- `smolVLA_task_box_1050` (main): training log starts on `2026-02-13`, with `020000` checkpoint at `2026-02-13` and `500000` at `2026-02-15`.
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
| `smolVLA_task_box_750` | `task_box_100` | `/home/internship/data/etri01/task_box_100` | `internship/temp_model_750` | `/home/internship/model/smolVLA_task_box_750` | `{"observation.images.front":"observation.images.camera1","observation.images.top":"observation.images.camera2"}` |
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
```bash
# [Local host] robot-side inference/evaluation example
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_follower \
  --robot.cameras="{ camera2: {type: opencv, index_or_path: '/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0', width: 640, height: 480, fps: 30, fourcc: MJPG} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_leader \
  --display_data=true \
  --dataset.single_task="pick the strawberry and put it in the blue box" \
  --dataset.repo_id=etri01/eval_task_box_1050_cam2only \
  --dataset.root=/home/etri01/model/eval_task_box_1050_cam2only \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=40 \
  --dataset.reset_time_s=10 \
  --dataset.num_episodes=4 \
  --resume=false \
  --policy.type=smolvla \
  --policy.pretrained_path=/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model

# [Server] attention dump example
PYTHONPATH=/home/internship/projects/lerobot/src \
  /home/internship/miniforge3/envs/lerobot/bin/python \
  /home/internship/projects/lerobot/scripts/dump_action_attn_eval.py \
  --dataset_root /home/internship/model/eval_task_box_750_0202 \
  --dataset_repo_id etri01/eval_task_box_750_0202 \
  --checkpoint /home/internship/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model \
  --dump_dir /home/internship/model/eval_task_box_750_0202/action_attn_dump_img_0211/_raw_tuned \
  --episodes 0-35 \
  --layers 1,5,9,13,15 \
  --denoise_steps 0,5,9 \
  --action_step all \
  --num_frames -1 \
  --stride 1 \
  --log_every 50 \
  --log_time
```

## 7) Expected results
Main run (`task_box_1050`) runtime summary:
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
