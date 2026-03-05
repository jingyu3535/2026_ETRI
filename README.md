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
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[smolvla]"
```

## 3) Dataset collection and transfer
```bash
# [Local host] SO101 teleoperation data collection example
OPENCV_VIDEOIO_PRIORITY_FFMPEG=0 OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0 lerobot-record \
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
  --dataset.num_episodes=75 \
  --dataset.reset_time_s=0 \
  --dataset.single_task="pick the banana and put it in the blue box"
```

Data was transferred between local and server using SCP/rsync in Termius-based workflow.

Language prompt template used for collection/evaluation:
- `pick the <object> and put it in the <box> box`

Verified prompts and counts from `/home/etri01/.cache/huggingface/lerobot/etri01/task_box_1050/meta/episodes/chunk-000/*.parquet`:
- `pick the banana and put it in the transparent box`: `100`
- `pick the socks and put it in the transparent box`: `100`
- `pick the strawberry and put it in the transparent box`: `100`
- `pick the banana and put it in the blue box`: `234`
- `pick the socks and put it in the blue box`: `250`
- `pick the strawberry and put it in the blue box`: `266`

Dataset composition summary:
- Transparent-box episodes: `300` (`100` per object)
- Blue-box episodes in final `task_box_1050`: `750` (`234/250/266` by banana/socks/strawberry)
- Intermediate `750` setting used during development: `300 + 450` where the additional blue split is `134/150/166` (banana/socks/strawberry)

Why transparent -> blue:
- Practical observation during early inference was unstable target-box recognition on transparent container, so later collection used blue box for stronger visual contrast.
- This reason is observation-based; no separate quantitative ablation log is currently archived.

## 4) Training runs used in paper
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

Main run command:
```bash
cd /home/internship/projects/lerobot
conda activate lerobot
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=task_box_1050 \
  --dataset.root=/home/internship/data/etri01/task_box_1050 \
  --policy.repo_id=internship/temp_model_1050 \
  --batch_size=32 \
  --steps=500000 \
  --output_dir=/home/internship/model/smolVLA_task_box_1050 \
  --policy.device=cuda \
  --wandb.enable=false \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.top":"observation.images.camera2"}' \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=15000 \
  --policy.scheduler_decay_steps=500000
```

## 5) Evaluation and attention dump
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

## 6) Expected results
Main run (`task_box_1050`) runtime summary:
- `cfg.steps=500000`
- `dataset.num_frames=421143`
- `dataset.num_episodes=1050`
- `effective batch size=32`
- Final task metrics and success rates: `TODO`

## 7) Local code changes summary
Core modified files:
- `src/lerobot/datasets/dataset_tools.py`
- `src/lerobot/datasets/lerobot_dataset.py`
- `src/lerobot/policies/smolvla/configuration_smolvla.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
- `src/lerobot/robots/so_follower/so_follower.py`
- `src/lerobot/scripts/lerobot_train.py`

See also `PAPER_RELEASE.md`.

## 8) Security note
Keep credentials in environment variables only. Do not hardcode tokens or keys.

## 9) Provenance
Evidence and source line references are documented in `REPRO_EVIDENCE.md`.
