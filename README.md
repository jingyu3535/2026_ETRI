# 2026_ETRI Paper Release

This repository is the code release for the paper experiments built on LeRobot with local SmolVLA-related changes.

## Base
- Upstream: `huggingface/lerobot`
- Base commit: `15724826`
- Paper branch: `paper_release`

## Repository scope
- Included: `src/`, `scripts/`, packaging/dependency files
- Excluded: upstream docs/examples/tests/CI and temporary backup files

## 1) Environment
This work used a split pipeline:
- Local robot host for data collection/inference with two SO101 arms (`leader`, `follower`)
- Remote server for model training and cross-attention dump analysis

### Local robot host (collection/inference)
- Hardware: `SO101 leader + SO101 follower`
- OS: `Ubuntu 22.04.5 LTS`
- Python: `3.10.19` (conda env: `lerobot`)
- CUDA: `11.8` (PyTorch build)
- PyTorch: `2.7.0+cu118`
- GPU: `NVIDIA GeForce GTX TITAN X (12209 MiB, driver 470.256.02)`
- Notes: robot control and episode recording were executed locally

### Remote training server (training/analysis)
- OS: `Ubuntu 24.04.3 LTS (x86_64)` (verified from training log)
- Python: `3.10.19` (conda env: `lerobot`)
- CUDA: `12.4`
- PyTorch: `2.6.0+cu124`
- GPU: `NVIDIA H200 NVL (143771 MiB, driver 570.195.03)`
- Notes: model training and cross-attention dump were executed on server

### Data transfer
- Episode data was transferred from local host to server via Termius/SCP workflow.
- Verified transfer example:
```bash
scp -o 'ProxyJump=etri01@<jump_host>' \
  /home/etri01/Downloads/smolvla/compare_object_attention.py \
  etri01@10.77.23.171:/home/etri01/Downloads/smolvla/
```

## 2) Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[smolvla]"
```

## 3) Dataset preparation
Describe local collection first, then server-side preprocessing.

```bash
# [Local host] collect episode data with SO101 leader/follower
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

# [Server] copy uploaded episodes into training dataset path
# TODO: server-side import/arrange command(s)

# [Server] preprocessing (if used)
# TODO: preprocessing command(s)
```

## 4) Training
Run on remote server.

```bash
# [Server] full train command used for reported result
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

## 5) Evaluation
Use exact checkpoint path. If inference was run on local robot host, separate it from server evaluation.

```bash
# [Server] offline eval command (if used)
# TODO

# [Local host] robot-side inference/eval command (if used)
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

# [Server] cross-attention dump command
/home/etri01/miniforge3/envs/lerobot/bin/python \
  /home/etri01/projects/lerobot/scripts/dump_action_attn_eval.py \
  --dataset_root /home/etri01/model/eval_task_box_1050 \
  --dataset_repo_id etri01/eval_task_box_1050 \
  --checkpoint /home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model \
  --dump_dir /home/etri01/model/eval_task_box_1050_toponly/action_attn_dump_img/_raw_tuned \
  --dump_action \
  --episodes 0-35 \
  --num_frames -1 \
  --stride 1 \
  --layers 1,3,5,7,9,11,13,15 \
  --denoise_steps all \
  --action_step all \
  --heads mean \
  --log_every 50 \
  --log_time
```

## 6) Expected results
- Main metric(s): `TODO`
- Expected range: `TODO`
- Random seed(s): `TODO`
- Number of runs and averaging rule: `TODO`

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
