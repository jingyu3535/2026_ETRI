# Reproducibility Evidence Log

This file tracks evidence used to avoid memory-only documentation.

## Verified (high confidence)

1. Remote training server OS and session context
- Source: `/home/etri01/model/smolVLA_task_box_1050/train_log_full_20260213_175207.log:1`
- Evidence: `Welcome to Ubuntu 24.04.3 LTS (GNU/Linux 6.14.0-36-generic x86_64)`

2. Remote training command (task_box_1050 run)
- Source: `/home/etri01/model/smolVLA_task_box_1050/train_log_full_20260213_175207.log:21`
- Source: `/home/etri01/model/smolVLA_task_box_1050/train_log_full_20260213_175207.log:35`
- Notes: command appears in wrapped form in the log; README includes normalized equivalent.

3. Training run key parameters
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:3`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:4`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:194`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:196`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:197`
- Values:
  - dataset repo: `task_box_1050`
  - dataset root: `/home/internship/data/etri01/task_box_1050`
  - seed: `1000`
  - batch size: `32`
  - steps: `500000`

4. Training dataset/frame summary from runtime log
- Source: `/home/etri01/model/smolVLA_task_box_1050/train_log_full_20260213_175207.log:196`
- Source: `/home/etri01/model/smolVLA_task_box_1050/train_log_full_20260213_175207.log:198`
- Values:
  - cfg.steps: `500000`
  - dataset.num_episodes: `1050`

5. Local robot data collection command (SO101 leader/follower)
- Source: `/home/etri01/.bash_history:1185`

6. Local robot-side inference/evaluation command variant
- Source: `/home/etri01/.bash_history:1627`

7. Attention dump command
- Source: `/home/etri01/.bash_history:2000`

8. Local-to-server transfer command via jump host
- Source: `/home/etri01/.bash_history:1159`

9. Local host runtime stack
- Source: local shell output on `2026-03-05` (`conda env: lerobot`)
- Values:
  - OS: `Ubuntu 22.04.5 LTS`
  - Python: `3.10.19`
  - PyTorch/CUDA: `2.7.0+cu118 / 11.8`
  - GPU: `NVIDIA GeForce GTX TITAN X`, driver `470.256.02`

10. Remote training server runtime stack
- Source: user-provided server shell output on `2026-03-05` (`internship@user`, `conda env: lerobot`)
- Values:
  - OS: `Ubuntu 24.04.3 LTS`
  - Python: `3.10.19`
  - PyTorch/CUDA: `2.6.0+cu124 / 12.4`
  - GPU: `NVIDIA H200 NVL`, driver `570.195.03`

## Partial / needs confirmation

1. Remote server exact package lock state
- `train_config.json` captures model/training config, but not full pip/conda lockfile at run time.

2. Final paper-selected run
- Multiple runs/checkpoints exist (`smolVLA_task_box_750`, `smolVLA_task_box_1050`, `smolVLA_task_box_1050_toponly`).
- Final run to report should be explicitly chosen and tagged.

## Recommendation

Before submission, export and archive:
- `conda env export` (or `pip freeze`) for both local and remote environments.
- Final chosen train/eval command block and checkpoint path.
- Metric summary file used in the paper table.
