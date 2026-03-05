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
- OS: `TODO`
- Python: `TODO`
- Notes: robot control and episode recording were executed locally

### Remote training server (training/analysis)
- OS: `TODO`
- Python: `TODO`
- CUDA: `TODO`
- PyTorch: `TODO`
- GPU: `TODO`
- Notes: model training and cross-attention dump were executed on server

### Data transfer
- Episode data was transferred from local host to server via Termius/SCP workflow.
- Transfer command/script: `TODO`

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
# TODO: local collection command(s)

# [Server] copy uploaded episodes into training dataset path
# TODO: server-side import/arrange command(s)

# [Server] preprocessing (if used)
# TODO: preprocessing command(s)
```

## 4) Training
Run on remote server.

```bash
# [Server] full train command used for reported result
# TODO
```

## 5) Evaluation
Use exact checkpoint path. If inference was run on local robot host, separate it from server evaluation.

```bash
# [Server] offline eval command (if used)
# TODO

# [Local host] robot-side inference/eval command (if used)
# TODO

# [Server] cross-attention dump command
# TODO
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
