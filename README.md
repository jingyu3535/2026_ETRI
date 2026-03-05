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
Fill these with exact values used in the paper.

- OS: `TODO`
- Python: `TODO`
- CUDA: `TODO`
- PyTorch: `TODO`
- GPU: `TODO`

## 2) Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[smolvla]"
```

## 3) Dataset preparation
Describe exactly how to obtain/prepare the dataset and include all commands.

```bash
# TODO: dataset download / collection command(s)
# TODO: preprocessing command(s)
```

## 4) Training
Use the exact command used for reported results.

```bash
# TODO: full train command
```

## 5) Evaluation
Use the exact checkpoint and evaluation command.

```bash
# TODO: full eval command
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
