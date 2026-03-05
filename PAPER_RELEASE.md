# Paper Release Guide

This branch (`paper_release`) keeps the LeRobot codebase plus the local modifications used for experiments,
while excluding external dump files that were only collected for backup.

## Base
- Upstream project: `huggingface/lerobot`
- Base commit used for local work: `15724826`

## Included local changes
### Core code changes
- `src/lerobot/datasets/dataset_tools.py`
- `src/lerobot/datasets/lerobot_dataset.py`
- `src/lerobot/policies/smolvla/configuration_smolvla.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
- `src/lerobot/robots/so_follower/so_follower.py`
- `src/lerobot/scripts/lerobot_train.py`

### Experiment scripts
- `scripts/*.py`
- `scripts/tools/fix_dataset.py`
- `scripts/tools/test_wandb.py`

## Excluded from paper branch
- `inbox/` (copied files from Downloads, not part of reproducible code release)

## Recommended release practice
1. Pin environment and package versions in the paper appendix.
2. Add exact train/eval commands used for the reported numbers.
3. Keep secrets in environment variables only (no tokens in code).
