# Paper Release Guide

This branch is the reproducibility-focused release for the paper.

## Quick Links
- Ordered workflow: `docs/paper_release/01_EXPERIMENT_WORKFLOW.md`
- Release checklist: `docs/paper_release/02_RELEASE_CHECKLIST.md`
- Artifact policy: `docs/paper_release/03_ARTIFACT_POLICY.md`
- D dataset reproduction: `docs/paper_release/04_DATASET_D_TASK_BOX_795.md`
- Result package spec: `docs/paper_release/05_RESULTS_PACKAGE.md`
- Generated summary assets: `results/paper/`

## Base
- Upstream project: `huggingface/lerobot`
- Base commit used for local work: `15724826`

## Included
- Runtime package source in `src/`
- Local experiment scripts in `scripts/`
- Packaging/dependency files (`pyproject.toml`, `requirements-*.txt`, `setup.py`)

## Excluded
- `.github/` workflows and templates
- Upstream docs/examples/tests/benchmarks/media/docker assets
- External backup dumps and temporary debug-only files

## Core local code modifications
- `src/lerobot/datasets/dataset_tools.py`
- `src/lerobot/datasets/lerobot_dataset.py`
- `src/lerobot/policies/smolvla/configuration_smolvla.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
- `src/lerobot/robots/so_follower/so_follower.py`
- `src/lerobot/scripts/lerobot_train.py`

## Release checklist
1. Pin environment and package versions in the paper appendix.
2. Add exact train/eval commands used for reported metrics.
3. Keep all credentials in environment variables only.
