# 2026_ETRI Paper Release

This repository is a paper-focused release based on LeRobot with local SmolVLA-related modifications.

## Base code
- Upstream project: `huggingface/lerobot`
- Base commit: `15724826`

## What is included
- Core package source: `src/`
- Dependencies and packaging files: `pyproject.toml`, `requirements-*.txt`, `setup.py`
- Experiment scripts used in this project: `scripts/`
- Paper release note: `PAPER_RELEASE.md`

## What is intentionally excluded
- CI/workflow configs: `.github/`
- Upstream docs/examples/tests/media/benchmarks/docker assets
- External dump backups and temporary debug files

## Quick setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[smolvla]"
```

## Run template
Train/eval command arguments depend on your dataset and hardware.
Use the project CLI entrypoints from this repo:

```bash
lerobot-train --help
lerobot-eval --help
```

## Notes
- Keep all credentials (HF/W&B/API tokens) in environment variables.
- Do not hardcode secrets in scripts.
