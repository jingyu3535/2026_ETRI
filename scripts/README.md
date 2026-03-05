# Local Scripts Index

This folder contains local experimentation scripts that are not part of the core `src/lerobot` package.

## Attention / visualization
- `attn_roi.py`: ROI-based attention visualization utility.
- `attn_track.py`: Tracks attention over language/image tokens.
- `dump_action_attn_eval.py`: Dumps attention values from evaluation runs.
- `realtime_action_attn.py`: Real-time action attention visualization (offline environment).
- `realtime_action_attn_robot.py`: Real-time action attention visualization (robot runtime).

## Segmentation / labeling
- `make_sam2_overlay_check.py`: SAM2 overlay check helper.
- `sam2_vos_test.py`: SAM2 VOS test runner.
- `sam2_vos_rerun_empty.py`: Re-run helper for empty SAM2/VOS outputs.
- `seg_label_tool.py`: Segmentation label helper tool.
- `roi_label_tool.py`: ROI label helper tool.

## Utilities
- `tools/fix_dataset.py`: Injects/fixes task label metadata for a local dataset copy.
- `tools/test_wandb.py`: Minimal W&B connectivity test (requires `WANDB_API_KEY` env var).

## Note
- Keep secrets out of source files (API keys/tokens should be environment variables only).
