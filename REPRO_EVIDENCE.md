# Reproducibility Evidence Log

This file tracks evidence used to avoid memory-only documentation.

## Verified (high confidence)

1. Local host runtime stack (`2026-03-05`)
- Source: user-provided local shell output (`hostname`, `os-release`, `python -V`, `nvidia-smi`, `torch`)
- Values:
  - host: `etri01`
  - OS: `Ubuntu 22.04.5 LTS`
  - Python: `3.10.19`
  - PyTorch/CUDA: `2.7.0+cu118 / 11.8`
  - GPU: `NVIDIA GeForce GTX TITAN X`, driver `470.256.02`

2. Remote training server runtime stack (`2026-03-05`)
- Source: user-provided server shell output (`hostname`, `os-release`, `python -V`, `nvidia-smi`, `torch`)
- Values:
  - host: `user`
  - OS: `Ubuntu 24.04.3 LTS`
  - Python: `3.10.19`
  - PyTorch/CUDA: `2.6.0+cu124 / 12.4`
  - GPU: `NVIDIA H200 NVL`, driver `570.195.03`

3. Main training command (`task_box_1050`)
- Source: `/home/etri01/model/smolVLA_task_box_1050/train_log_full_20260213_175207.log:21`
- Source: `/home/etri01/model/smolVLA_task_box_1050/train_log_full_20260213_175207.log:35`
- Key args seen in log:
  - `--dataset.repo_id=task_box_1050`
  - `--dataset.root=/home/internship/data/etri01/task_box_1050`
  - `--steps=500000`
  - `--batch_size=32`
  - `--rename_map={"observation.images.front":"observation.images.camera1","observation.images.top":"observation.images.camera2"}`
  - `--policy.optimizer_lr=5e-5`
  - `--policy.scheduler_warmup_steps=15000`
  - `--policy.scheduler_decay_steps=500000`

4. Main run runtime summary (`smolVLA_task_box_1050`)
- Source: `/home/etri01/model/smolVLA_task_box_1050/train_log_full_20260213_175207.log:195`
- Source: `/home/etri01/model/smolVLA_task_box_1050/train_log_full_20260213_175207.log:199`
- Values:
  - output dir: `/home/internship/model/smolVLA_task_box_1050`
  - `cfg.steps=500000`
  - `dataset.num_frames=421143`
  - `dataset.num_episodes=1050`
  - effective batch size: `32`

5. Run config snapshot (`smolVLA_task_box_750`, checkpoint `500000`)
- Source: `/home/etri01/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model/train_config.json:3`
- Source: `/home/etri01/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model/train_config.json:4`
- Source: `/home/etri01/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model/train_config.json:182`
- Source: `/home/etri01/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model/train_config.json:185`
- Source: `/home/etri01/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model/train_config.json:187`
- Source: `/home/etri01/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model/train_config.json:188`
- Source: `/home/etri01/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model/train_config.json:157`
- Source: `/home/etri01/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model/train_config.json:165`
- Source: `/home/etri01/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model/train_config.json:166`
- Source: `/home/etri01/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model/train_config.json:233`
- Source: `/home/etri01/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model/train_config.json:235`
- Values:
  - dataset repo/root: `task_box_100`, `/home/internship/data/etri01/task_box_100`
  - output dir: `/home/internship/model/smolVLA_task_box_750`
  - seed/batch/steps: `1000` / `32` / `500000`
  - lr/warmup/decay: `5e-5` / `15000` / `500000`
  - rename_map: `front->camera1`, `top->camera2`

6. Run config snapshot (`smolVLA_task_box_1050`, checkpoint `500000`)
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:3`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:4`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:191`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:194`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:196`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:197`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:166`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:174`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:175`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:242`
- Source: `/home/etri01/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/train_config.json:244`
- Values:
  - dataset repo/root: `task_box_1050`, `/home/internship/data/etri01/task_box_1050`
  - output dir: `/home/internship/model/smolVLA_task_box_1050`
  - seed/batch/steps: `1000` / `32` / `500000`
  - lr/warmup/decay: `5e-5` / `15000` / `500000`
  - rename_map: `front->camera1`, `top->camera2`

7. Run config snapshot (`smolVLA_task_box_1050_toponly`, checkpoint `500000`)
- Source: `/home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model/train_config.json:3`
- Source: `/home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model/train_config.json:4`
- Source: `/home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model/train_config.json:191`
- Source: `/home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model/train_config.json:194`
- Source: `/home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model/train_config.json:196`
- Source: `/home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model/train_config.json:197`
- Source: `/home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model/train_config.json:166`
- Source: `/home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model/train_config.json:174`
- Source: `/home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model/train_config.json:175`
- Source: `/home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model/train_config.json:242`
- Source: `/home/etri01/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model/train_config.json:243`
- Values:
  - dataset repo/root: `task_box_1050_toponly`, `/home/internship/data/etri01/task_box_1050_toponly`
  - output dir: `/home/internship/model/smolVLA_task_box_1050_toponly`
  - seed/batch/steps: `1000` / `32` / `500000`
  - lr/warmup/decay: `5e-5` / `15000` / `500000`
  - rename_map: `top->camera1`

8. Local robot data collection command (SO101 leader/follower)
- Source: `/home/etri01/.bash_history:1185`

9. Local robot-side inference/evaluation command variant
- Source: `/home/etri01/.bash_history:1627`

10. Attention dump command evidence
- Source: `/home/etri01/.bash_history:2000`
- Source: user-provided server `~/.bash_history` extraction (`dump_action_attn_eval`, lines `1934`, `1935`)

11. Data transfer workflow evidence
- Source: `/home/etri01/.bash_history:1159` (SCP via jump host)
- Source: `/home/etri01/.bash_history:1682` and `:1692` (rsync)

12. Local checkpoint config file hashes
- Source: `sha256sum` run on local copied artifacts (`2026-03-05`)
- Values:
  - `smolVLA_task_box_750`: `5451b01bf20c5289f1df724eaacede7e41fae9a5ef28b1b6485d885b5ef4198b`
  - `smolVLA_task_box_1050`: `4afe252ce4416ed3a63d41e8f746466f20808748b22e13b605f4f0f47c3ae198`
  - `smolVLA_task_box_1050_toponly`: `a7fddcae36ba6311657534244f7bb8af025817e8a65dbf42404d2cad9792676e`

13. Language prompt distribution in `task_box_1050` dataset
- Source files:
  - `/home/etri01/.cache/huggingface/lerobot/etri01/task_box_1050/meta/episodes/chunk-000/file-*.parquet`
  - Parsed via local Python (`pyarrow`) on `2026-03-05`
- Values:
  - `pick the banana and put it in the transparent box`: `100`
  - `pick the socks and put it in the transparent box`: `100`
  - `pick the strawberry and put it in the transparent box`: `100`
  - `pick the banana and put it in the blue box`: `234`
  - `pick the socks and put it in the blue box`: `250`
  - `pick the strawberry and put it in the blue box`: `266`
  - total episodes: `1050`

14. Intermediate 750-setting decomposition used in documentation
- Source: derived from item 13 counts plus user-provided experiment note
- Values:
  - transparent stage: `300` (`100` each object)
  - additional blue stage: `450` (`134/150/166`)
- Note: this is a derived breakdown for narrative consistency (`300 + 450 = 750`), while final `task_box_1050` blue totals are `234/250/266`.

15. Server-side training time evidence for all three runs
- Source: user-provided server `stat` output on `2026-03-05`:
  - `/home/internship/model/smolVLA_task_box_750/checkpoints/020000/pretrained_model/model.safetensors`
  - `/home/internship/model/smolVLA_task_box_750/checkpoints/500000/pretrained_model/model.safetensors`
  - `/home/internship/model/smolVLA_task_box_750/checkpoints/500000/training_state/training_step.json`
  - `/home/internship/model/smolVLA_task_box_1050/checkpoints/020000/pretrained_model/model.safetensors`
  - `/home/internship/model/smolVLA_task_box_1050/checkpoints/500000/pretrained_model/model.safetensors`
  - `/home/internship/model/smolVLA_task_box_1050/checkpoints/500000/training_state/training_step.json`
  - `/home/internship/model/smolVLA_task_box_1050_toponly/checkpoints/020000/pretrained_model/model.safetensors`
  - `/home/internship/model/smolVLA_task_box_1050_toponly/checkpoints/500000/pretrained_model/model.safetensors`
  - `/home/internship/model/smolVLA_task_box_1050_toponly/checkpoints/500000/training_state/training_step.json`
- Values:
  - `750/020000`: `2026-01-23 19:19:18 +0900`
  - `750/500000`: `2026-01-25 03:09:15 +0900`
  - `750/training_step.json`: `2026-01-25 03:09:15 +0900`
  - `1050/020000`: `2026-02-13 19:03:22 +0900`
  - `1050/500000`: `2026-02-15 00:51:43 +0900`
  - `1050/training_step.json`: `2026-02-15 00:51:43 +0900`
  - `1050_toponly/020000`: `2026-02-25 20:01:58 +0900`
  - `1050_toponly/500000`: `2026-02-27 16:55:53 +0900`
  - `1050_toponly/training_step.json`: `2026-02-27 16:55:54 +0900`
- Interpretation: date anchors for all runs are based on server checkpoint timestamps.

16. `toponly` train command presence without shell timestamp
- Source: user-provided server `~/.bash_history` extraction on `2026-03-05`
- Value:
  - `NO_TIMESTAMP | cd /home/internship/projects/lerobot && lerobot-train ... --dataset.repo_id=task_box_1050_toponly ...`
- Interpretation: command exists, but shell-history timestamp metadata was unavailable; server checkpoint `stat` times are the reliable date anchor.

17. Data-collection command variants in local history
- Source: `/home/etri01/.bash_history` (`lerobot-record` lines), parsed on `2026-03-05`
- Aggregate counts:
  - total `lerobot-record` entries scanned: `112`
  - `--dataset.repo_id=etri01/blue_box_data`: `27`
  - `--dataset.repo_id=etri01/eval_task_box_1050_cam2only`: `7`
  - `--manual_advance=true`: `21`
  - `--resume=true`: `27`
  - `--resume=false`: `7`
  - camera path style: `/dev/v4l/by-id/*` (`29`), `/dev/video*` (`1`), numeric index style (`4`)
  - OpenCV backend override text appears in a subset of entries (`3`)
- Interpretation:
  - The command evolved during development.
  - README uses a canonical late-stage collection command (stable `/dev/v4l/by-id` paths) plus explicit variant notes.

18. Training default-vs-used hyperparameter evidence
- Default-value sources:
  - `/home/etri01/projects/lerobot/src/lerobot/configs/train.py:55` (`batch_size=8`)
  - `/home/etri01/projects/lerobot/src/lerobot/configs/train.py:56` (`steps=100000`)
  - `/home/etri01/projects/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py:89` (`optimizer_lr=1e-4`)
  - `/home/etri01/projects/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py:95` (`scheduler_warmup_steps=1000`)
  - `/home/etri01/projects/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py:96` (`scheduler_decay_steps=30000`)
- Used-value sources:
  - `500000` run `train_config.json` files for `smolVLA_task_box_750`, `smolVLA_task_box_1050`, `smolVLA_task_box_1050_toponly`
  - user-provided train command blocks (`lerobot-train ... --batch_size=32 --steps=500000 --policy.optimizer_lr=5e-5 --policy.scheduler_warmup_steps=15000 --policy.scheduler_decay_steps=500000`)
- Used values (common across runs):
  - `batch_size=32`, `steps=500000`
  - `policy.optimizer_lr=5e-5`
  - `policy.scheduler_warmup_steps=15000`
  - `policy.scheduler_decay_steps=500000`
  - `rename_map` explicitly set per run (`front/top -> camera1/camera2` or `top -> camera1`)

19. Action-to-image cross-attention dump implementation evidence
- Config flag sources:
  - `/home/etri01/projects/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py:72`
  - `/home/etri01/projects/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py:78`
- Capture-flow sources:
  - `/home/etri01/projects/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:323` (`predict_action_chunk_with_attn`)
  - `/home/etri01/projects/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:909` (`dump_action_attn_denoise_step`)
  - `/home/etri01/projects/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:924` (`set_action_attn_capture`)
  - `/home/etri01/projects/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:949` (`pop_action_attn_buffer`)
- Attention tensor source:
  - `/home/etri01/projects/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:568`
  - `/home/etri01/projects/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:617`
  - `/home/etri01/projects/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:618`
- Dump file schema source:
  - `/home/etri01/projects/lerobot/scripts/dump_action_attn_eval.py:190`
  - `/home/etri01/projects/lerobot/scripts/dump_action_attn_eval.py:412`
  - `/home/etri01/projects/lerobot/scripts/dump_action_attn_eval.py:438`
  - `/home/etri01/projects/lerobot/scripts/dump_action_attn_eval.py:460`
- Confirmed behavior:
  - Script loops over selected `layers x denoise_steps x action_step` and saves per-combination dump files.
  - `.npz` stores `attn` (and optional `lang_attn`), with metadata in paired `*_meta.json`.
  - `--heads` is accepted but restricted to `mean` in this codebase.

20. SAM2-large mask generation pipeline for `eval_task_box_1050`
- Source type:
  - User-provided command transcript (`2026-03-06`) for end-to-end mask pipeline.
  - Local script argument verification in this repo.
- Script/flag verification sources:
  - `/home/etri01/projects/lerobot/scripts/seg_label_tool.py:9`
  - `/home/etri01/projects/lerobot/scripts/sam2_vos_test.py:12`
  - `/home/etri01/projects/lerobot/scripts/sam2_vos_test.py:28`
  - `/home/etri01/projects/lerobot/scripts/sam2_vos_test.py:30`
  - `/home/etri01/projects/lerobot/scripts/make_sam2_overlay_check.py:10`
- Pipeline summary recorded in README:
  - Dataset root: `/home/etri01/model/eval_task_box_1050`
  - Camera convention (analysis labels): `camera1=front`, `camera2=top`
  - Note: physical device/vendor mapping for `front/top` is host-dependent and may swap across sessions
  - Seeding: 5-bin frame sampling per episode + extra first-object seeds for selected camera1 episodes
  - SAM2 inference: `sam2.1_hiera_l.yaml` + `sam2.1_hiera_large.pt`, `fill_before_prompt`, `out_index_base=1`
  - QA: overlay generation and frame/mask filename-set equality check
- Note:
  - These commands are documented as the experiment procedure supplied by the user.
  - Re-execution logs of this full SAM2 chain are not yet attached in this repository.

## Partial / needs confirmation

1. Remote server exact package lock state at training time
- `train_config.json` does not replace full `conda env export` or `pip freeze`.

2. Final paper metric table values
- Task success metrics and confidence intervals are not yet inserted in README.

3. Byte-level equality between local copied checkpoints and server originals
- Local hashes are recorded.
- Server-side file times are now confirmed for `toponly`, but direct local-vs-server SHA comparison is still not recorded.

## Recommendation

Before submission, archive:
- `conda env export` (or `pip freeze`) for local and server
- Final metric CSV or raw evaluation logs used in paper table
- Remote SHA256 for the same `train_config.json` files to pair with local hashes
