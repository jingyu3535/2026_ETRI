# Metric Definitions and Aggregation

이 문서는 `image_ratio/object_ratio/object_mass` 계산식을 논문/리포지토리 기준으로 고정한다.

## 1) Source of truth
- `scripts/make_attention_ratio_table.py`
- `scripts/export_paper_summary_assets.py`

## 2) Row-level definitions
Action-to-prefix cross-attention row에서:
- `image_mass = sum(attn over selected camera image tokens)`
- `total_mass = sum(attn over full prefix tokens)`
- `image_ratio = image_mass / total_mass`
- `object_mass = sum(attn_token * coverage_token)`
- `object_ratio = object_mass / image_mass`

## 3) Soft coverage rule
- 기본 설정: `mask_mode=soft`, `mask_pad_size=512`
- SAM2 binary mask를 `512x512`로 resize+pad 후 token grid로 축소한다.
- Topcam 분석 기본은 `8x8` token grid.
- `coverage_token`은 `[0,1]` 연속값이며 object weighting에 그대로 사용한다.

## 4) Missing/empty mask handling
- 마스크 파일이 없거나 all-zero이면 해당 row의 `object_mass/object_ratio`는 `NaN`.
- `image_mass/image_ratio`는 attention만으로 계속 계산한다.

## 5) Aggregation used in paper tables/figures
- Packed dumps (`A_A/B_A/C_A/D_A/B_B`):
  - 선택된 denoise step, action query를 평균해 prefix-level attention 생성.
- `P_A` base dumps:
  - per-step dump를 합친 뒤 동일 prefix-level 형태로 평균.
- 결과 표는 이 row-level metric을 다시 group/camera/outcome 단위로 평균한 값이다.

## 6) Reporting convention
- 본문 메인: `image_ratio` + `object_ratio`
- 보조: `object_mass`
- `image_mass`는 CSV 호환성 때문에 유지하되, 현재 파이프라인에서는 `total_mass ~= 1`이라 `image_ratio`와 수치가 거의 같다.

## 7) Repro command (core)
```bash
/home/etri01/miniforge3/envs/lerobot/bin/python scripts/make_attention_ratio_table.py \
  --render_mode smooth \
  --mask_mode soft \
  --mask_pad_size 512
```
