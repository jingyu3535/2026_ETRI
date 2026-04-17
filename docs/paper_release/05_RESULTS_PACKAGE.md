# Results Package Spec

## 목적
논문 본문/부록에서 주장하는 수치를 GitHub에서 바로 재확인할 수 있게, 최소 결과 패키지를 고정합니다.

## 출력 위치
- `results/paper/tables`
- `results/paper/figures`

## 필수 표 (최소)
1. `abcd_global_metric_summary.csv`
- 그룹별(`P_A/A_A/B_A/C_A`) `object_ratio`, `image_ratio` 전역 평균/표준편차/표본수

2. `abcd_layer_metric_summary.csv`
- 그룹 x 레이어별 평균/표준편차/표본수
- 레이어 민감도(어느 레이어에서 변화가 큰지) 확인용

3. `aa_outcome_metric_summary.csv`
- `A_A` outcome별(`1/2/3`) 요약

4. `aa_outcome_counts.csv`
- outcome 표본 수(저표본 위험 공시)

5. `aa_mask_quality_summary.csv`
- `missing` vs `empty` mask 현황

6. `dump_inventory_summary.csv`
- A/B/C/D/P dump 수량/포맷 메타(steps, layers, action_step)

## 필수 그림 (최소)
1. `abcd_layer_object_ratio.png`
2. `abcd_layer_image_ratio.png`
3. `aa_outcome_object_ratio.png`
4. `aa_outcome_image_ratio.png`

## 자동 생성 커맨드
```bash
# 권장: /home/etri01/miniforge3/envs/lerobot/bin/python
python3 scripts/export_paper_summary_assets.py \
  --metrics_root /home/etri01/논문/eval/hetmap/metrics \
  --out_root /home/etri01/projects/lerobot/results/paper
```

## D 조건 관련
- 현재 기본 ratio long-table은 `P/A/B/C` 중심입니다.
- `D`는 dump inventory로 우선 관리하고, 동일 포맷 ratio 집계가 준비되면 본 표/그림에 확장합니다.

## 본문 연결 규칙
- 논문 figure/table 번호와 파일 경로를 1:1 매핑
- 숫자 인용 시 CSV 파일명+열 이름까지 명시
