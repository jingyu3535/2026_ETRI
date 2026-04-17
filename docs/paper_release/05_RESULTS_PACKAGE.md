# Results Package Spec

## 목적
논문 본문/부록에서 인용하는 핵심 수치를 GitHub 산출물로 바로 재검증할 수 있게 결과 패키지를 고정합니다.

## 출력 위치
- `results/paper/tables`
- `results/paper/figures`

## 핵심 표
1. `hypothesis_test_map.csv`
- H1~H4 가설과 대응 비교쌍/지표/표/그림 매핑

2. `T0_dual_camera_group_means.csv`
- `A_A/B_A/C_A/D_A/B_B`의 camera별 전역 평균(`image_mass/object_mass_est/image_ratio/object_ratio`)

3. `T1_fixed_frame_AA_to_BA.csv`
- 고정 프레임 모델 변화(`A@A -> B@A`) 비교

4. `T2_onpolicy_BA_to_BB.csv`
- on-policy 상태 변화(`B@A -> B@B`) 비교

5. `T3_prelim_AA_to_CA_DA.csv`
- `C@A`, `D@A`의 예비(preliminary) 비교

6. `T4_topcam_group_means_PA_AA_BA_CA.csv`
- top-camera 기준 `P@A/A@A/B@A/C@A` 전역 평균

7. `T4_topcam_transition_matrix_PA_AA_BA_CA.csv`
- `AA/PA`, `BA/AA`, `CA/BA` 변화율 매트릭스

8. `T5_topcam_outcome_matrix_PA_AA_BA_BB.csv`
- outcome(1/2/3) 기준 `P@A/A@A/B@A/B@B` 요약(가능 범위 지표)

9. `T6_b_only_outcome_summary.csv`
- `B@B` outcome summary 원본(카메라별)

## 핵심 그림
1. `G1_fixed_frame_AA_to_BA_pct.png`
2. `G2_onpolicy_BA_to_BB_pct.png`
3. `G3_prelim_ablation_object_ratio.png`
4. `G4_prelim_dose_image_mass.png`
5. `G5_layer_profile_object_ratio_abcd.png`
6. `G6_layer_profile_image_mass_abcd.png`
7. `G7_topcam_object_ratio_groups.png`
8. `G8_topcam_image_ratio_groups.png`

## 보조/검증 표
- `dump_inventory_summary.csv`
- `aa_mask_quality_summary.csv`
- `aa_outcome_counts.csv`
- `aa_outcome_significance_overall.csv`
- `pair_delta_long_all.csv`
- `export_summary.json`

## 자동 생성 커맨드
```bash
# 권장: /home/etri01/miniforge3/envs/lerobot/bin/python
python3 scripts/export_paper_summary_assets.py \
  --paper_eval_root /home/etri01/paper/eval \
  --out_root /home/etri01/projects/lerobot/results/paper
```

## 본문 연결 규칙
- 논문 table/figure 번호와 파일 경로를 1:1 매핑
- 숫자 인용 시 CSV 파일명과 열 이름까지 같이 명시
- `C@A`, `D@A`는 현재 `preliminary` 라벨을 유지
