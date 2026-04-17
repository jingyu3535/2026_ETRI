# Experiment Workflow (Paper Order)

이 문서는 분석 논문 작성 기준으로 실험을 `정의 -> 실행 -> 검증 -> 패키징 -> 공개` 순서로 고정합니다.

## 0) 연구 질문/가설 고정
- 질문: action latent query의 cross-attention 변화가 실제 실패(outcome)와 연관되는가?
- 가설 H1: 근접구간 데이터 과주입은 wrist shortcut 편향을 키워 object grounding을 약화시킨다.
- 가설 H2: 성능 저하는 단순 존재 여부가 아니라 주입 강도(dose)에 비선형 임계가 있다.

추가로 하면 좋은 분석:
- H1/H2를 분리해 본문에서 각각 검증 가능한 관측량(`image_ratio`, `object_ratio`, outcome gap)으로 연결.

## 1) 코드/실험 버전 고정
- 브랜치: `paper_release`
- 재현 기준 문서: `REPRO_EVIDENCE.md`, `PAPER_RELEASE.md`
- 핵심 코드 위치:
  - `src/lerobot/policies/smolvla/configuration_smolvla.py`
  - `src/lerobot/policies/smolvla/modeling_smolvla.py`
  - `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
  - `scripts/dump_action_attn_eval.py`

추가로 하면 좋은 분석:
- 최종 논문 제출 SHA를 고정하고, 본문/부록에 동일 SHA를 명시.

## 2) 데이터셋 계보 고정 (A/B/C/D)
- A: `task_box_750`
- B: `task_box_1050` (근접구간 +300)
- C: `task_box_1050_toponly` (wrist 제거)
- D: `task_box_795` (근접구간 +45)

추가로 하면 좋은 분석:
- 조건별 episode/object 분포표를 한 장으로 통합해 데이터 편향을 먼저 공개.

## 3) D 조건(`task_box_795`) 생성
- 스크립트: `scripts/make_task_box_795_from_1050.py`
- 규칙: `episode_index >= 750`에서 task별 tail 15개 선택(총 45ep) 후 `task_box_750`과 merge.

추가로 하면 좋은 분석:
- merge 후 재인덱싱(750~794) 매핑표를 저장해 추적성 보장.

## 4) 학습 실행 고정
- 공통 하이퍼파라미터: `steps=500000`, `batch_size=32`, `lr=5e-5`, warmup/decay 고정.
- 카메라 매핑: `front/top -> camera1/camera2`, toponly는 `top -> camera1`.

추가로 하면 좋은 분석:
- 각 run의 `train_config` 해시 비교표를 남겨 실험 차이가 데이터셋/카메라 조건뿐임을 증명.

## 5) 평가 프레임셋 생성
- on-policy rollout 프레임을 생성하고, cross-model dump는 동일 stem 기준으로 비교.
- 공정 비교 핵심: 파일 stem 교집합 100% 확인.

추가로 하면 좋은 분석:
- episode 길이 분포(프레임 수) 차이를 같이 보고해 시간축 편향 가능성 점검.

## 6) Action->Prefix Cross-Attention Dump
- 저장물: `*_action_attn.npz`, `*_action_attn_meta.json`
- 현재 확인된 포맷 차이:
  - `A_A`: `dump_action_step=0`
  - `B/C/D`: `dump_action_step=all`

추가로 하면 좋은 분석:
- 동일 action step 기준으로 재덤프한 작은 샘플셋을 만들어 포맷 차이 민감도 점검.

## 7) SAM2 마스크 생성 및 QA
- seed 라벨 -> SAM2 추적 -> overlay 확인 -> 이름셋 검증.
- 중요 구분: `missing mask`와 `empty mask(0-only)`를 분리 집계.

추가로 하면 좋은 분석:
- empty mask 비율을 outcome/object별로 보고하고, 통계에서 가중/제외 민감도 함께 제시.

## 8) Heatmap + Ratio 계산
- 핵심 스크립트:
  - `scripts/make_hetmap_episode_grid.py`
  - `scripts/make_attention_ratio_table.py`
  - `scripts/summarize_attention_ratios_per_episode.py`
  - `scripts/summarize_attention_ratios.py`
  - `scripts/plot_ratio_7x8_sets.py`
  - `scripts/build_episode_triptych.py`
  - `scripts/build_mean_triptych.py`

추가로 하면 좋은 분석:
- long-table 평균 vs server-style 평균(표본 단위 차이)을 본문에서 명확히 분리 설명.

## 9) 통계 분석
- outcome 중심:
  - `scripts/analyze_outcome_soft_aa.py`
  - `scripts/compare_outcome_attention_aa.py`
  - `scripts/analyze_cell_gap_outcome.py`
- phase/핫스팟:
  - `scripts/build_joint_phase_boundaries.py`
  - `scripts/compute_pick_after_hotspot_scores.py`
  - `scripts/build_layer_metric_tables.py`

추가로 하면 좋은 분석:
- outcome=3 표본 수가 작으므로 exploratory 라벨 명시.
- permutation + bootstrap + effect size(Cliff's delta) 3축 보고 유지.

## 10) 그림/표 패키징
- 최소 필수 산출물:
  - 조건별 layer-curve(`image_ratio`, `object_ratio`)
  - outcome별 요약표/통계표
  - 대표 episode triptych
- 자동 생성: `scripts/export_paper_summary_assets.py`

추가로 하면 좋은 분석:
- 본문에 들어간 figure 번호와 파일 경로를 1:1 매핑한 인덱스 파일 생성.

## 11) D 조건 보강 분석
- 현재 메인 ratio 파이프라인은 `P/A/B/C` 중심.
- `D`는 dump는 준비되어 있으나, 동일 파이프라인 결과표에 아직 상시 포함되지 않음.

추가로 하면 좋은 분석:
- `D_A`를 `P/A/B/C`와 동일 포맷으로 집계해 dose-response 곡선을 완성.

## 12) GitHub 릴리즈
- 포함: 코드, 실행 커맨드, 최종 표/그림, external artifact manifest.
- 제외: raw dump/mask/video/checkpoint 원본.
- 릴리즈 태그: 예) `paper_release_v1`.

추가로 하면 좋은 분석:
- 태그 SHA와 논문 본문 버전(제출본/카메라레디본) 매핑표를 함께 관리.
