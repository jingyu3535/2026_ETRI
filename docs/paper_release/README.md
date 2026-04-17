# Paper Release Playbook

`paper_release` 브랜치를 논문 제출/리뷰 대응 가능한 형태로 정리하기 위한 실행 문서 모음입니다.

## 권장 진행 순서
1. [01_EXPERIMENT_WORKFLOW.md](./01_EXPERIMENT_WORKFLOW.md)
2. [04_DATASET_D_TASK_BOX_795.md](./04_DATASET_D_TASK_BOX_795.md)
3. [03_ARTIFACT_POLICY.md](./03_ARTIFACT_POLICY.md)
4. [05_RESULTS_PACKAGE.md](./05_RESULTS_PACKAGE.md)
5. [02_RELEASE_CHECKLIST.md](./02_RELEASE_CHECKLIST.md)

## 빠른 시작
```bash
# 권장: lerobot conda env
# /home/etri01/miniforge3/envs/lerobot/bin/python 사용 시 그래프까지 생성
python3 scripts/export_paper_summary_assets.py \
  --metrics_root /home/etri01/논문/eval/hetmap/metrics \
  --out_root /home/etri01/projects/lerobot/results/paper
```

생성 위치:
- `results/paper/tables`
- `results/paper/figures`

## 핵심 원칙
- GitHub에는 `코드 + 재현 커맨드 + 논문 본문에 들어가는 최종 표/그림`을 함께 둡니다.
- 대용량 원본(`raw dump/mask/video/checkpoint`)은 외부 스토리지에 두고, 리포에는 경로/무결성 정보만 남깁니다.

## 현재 확인된 주의사항
- dump 메타: `A_A`는 `dump_action_step=0`, `B/C/D`는 `dump_action_step=all`.
- `camera2` SAM2는 누락 파일보다 빈 mask(0-only) 프레임이 문제.
- 기본 ratio 파이프라인은 `P_A/A_A/B_A/C_A` 중심이며, `D_A`는 별도 추가 분석이 필요.
