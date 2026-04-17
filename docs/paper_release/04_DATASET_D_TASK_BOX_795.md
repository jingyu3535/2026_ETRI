# D Dataset Reproduction (`task_box_795`)

## 목적
`task_box_750 + (task_box_1050의 추가 300 중 task별 tail 15)`를 합쳐 `task_box_795`를 생성합니다.

## 선택 규칙
- source: `task_box_1050` (`episode_index: 0..1049`)
- 추가분 정의: `episode_index >= 750` (300ep)
- 선택: task별 마지막 15개
- 원본 1050 기준 기대 범위:
  - banana: `835..849` (15)
  - socks: `935..949` (15)
  - strawberry: `1035..1049` (15)

## 스크립트
- 경로: `scripts/make_task_box_795_from_1050.py`

## 실행 커맨드
```bash
export PYTHONPATH=/home/etri01/projects/lerobot/src
python3 scripts/make_task_box_795_from_1050.py \
  --root /home/internship/data/etri01 \
  --src_750 task_box_750 \
  --src_1050 task_box_1050 \
  --tmp_45 task_box_45_from1050 \
  --dst_795 task_box_795 \
  --added_start 750 \
  --tail_per_task 15 \
  --overwrite
```

## 내부 처리
1. `task_box_1050`에서 `episode_index >= 750` 추출
2. task별 tail 15 선택(총 45)
3. 임시셋 `task_box_45_from1050` 생성
4. `task_box_750` + 임시셋 merge
5. 최종 `task_box_795` 생성

## 검증 포인트
- 최종 경로: `/home/internship/data/etri01/task_box_795`
- 구조: `data/`, `meta/`, `videos/`
- `info.json`의 `total_episodes == 795`
- merged dataset 마지막 45개(task 분포): `15/15/15`

## 주의사항
- merge 후 `episode_index`는 재매핑되므로 원본 인덱스와 1:1 동일하지 않습니다.
- `dataset_tools.py` 패치 버전(타입 혼합 이슈 회피)이 필요할 수 있습니다.
