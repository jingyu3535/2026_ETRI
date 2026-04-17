# Paper Result Assets

이 디렉터리는 논문에 직접 연결되는 요약 산출물만 저장합니다.

- `tables/`: 최종 통계/요약 CSV
- `figures/`: 본문/부록용 PNG

생성 스크립트:
```bash
# 권장: /home/etri01/miniforge3/envs/lerobot/bin/python
python3 scripts/export_paper_summary_assets.py \
  --metrics_root /home/etri01/논문/eval/hetmap/metrics \
  --out_root /home/etri01/projects/lerobot/results/paper
```
