# Artifact Policy (GitHub vs External)

## 결론
- GitHub에는 `코드만` 올리는 것이 아니라, 논문 주장 재현에 필요한 `최종 표/그래프`도 함께 두는 것이 맞습니다.
- 다만 `원본 대용량`은 외부 저장소에 두고, GitHub에는 접근/검증 정보를 남깁니다.

## 1) GitHub에 포함 (필수)
- 실험/분석 코드 (`src/`, `scripts/`)
- 실행/재현 문서 (`docs/paper_release/`, `PAPER_RELEASE.md`, `REPRO_EVIDENCE.md`)
- 논문 본문에 사용한 최종 표/그림 (`results/paper/tables`, `results/paper/figures`)
- 외부 아티팩트 인덱스 (`manifests/paper_external_artifacts.csv`)

## 2) GitHub에 포함 (권장)
- 대표 샘플 triptych/시계열 그림(압축된 PNG)
- 통계 요약 CSV(행 수가 크지 않은 파일)
- figure-to-path 매핑 문서

## 3) 외부 스토리지로 분리 (권장)
- raw attention dump 전체(`*.npz`, `*_meta.json` 대량)
- SAM2 mask 전체 프레임
- 원본 비디오/프레임 덤프
- 전체 학습 checkpoint

## 4) 외부 보관 시 GitHub에 남길 최소 정보
- 저장 위치(서버/경로)
- 생성 스크립트와 커맨드
- 샘플 수량(`npz/meta/frame`)과 간단 무결성(예: 개수, 날짜, SHA256 일부)
- 라이선스/접근 권한 메모(회사 내부 여부)

## 5) 권장 디렉터리 규약
- `results/paper/tables`: 본문/부록 표 원본 CSV
- `results/paper/figures`: 본문/부록 그림 PNG
- `manifests/`: 외부 원본 아티팩트 경로/설명 CSV

## 6) 리뷰 대응 관점 체크
- 논문에서 인용한 숫자는 GitHub에 있는 CSV에서 바로 재확인 가능해야 함
- 그림 파일명은 논문 figure 번호와 매핑되어야 함
- "코드는 공개됐지만 결과는 재현 불가" 상태를 피하도록 최소 요약 결과를 반드시 포함
