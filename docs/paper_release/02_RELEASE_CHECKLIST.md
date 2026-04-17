# Release Checklist

## A. 브랜치/버전
- [ ] 작업 브랜치가 `paper_release`인지 확인
- [ ] 최종 push 대상 remote/branch 확인
- [ ] 논문 본문에 들어갈 commit SHA 확정

## B. 데이터/조건 정의
- [ ] A/B/C/D 조건 정의 문서화 완료
- [ ] `task_box_795` 생성 절차와 검증 로그 포함
- [ ] 조건별 episode/object 분포표 저장

## C. dump/마스크 품질
- [ ] dump 디렉터리별 `npz/meta` 수량 확인
- [ ] `dump_action_step`, `layers`, `steps`, `heads` 메타 고정
- [ ] mask 품질표(`missing` vs `empty`) 생성

## D. 지표/통계
- [ ] `object_ratio`, `image_ratio` 요약표 생성
- [ ] outcome별 통계표(`delta`, `p`, `q`, effect size) 포함
- [ ] outcome=3 저표본 해석 제한 문구 반영

## E. 그림
- [ ] 조건별 layer 곡선 그림 생성
- [ ] 대표 episode triptych 그림 선별
- [ ] 논문 figure 번호와 파일 경로 매핑 완료

## F. GitHub 아티팩트 구성
- [ ] 코드/스크립트/문서 반영
- [ ] 최종 표/그림(`results/paper`) 반영
- [ ] 대용량 원본은 제외하고 `manifests/`에 경로+설명 기록

## G. 재현 커맨드
- [ ] 실행 커맨드가 문서에서 그대로 동작하는지 스모크 테스트
- [ ] 기본 경로(`/home/etri01/논문/eval/...`)와 대체 경로 안내 병기

## H. 최종 검수
- [ ] `git status`에서 의도치 않은 민감정보/임시파일 제외 확인
- [ ] README 진입 경로(Playbook/Results) 링크 정상
- [ ] 태그 생성 전 최종 diff 검토
