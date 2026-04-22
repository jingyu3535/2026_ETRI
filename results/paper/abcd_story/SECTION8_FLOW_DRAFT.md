## 8) Results package (tables + figures) - draft flow

### 8.1 Model A baseline outcome (36 episodes)
- Figure/Table: `figures/TableFig_01_A_36ep_outcomes.png`
- Numeric source: `tables/Table_01_A_36ep_outcomes.csv`

핵심 메시지:
- A 모델은 접근 단계는 수행하지만, 집기(특히 strawberry)에서 성능 저하가 크다.
- 36ep 결과에서 strawberry 성공이 매우 낮고, socks 대비 난이도 차이가 크게 나타난다.

### 8.2 P@A vs A@A attention comparison (Model A analysis)
- Figure/Table: `figures/TableFig_02_PA_AA_image_mass_by_camera.png`
- Numeric source: `tables/Table_02_PA_AA_image_mass_by_camera.csv`
- Figure/Table: `figures/TableFig_03_PA_AA_object_ratio_by_camera.png`
- Numeric source: `tables/Table_03_PA_AA_object_ratio_by_camera.csv`

해석 문장(본문용):
- camera1(wrist) 기준, `image_mass`는 `0.291 -> 0.231 (-20.3%)`, `object_ratio`는 `0.142 -> 0.126 (-11.2%)`로 모두 감소했다.
- 동시에 `object_ratio`는 camera2(top)에서 `+47.0%` 상승했다.
- 즉, pretrained(P@A) 대비 A@A에서 top 참조는 강화되지만 wrist 기반 객체 참조는 약화된 패턴이 관측된다.
- 그런데 실제 집기 성능(특히 strawberry)은 여전히 낮아, wrist 영향을 많이 받는 정밀 집기 문제가 해결되지 않았다고 볼 수 있다.

### 8.3 Model B로 넘어간 이유와 학습 설정 (표 뒤 문단)

다음 문단 연결(본문용):
- 위 두 표를 함께 보면, A 모델은 접근은 가능하지만 집기 정밀도가 낮고(특히 strawberry), wrist-cam attention도 pretrained 대비 약화되어 있다.
- 보조 확인용 outcome 분할 표(`Table_11`, `Table_12`)를 보면, outcome별 추가 분리 근거는 강하지 않아 본문 핵심 근거로는 사용하지 않는다(특히 outcome3는 표본 `n=4`).
- 이 한계를 보완하기 위해 pick->place 구간 중심의 근접 데이터를 추가 수집해 Model B를 학습했다.
- 구체적으로 `+300 episodes`(banana/socks/strawberry 각 100)를 추가해 `task_box_750 -> task_box_1050`으로 확장했다.

### 8.4 Model B 학습 결과 (36ep 결과표)
- Numeric source: `tables/Table_04_B_36ep_outcomes.csv`

| object | 1(success) | 2(attempt fail) | 3(no approach) |
|---|---:|---:|---:|
| banana | 1 | 7 | 4 |
| socks | 9 | 3 | 0 |
| strawberry | 1 | 10 | 1 |

해석 문장(본문용):
- strawberry는 `0 -> 1`로 1회 성공이 나타났고, 근접 구간의 세밀한 동작은 정성적으로 개선된 면이 있다.
- 하지만 접근 단계에서 목표 object 인식 오류가 증가했다(예: banana 명령인데 wrist에 보이는 socks를 집는 사례).

### 8.5 B on-policy attention 변화 (B@A -> B@B)
- Numeric source: `tables/Table_05_BA_BB_image_mass_by_camera.csv`
- Numeric source: `tables/Table_06_BA_BB_object_ratio_by_camera.csv`

Image Mass table:
| camera | B@A | B@B | Δ(%) |
|---|---:|---:|---:|
| camera1 | 0.223 | 0.362 | +62.5% |
| camera2 | 0.405 | 0.276 | -32.0% |

Object Ratio table:
| camera | B@A | B@B | Δ(%) |
|---|---:|---:|---:|
| camera1 | 0.121 | 0.116 | -4.3% |
| camera2 | 0.029 | 0.017 | -40.5% |

해석 문장(본문용):
- B@B에서 image_mass는 camera1 급상승, camera2 급하락으로 재분배된다.
- object_ratio는 두 camera 모두 하락해 object grounding이 약화된 방향과 일치한다.
- 따라서 pick->place 구간 집중 학습의 단기 이득은 있으나, 전체 성공률/일반화 측면 개선은 추가 보완이 필요하다.
