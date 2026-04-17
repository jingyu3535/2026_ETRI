## 8) Results package (tables + figures) - draft flow

### 문단형 초안 (본문 삽입용)

먼저 Model A를 baseline 데이터로 학습하고 36 episode 평가를 수행했다.
결과표(Table 01)에서 socks는 11/12로 상대적으로 높은 성공률을 보였지만, strawberry는 0/12로 실패가 집중되었다.
즉, 접근 자체는 가능한 경우가 많지만 작은 물체를 안정적으로 집는 정밀 조작 단계에서 한계가 확인되었다.

다음으로 행동 실패 패턴과 attention 변화를 연결하기 위해 pretrained 기준(P@A)과 A 모델(A@A)을 비교했다.
camera1(wrist) 기준으로 image_mass는 0.291에서 0.231로(-20.3%), object_ratio는 0.142에서 0.126으로(-11.2%) 감소했다.
pretrained 대비 wrist-cam의 객체 관련 참조가 약해진다는 신호로 해석할 수 있다.

정성 결과와 정량 attention 결과를 함께 고려해, 이후 단계에서는 접근 이후 구간(pick->place)에 해당하는 근접 데이터 추가 수집을 진행하고 Model B를 학습했다.
다음 절에서는 B 결과를 동일한 표 기반 형식으로 이어서 제시한다.

### 8.1 Model A baseline outcome (36 episodes)
- Figure/Table: `figures/TableFig_01_A_36ep_outcomes.png`
- Numeric source: `tables/Table_01_A_36ep_outcomes.csv`

핵심 메시지:
- A 모델은 접근 단계는 수행하지만, 집기(특히 strawberry)에서 성능 저하가 크다.
- 36ep 결과에서 strawberry 성공이 매우 낮고, socks 대비 난이도 차이가 크게 나타난다.

### 8.2 P@A vs A@A attention comparison (before Model B)
- Figure/Table: `figures/TableFig_02_PA_AA_image_mass_by_camera.png`
- Numeric source: `tables/Table_02_PA_AA_image_mass_by_camera.csv`
- Figure/Table: `figures/TableFig_03_PA_AA_object_ratio_by_camera.png`
- Numeric source: `tables/Table_03_PA_AA_object_ratio_by_camera.csv`

해석 문장(본문용):
- camera1(wrist) 기준, `image_mass`는 `0.291 -> 0.231 (-20.3%)`, `object_ratio`는 `0.142 -> 0.126 (-11.2%)`로 모두 감소했다.
- 즉, pretrained(P@A) 대비 A@A에서 wrist-cam 참조/객체 집중이 약화된 패턴이 관측된다.

다음 문단 연결(본문용):
- 정성적으로도 A 모델은 접근 자체는 가능하지만, 집는 동작(특히 작은 strawberry)에서 실패가 잦았다.
- 이를 보완하기 위해 접근 이후 구간(pick->place) 데이터를 추가 수집하여 Model B를 학습했다.
