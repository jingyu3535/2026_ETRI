## 8) Results package (tables + figures) - draft flow

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
