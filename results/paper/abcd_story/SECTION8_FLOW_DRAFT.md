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
- 즉, pretrained(P@A) 대비 A@A에서 wrist-cam 참조/객체 집중이 약화된 패턴이 관측된다.

### 8.3 Model B로 넘어간 이유와 학습 설정 (표 뒤 문단)

다음 문단 연결(본문용):
- 위 두 표를 함께 보면, A 모델은 접근은 가능하지만 집기 정밀도가 낮고(특히 strawberry), wrist-cam attention도 pretrained 대비 약화되어 있다.
- 이 한계를 보완하기 위해 pick->place 구간 중심의 근접 데이터를 추가 수집해 Model B를 학습했다.
- 구체적으로 `+300 episodes`(banana/socks/strawberry 각 100)를 추가해 `task_box_750 -> task_box_1050`으로 확장했다.
