# 8) Results Package (Current Draft)

This is the current step-by-step package for the paper storyline (A -> B -> C -> D), rebuilt from scratch in the requested order.

## Step 1. Model A baseline outcome (36 episodes)

Source files:
- `tables/Table_01_A_36ep_outcomes.csv`
- `figures/TableFig_01_A_36ep_outcomes.png`

| object | 1(success) | 2(attempt fail) | 3(no approach) |
|---|---:|---:|---:|
| banana | 4 | 7 | 1 |
| socks | 11 | 1 | 0 |
| strawberry | 0 | 9 | 3 |

Interpretation:
- A model can approach, but pick quality is weak (especially strawberry).

## Step 2. Attention comparison before Model B (P@A vs A@A)

### 2-1) Image Mass

Source files:
- `tables/Table_02_PA_AA_image_mass_by_camera.csv`
- `figures/TableFig_02_PA_AA_image_mass_by_camera.png`

| camera | P@A | A@A | Δ(%) |
|---|---:|---:|---:|
| camera1 | 0.291 | 0.231 | -20.3% |
| camera2 | 0.300 | 0.397 | +32.2% |

### 2-2) Object Ratio

Source files:
- `tables/Table_03_PA_AA_object_ratio_by_camera.csv`
- `figures/TableFig_03_PA_AA_object_ratio_by_camera.png`

| camera | P@A | A@A | Δ(%) |
|---|---:|---:|---:|
| camera1 | 0.142 | 0.126 | -11.2% |
| camera2 | 0.023 | 0.033 | +46.8% |

Interpretation:
- For camera1 (wrist), both `image_mass` and `object_ratio` decrease vs pretrained.
- This fits the narrative: approach is often possible, but precise picking is unstable; therefore additional pick/place-focused data was collected for Model B.
