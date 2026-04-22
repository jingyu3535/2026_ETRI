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

## Step 2. P@A vs A@A attention (Model A analysis)

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
| camera2 | 0.023 | 0.033 | +47.0% |

Interpretation:
- `object_ratio` rises on top camera (`camera2`, `+47.0%`) but drops on wrist camera (`camera1`, `-11.2%`).
- Despite this shift, precise pick remains weak in Model A (strawberry `0/12`), so wrist-dependent fine manipulation is not reliably solved.

## Step 3. Why we moved to Model B (after tables)

After the two tables above, the flow is:
- behavior: Model A can often approach but struggles in precise picking (especially strawberry),
- attention: wrist camera (`camera1`) decreases in both metrics vs pretrained.

Supplementary note (outcome split, camera-specific only):
- `tables/Table_11_PA_AA_outcome_compare_camera1.csv`
- `tables/Table_12_PA_AA_outcome_compare_camera2.csv`
- Outcome-wise split is treated as supporting context only; it does not provide a strong extra separation by outcome beyond the main `P@A -> A@A` shift. (Outcome `3` is small: `n=4` episodes.)

To address this gap, we trained Model B with additional close-range pick/place-focused data:
- added episodes: `+300` (`banana 100`, `socks 100`, `strawberry 100`),
- dataset transition: `task_box_750 -> task_box_1050`.

## Step 4. Model B outcome (36 episodes, manual labels)

Source file:
- `tables/Table_04_B_36ep_outcomes.csv`

| object | 1(success) | 2(attempt fail) | 3(no approach) |
|---|---:|---:|---:|
| banana | 1 | 7 | 4 |
| socks | 9 | 3 | 0 |
| strawberry | 1 | 10 | 1 |

Interpretation:
- Strawberry pick appears (`0 -> 1`) and fine close-range motion is qualitatively better.
- But approach-stage recognition errors increase (e.g., banana command but socks in wrist view gets selected), reducing robustness.

## Step 5. B on-policy attention shift (B@A -> B@B)

Image Mass source:
- `tables/Table_05_BA_BB_image_mass_by_camera.csv`

| camera | B@A | B@B | Δ(%) |
|---|---:|---:|---:|
| camera1 | 0.223 | 0.362 | +62.5% |
| camera2 | 0.405 | 0.276 | -32.0% |

Object Ratio source:
- `tables/Table_06_BA_BB_object_ratio_by_camera.csv`

| camera | B@A | B@B | Δ(%) |
|---|---:|---:|---:|
| camera1 | 0.121 | 0.116 | -4.3% |
| camera2 | 0.029 | 0.017 | -40.5% |

Interpretation:
- Image attention is reallocated toward wrist camera on-policy (`camera1` up, `camera2` down).
- `object_ratio` decreases in both cameras, consistent with weaker grounding and lower real-task reliability.

## Step 6. Training-log reporting note

- Raw training logs are not inlined in this draft section.
- For paper submission, include a compact per-run summary table (run id, dataset, steps, LR, final loss, selected checkpoint, eval success), and keep full logs as appendix/artifact links.
