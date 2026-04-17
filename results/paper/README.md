# Paper Result Assets

This directory stores paper-facing summary artifacts only.

- `tables/`: compact CSVs for claims and appendix checks
- `figures/`: compact PNGs for manuscript insertion

Main export command:
```bash
# Recommended Python: /home/etri01/miniforge3/envs/lerobot/bin/python
python3 scripts/export_paper_summary_assets.py \
  --paper_eval_root /home/etri01/paper/eval \
  --out_root /home/etri01/projects/lerobot/results/paper
```

Core tables:
- `hypothesis_test_map.csv`
- `T0_dual_camera_group_means.csv`
- `T1_fixed_frame_AA_to_BA.csv`
- `T2_onpolicy_BA_to_BB.csv`
- `T3_prelim_AA_to_CA_DA.csv`
- `T4_topcam_group_means_PA_AA_BA_CA.csv`
- `T4_topcam_transition_matrix_PA_AA_BA_CA.csv`
- `T5_topcam_outcome_matrix_PA_AA_BA_BB.csv`
- `T6_b_only_outcome_summary.csv`

Core figures:
- `G1_fixed_frame_AA_to_BA_pct.png`
- `G2_onpolicy_BA_to_BB_pct.png`
- `G3_prelim_ablation_object_ratio.png`
- `G4_prelim_dose_image_mass.png`
- `G5_layer_profile_object_ratio_abcd.png`
- `G6_layer_profile_image_mass_abcd.png`
- `G7_topcam_object_ratio_groups.png`
- `G8_topcam_image_ratio_groups.png`

Storyline package (A->B->C/D):
- `abcd_story/tables/Table_01_model_design_abcd.csv`
- `abcd_story/tables/Table_02_main_transition_metrics.csv`
- `abcd_story/tables/Table_03_preliminary_cd_metrics.csv`
- `abcd_story/tables/Table_04_A_36ep_outcome_by_object.csv`
- `abcd_story/tables/Table_06_B_36ep_outcome_by_object_manual.csv`
- `abcd_story/tables/Table_07_A_vs_B_36ep_outcome_by_object.csv`
- `abcd_story/tables/Table_08_stage_series_AA_BA_BB.csv`
- `abcd_story/figures/Fig_03_main_transition_lines.png`
- `abcd_story/figures/Fig_04_progress_timeseries_AA_BA_BB.png`
- `abcd_story/figures/Fig_05_progress_timeseries_AA_CA_DA_topcam.png`
- Insertion guide: `ABCD_STORY_INSERTION.md`
