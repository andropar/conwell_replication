# Results Guide

Use this as the entry point for plotting or inspecting the completed outputs.

## Polished Report Outputs

- `reports/laion_fmri_conwell_fig2_4_report.md`: narrative report for the
  current Fig. 2-4 controlled-comparison readout.
- `reports/laion_fmri_conwell_fig2_4_score_summary.csv`: compact score summary
  across split families.
- `reports/laion_fmri_conwell_fig2_4_effects.csv`: compact controlled-effect
  table.
- `reports/laion_fmri_conwell_fig2_4_taskonomy_endpoints.csv`: Taskonomy
  endpoint summary.
- `reports/figures/`: PNG/PDF figures embedded in the report.

Regenerate these report-level plots and Markdown with:

```bash
python scripts/build_fig2_4_report.py
```

## Main Replotting Tables

The lower-level analysis outputs live under `figures/`. These are the best
files to point someone to if they want to replot rather than rerun evaluation.

Split-half:

- `figures/splithalf_results/`
- `figures/splithalf_results_ood/`

Generalization splits:

- `figures/min_nn_results/shared_random_agg/`
- `figures/min_nn_results/subject_random_agg/`
- `figures/min_nn_results/shared_cluster_k5_agg/`
- `figures/min_nn_results/subject_cluster_k5_agg/`
- `figures/min_nn_results/shared_tau/`
- `figures/min_nn_results/subject_tau/`
- `figures/min_nn_results/shared_ood/`
- `figures/min_nn_results/subject_ood/`

Within each analysis directory, the main CSVs are:

- `model_metric_subject_scores.csv`: subject-level best-layer test scores.
- `model_metric_group_summary.csv`: grouped summary used for model-family
  comparisons.
- `controlled_figure_model_stats.csv`: model-level table behind the Fig. 2-4
  controlled-comparison plots.
- `controlled_comparison_model_counts.csv`: coverage/count audit for plotted
  comparisons.
- `noise_ceiling_summary.csv`: noise-ceiling values used as plot references.
- `statistics/*.csv`: statistical-test inputs, descriptives, coverage, model
  rankings, and breakpoint analyses.

The matching already-rendered Conwell-style plots are in the same directories:

- `fig2_architecture_variation.png/.pdf`
- `fig3_task_variation.png/.pdf`
- `fig4_input_variation.png/.pdf`

## Other Useful Outputs

- `figures/min_nn_results/combo_metric_summary.csv`: compact summary across
  min-nn split/pool combinations.
- `figures/roi_stats_summary/`: ROI-level summaries.
- `figures/diag1_cross_roi_rdm_corr.csv` and
  `figures/diag2_ood_effect_on_wrsa.csv`: diagnostic tables.
- `features/stimulus_pool.csv`: stimulus manifest used for extraction and
  alignment.

## Recreating From Raw Evaluation Shards

The raw evaluation shards are not stored in Git when they are large parquet/HDF5
intermediates. To regenerate plots from a completed local evaluation directory,
use the package commands in `README.md` or the helper scripts in `scripts/`.
The top-level `figures/` and `reports/` CSVs are the shareable replotting
snapshot.
