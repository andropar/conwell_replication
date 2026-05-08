# Scripts

Top-level scripts are reusable entry points that should work from a fresh clone
after installing the package and setting the documented environment variables.

- `run_pipeline.sh`: compact end-to-end launcher for the current controlled
  replication subset.
- `download_laion_fmri.py`: convenience downloader for the five subjects used
  here.
- `compute_ncsnr.py`: merge per-voxel NCSNR estimates into a local brain cache.
- `compute_noise_ceiling.py`: generate the split-half noise-ceiling CSV used by
  the analysis plots.
- `make_eval_model_manifest.py`: derive the controlled-comparison evaluation
  model list from available feature files.
- `plot_splithalf_results.py`: regenerate split-half controlled-comparison
  figures from evaluation shards.
- `plot_otc_roi.py`: generate the OTC ROI sanity-check figure.
- `build_fig2_4_report.py`: rebuild `reports/laion_fmri_conwell_fig2_4_report.md`
  and its report-level summary figures.

Cluster launchers, development-test helpers, transfer scripts, and one-off
diagnostics are kept in `scripts/archive/`.
