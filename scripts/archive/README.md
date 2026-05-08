# Archived Scripts

These scripts record how the completed runs were launched on MPCDF Viper/Raven
and how intermediate caches were copied or diagnosed. They are intentionally
not part of the clean top-level workflow.

Paths inside these files may point to the original machine layout
(`/ptmp/rothj`, `/u/rothj`, `/raven/...`). Treat them as provenance templates:
adapt paths and environment names before rerunning them.

- `cluster/`: SLURM launchers, auto-submit wrappers, rsync jobs, and Viper/Raven
  environment setup.
- `diagnostics/`: one-off scripts used to compare old/new brain caches and
  probe ROI/OOD effects.
- `smoke_tests/`: early smoke-test and screen-launch helpers.
