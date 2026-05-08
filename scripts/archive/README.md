# Archived Scripts

These scripts are not part of the main public workflow. They are retained as
reference material for batch submission, data-transfer helpers, development
tests, and diagnostics.

Some files contain site-specific scheduler settings, environment names, or
filesystem paths. Review and adapt them before running on a different system.

- `cluster/`: SLURM launchers, auto-submit wrappers, transfer jobs, and
  environment setup helpers.
- `diagnostics/`: one-off scripts used for cache comparisons and ROI/OOD
  checks.
- `smoke_tests/`: minimal extraction/download test helpers.
