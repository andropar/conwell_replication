# conwell_replication

Replication analyses for [Conwell et al. (2024)](https://doi.org/10.1038/s41467-024-53972-1)
on the [LAION-fMRI](https://laion-fmri.hebartlab.com/laion_fmri_package/index.html)
dataset.

The code follows the DeepNSD feature-extraction protocol used by Conwell et al.:

- DeepNSD model loading through `model_opts.model_options.get_model_options`.
- Model-specific preprocessing via `get_recommended_transforms`.
- Hook-based feature extraction with DeepNSD-style `ModuleType-N` layer names.
- Sparse random projection feature reduction.

The required DeepNSD source files are vendored under
`src/conwell_replication/_vendor/deepnsd/`. Large model checkpoints and raw
data are not committed.

## Repository Layout

```text
src/conwell_replication/
  data/              LAION-fMRI adapter, ROI masks, stimulus-pool construction
  extract/           DeepNSD-protocol feature extraction
  eval/              split-half and min-nn RSA evaluators
  analysis/          best-layer selection and statistical tests
  figures/           plotting code
  _vendor/deepnsd/   vendored DeepNSD model-loading utilities

configs/             example extraction/evaluation configs
features/            lightweight stimulus-pool CSV manifests
figures/             result tables, diagnostics, and rendered analysis figures
reports/             report-level summary tables and figures
resources/           model manifests, comparison metadata, and weight notes
scripts/             reusable helper scripts
```

Archived run launchers and legacy manifests are kept under `scripts/archive/`
and `resources/archive/`.

For completed output tables and replotting entry points, see
[RESULTS.md](RESULTS.md). For methodological details, see [METHODS.md](METHODS.md).

## Installation

```bash
git clone <repo-url>
cd conwell_replication

pip install -e .
```

Optional dependencies are grouped by model source:

```bash
pip install -e '.[clip,openclip,vissl]'
pip install -e '.[full]'
```

`.[full]` installs heavier dependencies such as detectron2, TensorFlow, and
VQGAN-related packages. Install only the extras needed for the model sources
you plan to extract.

## Data Requirements

Set `LAION_FMRI_ROOT` to a local LAION-fMRI dataset directory:

```bash
export LAION_FMRI_ROOT=/path/to/laion_fmri_data
```

Brain data can be downloaded through the `laion_fmri` package:

```python
from laion_fmri import dataset_initialize, set_aws_credentials, download

dataset_initialize("/path/to/laion_fmri_data")
set_aws_credentials(...)

for subject in ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07"):
    download(subject=subject, n_jobs=8)
```

The feature extractor also needs local stimulus images. The build-pool step
expects an image tree with the LAION-fMRI shared and subject-specific stimulus
directories:

```text
image_sets/
  deepvision_shared/
  deepvision_unique_sub-01/
  deepvision_unique_sub-03/
  deepvision_unique_sub-05/
  deepvision_unique_sub-06/
  deepvision_unique_sub-07/
```

Pass that directory with `--stimuli-root` when constructing the stimulus-pool
manifest. The committed `features/stimulus_pool.csv` is a lightweight manifest;
the image files themselves are not included.

Some DeepNSD models require checkpoint files that are too large for Git. See
[resources/weights/README.md](resources/weights/README.md) for the expected
SLIP checkpoint filenames and destination path.

## Model Manifests

- `resources/conwell_model_list.csv`: full trained DeepNSD registry used by
  this project after excluding random-weight models.
- `resources/conwell_model_list_replication.csv`: controlled-comparison subset
  used for the current Fig. 2-4 replication outputs.
- `resources/conwell_model_list_core.csv`: dependency-practical subset for core
  model sources.
- `resources/model_contrasts.csv`: metadata for controlled model comparisons.

## Pipeline

Build the deduplicated stimulus-pool manifest:

```bash
conwell-build-pool build-pool \
    --stimuli-root /path/to/image_sets \
    --output features/stimulus_pool.csv
```

Extract model features:

```bash
conwell-extract \
    --models resources/conwell_model_list_replication.csv \
    --pool features/stimulus_pool.csv \
    --out features/
```

Compute noise ceilings:

```bash
conwell-noise-ceiling \
    --mode shared \
    --out results/splithalf/noise_ceilings.csv

conwell-noise-ceiling \
    --mode min_nn \
    --pool shared \
    --out results/min_nn_shared/noise_ceilings.csv
```

Run evaluations:

```bash
conwell-eval-splithalf \
    --features features/ \
    --out results/splithalf/

conwell-eval-min-nn \
    --features features/ \
    --pool shared \
    --out results/min_nn_shared/
```

Prepare summary tables, statistics, and figures:

```bash
conwell-prepare-scores \
    --results results/splithalf/results_all.parquet \
    --noise-ceiling results/splithalf/noise_ceilings.csv \
    --out results/splithalf/

conwell-stats --results-dir results/splithalf/
conwell-figures --results-dir results/splithalf/
```

`scripts/run_pipeline.sh` provides a compact end-to-end launcher for the same
basic workflow.

## Results

The repository includes lightweight CSV summaries and rendered figures for the
completed analyses:

- `reports/`: report-level summaries and figures.
- `figures/splithalf_results/`: split-half result tables and plots.
- `figures/splithalf_results_ood/`: split-half outputs with OOD images included.
- `figures/min_nn_results/`: generalization-split result tables and plots.

See [RESULTS.md](RESULTS.md) for the specific files to use when replotting.
