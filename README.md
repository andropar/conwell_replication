# conwell_replication

Replication of [Conwell et al. (2024)](https://doi.org/10.1038/s41467-024-53972-1)
on the [LAION-fMRI](https://laion-fmri.hebartlab.com/laion_fmri_package/index.html)
dataset, with two evaluation modes:

1. **Split-half** (Conwell-style): even/odd split of the shared-stimulus subset.
   Used as a sanity gate against the existing `rsa_20260223_154344` results.
2. **min_nn generalization splits**: per-subject train/test partitions over the
   full per-subject stimulus pool. 13 splits per subject (3 `tau_*`, 5 `random_*`,
   5 `cluster_k5_*`).

Models: the curated 152-model subset (timm / torchvision / openclip / vissl /
slip / dino / vicreg / seer / cornet / robustness / dinov3). Loaded via the
vendored `universal_extractor`, which delegates to `deepjuice` for most
checkpoints and uses local custom loaders for the rest.

## Layout

```
src/conwell_replication/
  data/              LAIONBenchmark adapter, split loading, stimuli pool
  extract/           Feature extraction (union pool, per-model HDF5 with SRP)
  eval/              RSA evaluators (split-half, min_nn) + noise ceiling
  analysis/          Best-layer selection, statistical tests
  figures/           Replication of fig1/fig2/fig3 + min_nn variants
  _vendor/           DeepNSD ridge_gcv_mod / mapping_methods / feature_reduction
                     and the universal_extractor

resources/
  curated_model_list.csv   152 models
  model_metadata.csv       comparison-group / typology metadata
  splits/p0{1..5}/         min_nn split JSONs (variant lists per split)

configs/             hydra YAMLs
scripts/             helper shell scripts
```

## Install (new server)

```bash
git clone git@github.com:andropar/conwell_replication.git
cd conwell_replication

# Editable install
pip install -e .

# Required external (private/GH-hosted) dependencies — install separately:
pip install git+https://github.com/ColinConwell/DeepJuice.git    # or your fork
pip install <laion_fmri install URL>                              # see docs
```

## Data layout (assumed on new server)

You download the LAION-fMRI dataset via the `laion_fmri` package:

```python
from laion_fmri import dataset_initialize, set_aws_credentials, download
dataset_initialize("/path/to/laion_fmri_data")
set_aws_credentials(...)
for sub in ("sub-01","sub-02","sub-03","sub-04","sub-05"):
    download(subject=sub, n_jobs=8)
```

The repo expects an environment variable pointing at that root:

```bash
export LAION_FMRI_ROOT=/path/to/laion_fmri_data
```

Subject-to-participant mapping (matches DeepVision / pre-existing analyses):

| subject  | participant |
|----------|-------------|
| sub-01   | p01         |
| sub-03   | p02         |
| sub-05   | p03         |
| sub-06   | p04         |
| sub-07   | p05         |

> **TODO**: confirm whether the *new* preprocessing keeps the same
> `sub-XX → pXX` mapping. If LAION-fMRI uses sequential `sub-01..sub-05`
> instead, edit `data/benchmark.py:SUBJECT_TO_PARTICIPANT`.

## Pipeline

```bash
# 1. Cache the union of per-subject stimulus pools (deduplicated)
python -m conwell_replication.data.stimuli build-pool \
    --output features/stimulus_pool.csv

# 2. Extract features for all 152 models on the union pool
python -m conwell_replication.extract.extract_features \
    --models resources/curated_model_list.csv \
    --pool   features/stimulus_pool.csv \
    --out    features/

# 3. Compute noise ceilings (per subject, on test sets per split)
python -m conwell_replication.eval.noise_ceiling \
    --out results/noise_ceilings.csv

# 4a. (Sanity) Split-half evaluation on shared 1492 stimuli
python -m conwell_replication.eval.rsa_splithalf \
    --features features/ \
    --out      results/splithalf/

# 4b. min_nn evaluation on 13 splits × 5 subjects
python -m conwell_replication.eval.rsa_min_nn \
    --features features/ \
    --splits   resources/splits/ \
    --out      results/min_nn/

# 5. Prepare best-layer scores + statistical tests
python -m conwell_replication.analysis.prepare_scores \
    --results results/min_nn/results_all.parquet \
    --out     results/min_nn/
python -m conwell_replication.analysis.statistical_tests \
    --scores  results/min_nn/best_layer_scores.csv \
    --out     results/min_nn/

# 6. Figures
python -m conwell_replication.figures.plot_figures \
    --results-dir results/min_nn/ \
    --out         results/min_nn/figures/
```

## Sanity replication

Before trusting min_nn numbers, run the split-half evaluator and confirm the
controlled-comparison effect sizes match
`experiments/ccn2025_continuation/rsa_large_scale_benchmark/results/rsa_20260223_154344/`
within tolerance.

## See also

- Source replication report:
  `experiments/ccn2025_continuation/rsa_large_scale_benchmark/results/rsa_20260223_154344/REPLICATION_REPORT.md`
- min_nn split definitions:
  `experiments/generalization_split/min_nn/`
