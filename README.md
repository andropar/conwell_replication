# conwell_replication

Replication of [Conwell et al. (2024)](https://doi.org/10.1038/s41467-024-53972-1)
on the [LAION-fMRI](https://laion-fmri.hebartlab.com/laion_fmri_package/index.html)
dataset, using **Conwell's DeepNSD extraction protocol** end-to-end:

- DeepNSD's call-string registry to load each model
  (`model_opts.model_options.get_model_options`).
- DeepNSD's `get_recommended_transforms(option_key, input_type="PIL")` for
  preprocessing per model.
- Hook-based feature extraction
  (`model_opts.feature_extraction.get_all_feature_maps`) — same `ModuleType-N`
  layer naming as the paper.
- SRP reduction with `eps=0.1`.

The DeepNSD source is **vendored** at
[src/conwell_replication/_vendor/deepnsd/](src/conwell_replication/_vendor/deepnsd/);
no separate DeepNSD checkout is required on the new server. The 31 GB of
SLIP `.pt` weights are gitignored — see [resources/weights/README.md](resources/weights/README.md)
for the manual transfer step.

Two evaluation modes:

1. **Split-half** (Conwell-style): even/odd split of the shared-stimulus
   subset. Used as a sanity gate against
   `experiments/ccn2025_continuation/rsa_large_scale_benchmark/results/rsa_20260223_154344/`.
2. **min_nn generalization splits**: per-subject train/test partitions over
   the full per-subject stimulus pool. 13 splits per subject (3 `tau_*`,
   5 `random_*`, 5 `cluster_k5_*`).

Models: DeepNSD's **335 trained models** (the 489-model registry filtered to
`train_type != "random"`). Defined in
[resources/conwell_model_list.csv](resources/conwell_model_list.csv).

## Layout

```
src/conwell_replication/
  data/              LAIONBenchmark adapter, split loading, stimuli pool
  extract/           Feature extraction (DeepNSD protocol over union pool)
  eval/              RSA evaluators (split-half, min_nn) + noise ceiling
  analysis/          Best-layer selection, statistical tests
  figures/           Replication of fig1/fig2/fig3 + min_nn variants
  _vendor/
    deepnsd/         Conwell's pressures/model_opts tree (vendored)

resources/
  conwell_model_list.csv   335-model trained subset of DeepNSD's registry
  model_metadata.csv       Per-model comparison-group metadata (from the
                           prior 413-model replication)
  splits/p0{1..5}/         min_nn split JSONs (variant lists per split)
  weights/README.md        Notes on transferring SLIP / other heavy weights

configs/             hydra YAMLs
scripts/             helper shell scripts
```

## Install (new server)

```bash
git clone git@github.com:andropar/conwell_replication.git
cd conwell_replication

# Editable install. Pulls in core deps (torch/torchvision/timm/transformers
# plus the analysis stack). Extras for specific model_sources are gated:
pip install -e .                  # core only
pip install -e .[clip,openclip,vissl]   # the most-used extras
pip install -e .[full]            # everything (heavy: detectron2, tensorflow, ...)

# Plus the data downloader:
pip install <laion_fmri install URL>   # see https://laion-fmri.hebartlab.com/

# Plus the SLIP weights (~31 GB) — see resources/weights/README.md
```

## Data layout (assumed on new server)

Download the LAION-fMRI dataset:

```python
from laion_fmri import dataset_initialize, set_aws_credentials, download
dataset_initialize("/path/to/laion_fmri_data")
set_aws_credentials(...)
for sub in ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07"):
    download(subject=sub, n_jobs=8)
```

Set the env var so the LAIONBenchmark adapter can find it:

```bash
export LAION_FMRI_ROOT=/path/to/laion_fmri_data
```

Subject-to-participant mapping (carried over from DeepVision; verify on the
new server):

| subject  | participant |
|----------|-------------|
| sub-01   | p01         |
| sub-03   | p02         |
| sub-05   | p03         |
| sub-06   | p04         |
| sub-07   | p05         |

## Pipeline

```bash
# 1. Cache the union of per-subject stimulus pools (deduplicated, ~24k images)
conwell-build-pool build-pool --output features/stimulus_pool.csv

# 2. Extract features for the 335-model registry on the union pool
#    (one .h5 per model, image_id-indexed, SRP-reduced)
conwell-extract \
    --models resources/conwell_model_list.csv \
    --pool   features/stimulus_pool.csv \
    --out    features/

#    Subset by model_source if desired:
# conwell-extract --sources timm torchvision openclip ...

# 3. Compute noise ceilings
conwell-noise-ceiling --mode shared --out results/splithalf/noise_ceilings.csv
conwell-noise-ceiling --mode min_nn --out results/min_nn/noise_ceilings.csv

# 4a. (Sanity) Split-half evaluation on shared-stimulus subset
conwell-eval-splithalf --features features/ --out results/splithalf/

# 4b. min_nn evaluation on 13 splits × 5 subjects
conwell-eval-min-nn --features features/ --out results/min_nn/

# 5. Best-layer selection + statistical tests
for D in results/splithalf results/min_nn; do
    conwell-prepare-scores \
        --results       "$D/results_all.parquet" \
        --noise-ceiling "$D/noise_ceilings.csv" \
        --out           "$D/"
    conwell-stats    --results-dir "$D/"
    conwell-figures  --results-dir "$D/"
done
```

`scripts/run_pipeline.sh` does the same steps as a single shell entry-point.

## Sanity replication

Before trusting min_nn numbers, run the split-half evaluator and confirm the
controlled-comparison effect sizes match
`experiments/ccn2025_continuation/rsa_large_scale_benchmark/results/rsa_20260223_154344/`
within tolerance. **Note**: that earlier run used a hybrid extractor
(deepjuice/FX for the curated 152, DeepNSD-hooks for the rest). The new
pipeline is uniformly DeepNSD-hooks, so small numerical differences are
expected on the curated-152 subset; the qualitative effect-size hierarchy
should match.

## See also

- Source replication report:
  `experiments/ccn2025_continuation/rsa_large_scale_benchmark/results/rsa_20260223_154344/REPLICATION_REPORT.md`
- min_nn split definitions:
  `experiments/generalization_split/min_nn/`
