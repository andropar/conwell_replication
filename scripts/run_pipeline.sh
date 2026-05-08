#!/usr/bin/env bash
# End-to-end pipeline launcher.
#
# Assumes:
#   * laion_fmri data is downloaded and reachable via $LAION_FMRI_ROOT
#   * the conwell_replication package itself is pip-installed
#   * model weights that cannot be downloaded automatically have been copied in
#
# Run as `bash scripts/run_pipeline.sh [--skip-extract] [--mode {min_nn|splithalf|both}]`.

set -euo pipefail

cd "$(dirname "$0")/.."

MODE="both"
SKIP_EXTRACT=0
GPUS="0"
MODELS_CSV="${MODELS_CSV:-resources/conwell_model_list_replication.csv}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-extract) SKIP_EXTRACT=1; shift ;;
        --mode)         MODE="$2"; shift 2 ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p features results/{splithalf,min_nn} logs

# 1. Build the union stimulus pool ----------------------------------------
if [[ ! -f features/stimulus_pool.csv ]]; then
    python -m conwell_replication.data.stimuli build-pool \
        --output features/stimulus_pool.csv
fi

# 2. Feature extraction for the current controlled-replication model subset ---
if [[ "$SKIP_EXTRACT" -eq 0 ]]; then
    echo "Extracting features on GPU $GPUS..."
    CUDA_VISIBLE_DEVICES="$GPUS" python -m conwell_replication.extract.extract_features \
        --models "$MODELS_CSV" \
        --pool   features/stimulus_pool.csv \
        --out    features/ 2>&1 | tee logs/extract.log
fi

# 3. Noise ceilings ------------------------------------------------------
python -m conwell_replication.eval.noise_ceiling \
    --mode shared --out results/splithalf/noise_ceilings.csv 2>&1 | tee logs/nc_shared.log
python -m conwell_replication.eval.noise_ceiling \
    --mode min_nn --out results/min_nn/noise_ceilings.csv 2>&1 | tee logs/nc_min_nn.log

# 4. Evaluations ---------------------------------------------------------
if [[ "$MODE" == "splithalf" || "$MODE" == "both" ]]; then
    python -m conwell_replication.eval.rsa_splithalf \
        --features features/ \
        --out      results/splithalf/ 2>&1 | tee logs/splithalf.log
fi

if [[ "$MODE" == "min_nn" || "$MODE" == "both" ]]; then
    python -m conwell_replication.eval.rsa_min_nn \
        --features features/ \
        --out      results/min_nn/ 2>&1 | tee logs/min_nn.log
fi

# 5. Analysis + figures --------------------------------------------------
for D in results/splithalf results/min_nn; do
    if [[ -f "$D/results_all.parquet" ]]; then
        python -m conwell_replication.analysis.prepare_scores \
            --results        "$D/results_all.parquet" \
            --noise-ceiling  "$D/noise_ceilings.csv" \
            --out            "$D/"
        python -m conwell_replication.analysis.statistical_tests \
            --results-dir    "$D/"
        python -m conwell_replication.figures.plot_figures \
            --results-dir    "$D/"
    fi
done

echo "Pipeline finished. Results in results/"
