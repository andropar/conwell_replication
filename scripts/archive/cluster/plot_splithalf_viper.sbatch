#!/bin/bash -l
#SBATCH --job-name=plot_splithalf
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --time=00:30:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/plot_splithalf_%j.log

set -eo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
RESULTS_DIR=${RESULTS_DIR:-/ptmp/rothj/conwell_replication/eval_splithalf_otc_ncsnr}
NOISE_CEILING=${NOISE_CEILING:-/ptmp/rothj/conwell_replication/noise_ceiling/noise_ceiling_ncsnr.csv}
REGION=${REGION:-otc}
NC_METRIC=${NC_METRIC:-rsa}
OUT_DIR=${OUT_DIR:-${REPO}/figures/splithalf_results}

mkdir -p /ptmp/rothj/conwell_replication/.copy_logs

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== RESULTS_DIR=${RESULTS_DIR}"
echo "=== NOISE_CEILING=${NOISE_CEILING}"
echo "=== REGION=${REGION} NC_METRIC=${NC_METRIC}"
echo "=== OUT_DIR=${OUT_DIR}"

"${ENV}/bin/python" "${REPO}/scripts/plot_splithalf_results.py" \
    --results-dir "${RESULTS_DIR}" \
    --noise-ceiling "${NOISE_CEILING}" \
    --region "${REGION}" \
    --nc-metric "${NC_METRIC}" \
    --out-dir "${OUT_DIR}"

echo "=== finished: $(date --iso-8601=seconds)"
