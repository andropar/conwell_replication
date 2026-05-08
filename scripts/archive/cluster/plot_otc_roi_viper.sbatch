#!/bin/bash -l
#SBATCH --job-name=plot_otc_roi
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --time=00:30:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/plot_otc_roi_%j.log

set -eo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
DATA_ROOT=${DATA_ROOT:-/ptmp/rothj/laion_fmri}
CACHE_ROOT=${CACHE_ROOT:-/ptmp/rothj/conwell_replication/brain_cache_otc}
OUT=${OUT:-${REPO}/figures/otc_roi_sanity.pdf}

mkdir -p /ptmp/rothj/conwell_replication/.copy_logs "$(dirname "${OUT}")"

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== DATA_ROOT=${DATA_ROOT} CACHE_ROOT=${CACHE_ROOT} OUT=${OUT}"

"${ENV}/bin/python" "${REPO}/scripts/plot_otc_roi.py" \
    --data-root "${DATA_ROOT}" \
    --cache-root "${CACHE_ROOT}" \
    --out "${OUT}"

echo "=== finished: $(date --iso-8601=seconds)"
