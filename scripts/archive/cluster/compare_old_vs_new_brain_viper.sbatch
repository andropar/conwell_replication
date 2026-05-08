#!/bin/bash -l
#SBATCH --job-name=compare_old_new_brain
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --time=00:30:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/compare_old_new_brain_%j.log

set -eo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
OLD_ROOT=${OLD_ROOT:-/ptmp/rothj/deepvision_fmri_old}
NEW_CACHE_ROOT=${NEW_CACHE_ROOT:-/ptmp/rothj/conwell_replication/brain_cache_otc}
OUT=${OUT:-${REPO}/figures/old_vs_new_brain_comparison.csv}

mkdir -p /ptmp/rothj/conwell_replication/.copy_logs "$(dirname "${OUT}")"

echo "=== started: $(date --iso-8601=seconds) on $(hostname)"
echo "=== OLD_ROOT=${OLD_ROOT}"
echo "=== NEW_CACHE_ROOT=${NEW_CACHE_ROOT}"
echo "=== OUT=${OUT}"

"${ENV}/bin/python" "${REPO}/scripts/compare_old_vs_new_brain.py" \
    --old-root "${OLD_ROOT}" \
    --new-cache-root "${NEW_CACHE_ROOT}" \
    --out "${OUT}"

echo "=== done: $(date --iso-8601=seconds)"
