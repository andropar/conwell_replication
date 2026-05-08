#!/bin/bash -l
#SBATCH --job-name=cache_brain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220000
#SBATCH --time=08:00:00
#SBATCH --output=/raven/ptmp/rothj/conwell_replication/.copy_logs/cache_brain_%j.log

set -eo pipefail

ENV=${ENV:-/raven/u/rothj/conda-envs/conwell_replication}
LAION_FMRI_ROOT=${LAION_FMRI_ROOT:-/raven/ptmp/rothj/laion_fmri}
CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE:-/raven/ptmp/rothj/conwell_replication/brain_cache}
SUBJECTS=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}
N_WORKERS=${N_WORKERS:-4}

mkdir -p "${CONWELL_BRAIN_CACHE}" /raven/ptmp/rothj/conwell_replication/.copy_logs

export LAION_FMRI_ROOT

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== LAION_FMRI_ROOT=${LAION_FMRI_ROOT}"
echo "=== CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE}"
echo "=== SUBJECTS=${SUBJECTS}"
echo "=== N_WORKERS=${N_WORKERS}"

# shellcheck disable=SC2086
"${ENV}/bin/conwell-cache-brain" \
    --cache-root "${CONWELL_BRAIN_CACHE}" \
    --subjects ${SUBJECTS} \
    --n-workers "${N_WORKERS}"

echo "=== finished: $(date --iso-8601=seconds)"
