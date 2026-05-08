#!/bin/bash -l
#SBATCH --job-name=noise_ceiling
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --time=01:00:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/noise_ceiling_%j.log

set -eo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
CACHE_ROOT=${CACHE_ROOT:-/ptmp/rothj/conwell_replication/brain_cache_otc}
OUT=${OUT:-/ptmp/rothj/conwell_replication/noise_ceiling/noise_ceiling_ncsnr.csv}
N_SPLITS=${N_SPLITS:-50}

export LAION_FMRI_ROOT=${LAION_FMRI_ROOT:-/ptmp/rothj/laion_fmri}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "$(dirname "${OUT}")" /ptmp/rothj/conwell_replication/.copy_logs

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== CACHE_ROOT=${CACHE_ROOT}"
echo "=== OUT=${OUT}"
echo "=== N_SPLITS=${N_SPLITS}"

"${ENV}/bin/python" "${REPO}/scripts/compute_noise_ceiling.py" \
    --cache-root "${CACHE_ROOT}" \
    --out "${OUT}" \
    --n-splits "${N_SPLITS}"

echo "=== finished: $(date --iso-8601=seconds)"
