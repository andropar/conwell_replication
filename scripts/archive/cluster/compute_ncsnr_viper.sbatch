#!/bin/bash -l
#SBATCH --job-name=compute_ncsnr
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --time=00:30:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/compute_ncsnr_%j.log

set -eo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
CACHE_ROOT=${CACHE_ROOT:-/ptmp/rothj/conwell_replication/brain_cache_otc}
THRESHOLD=${THRESHOLD:-0.2}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p /ptmp/rothj/conwell_replication/.copy_logs

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== CACHE_ROOT=${CACHE_ROOT} THRESHOLD=${THRESHOLD}"

"${ENV}/bin/python" "${REPO}/scripts/compute_ncsnr.py" \
    --cache-root "${CACHE_ROOT}" \
    --threshold "${THRESHOLD}"

echo "=== finished: $(date --iso-8601=seconds)"
