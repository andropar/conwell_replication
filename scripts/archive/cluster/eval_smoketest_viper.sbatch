#!/bin/bash -l
#SBATCH --job-name=eval_smoketest
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=02:00:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/eval_smoketest_%j.log

set -eo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
OUT=${OUT:-/ptmp/rothj/conwell_replication/eval_smoketest_otc}
FEATURES=${FEATURES:-/ptmp/rothj/conwell_replication/features}
MODEL_LIST=${MODEL_LIST:-/ptmp/rothj/conwell_replication/eval_model_manifest_replication.txt}
MODEL_CONTRASTS=${MODEL_CONTRASTS:-${REPO}/resources/model_contrasts.csv}
MANIFEST_AUDIT=${MANIFEST_AUDIT:-${MODEL_LIST%.txt}.audit.csv}

export LAION_FMRI_ROOT=/ptmp/rothj/laion_fmri
export CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE:-/ptmp/rothj/conwell_replication/brain_cache_otc}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CONWELL_EVAL_VERBOSE_LAYERS=1

mkdir -p "${OUT}" /ptmp/rothj/conwell_replication/.copy_logs
if [[ ! -s "${MODEL_LIST}" ]]; then
    "${ENV}/bin/python" "${REPO}/scripts/make_eval_model_manifest.py" \
        --features "${FEATURES}" \
        --model-contrasts "${MODEL_CONTRASTS}" \
        --out "${MODEL_LIST}" \
        --audit "${MANIFEST_AUDIT}"
fi

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== job: ${SLURM_JOB_ID}"
echo "=== cpus: ${SLURM_CPUS_PER_TASK}"
echo "=== mem mb: ${SLURM_MEM_PER_NODE:-unknown}"
echo "=== python: ${ENV}/bin/python"

"${ENV}/bin/python" - <<'PY'
import numpy as np
import sklearn
from conwell_replication.eval._common import _fit_ridge_single_alpha
print("numpy", np.__version__)
print("sklearn", sklearn.__version__)
print("ridge helper", _fit_ridge_single_alpha.__module__)
PY

/usr/bin/time -v "${ENV}/bin/conwell-eval-min-nn" \
    --features "${FEATURES}" \
    --out "${OUT}" \
    --subjects sub-01 \
    --voxel-set otc \
    --splits random_0 \
    --model-list "${MODEL_LIST}" \
    --models-per-batch 5 \
    --model-batch 0 \
    --alphas 100000 \
    --srpr-voxelwise all

echo "=== finished: $(date --iso-8601=seconds)"
