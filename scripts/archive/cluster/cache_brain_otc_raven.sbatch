#!/bin/bash -l
#SBATCH --job-name=cache_brain_otc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=119000
#SBATCH --time=08:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/raven/ptmp/rothj/conwell_replication/.copy_logs/cache_brain_otc_%A_%a.log

set -euo pipefail

ENV=${ENV:-/raven/u/rothj/conda-envs/conwell_replication}
LAION_FMRI_ROOT=${LAION_FMRI_ROOT:-/raven/ptmp/rothj/laion_fmri}
CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE:-/raven/ptmp/rothj/conwell_replication/brain_cache_otc}
ROI_ROOT=${ROI_ROOT:-/raven/ptmp/rothj/laion_fmri/derivatives/rois}
SUBJECTS=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}
N_WORKERS=${N_WORKERS:-4}
R2_THRESHOLD=${R2_THRESHOLD:-0.15}
FORCE=${FORCE:-0}

read -r -a SUBJECT_ARR <<< "${SUBJECTS}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if (( SLURM_ARRAY_TASK_ID >= ${#SUBJECT_ARR[@]} )); then
        echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} outside SUBJECTS length ${#SUBJECT_ARR[@]}; exiting"
        exit 0
    fi
    RUN_SUBJECTS=("${SUBJECT_ARR[$SLURM_ARRAY_TASK_ID]}")
else
    RUN_SUBJECTS=("${SUBJECT_ARR[@]}")
fi

mkdir -p "${CONWELL_BRAIN_CACHE}" /raven/ptmp/rothj/conwell_replication/.copy_logs

export LAION_FMRI_ROOT

FORCE_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
    FORCE_ARGS=(--force)
fi

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== LAION_FMRI_ROOT=${LAION_FMRI_ROOT}"
echo "=== CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE}"
echo "=== ROI_ROOT=${ROI_ROOT}"
echo "=== SUBJECTS=${RUN_SUBJECTS[*]}"
echo "=== N_WORKERS=${N_WORKERS}"
echo "=== R2_THRESHOLD=${R2_THRESHOLD}"
echo "=== FORCE=${FORCE}"

"${ENV}/bin/python" -m conwell_replication.data.brain_cache \
    --cache-root "${CONWELL_BRAIN_CACHE}" \
    --subjects "${RUN_SUBJECTS[@]}" \
    --voxel-set otc \
    --roi-root "${ROI_ROOT}" \
    --r2-threshold "${R2_THRESHOLD}" \
    --n-workers "${N_WORKERS}" \
    "${FORCE_ARGS[@]}"

echo "=== finished: $(date --iso-8601=seconds)"
