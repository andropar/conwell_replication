#!/bin/bash -l
#SBATCH --job-name=eval_splithalf
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=01:00:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/eval_splithalf_%A_%a.log

set -eo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
FEATURES=${FEATURES:-/ptmp/rothj/conwell_replication/features}
OUT=${OUT:-/ptmp/rothj/conwell_replication/eval_splithalf_otc_ncsnr}
NCSNR_THRESHOLD=${NCSNR_THRESHOLD:-0.2}
INCLUDE_OOD=${INCLUDE_OOD:-0}
EXTRA_FEATURES_DIR=${EXTRA_FEATURES_DIR:-}
MODEL_LIST=${MODEL_LIST:-/ptmp/rothj/conwell_replication/eval_model_manifest_replication.txt}
MODEL_CONTRASTS=${MODEL_CONTRASTS:-${REPO}/resources/model_contrasts.csv}
MANIFEST_AUDIT=${MANIFEST_AUDIT:-${MODEL_LIST%.txt}.audit.csv}
TASK_OFFSET=${TASK_OFFSET:-0}
MODELS_PER_TASK=${MODELS_PER_TASK:-1}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SRPR_VOXELWISE=${SRPR_VOXELWISE:-all}
VOXEL_SET=${VOXEL_SET:-otc}
ROIS=${ROIS:-}
MIN_ROI_VOXELS=${MIN_ROI_VOXELS:-10}

SUBJECTS_STR=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}
read -r -a SUBJECT_ARR <<< "${SUBJECTS_STR}"

export LAION_FMRI_ROOT=${LAION_FMRI_ROOT:-/ptmp/rothj/laion_fmri}
export CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE:-/ptmp/rothj/conwell_replication/brain_cache_otc}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "${OUT}" /ptmp/rothj/conwell_replication/.copy_logs "$(dirname "${MODEL_LIST}")"
if [[ ! -s "${MODEL_LIST}" ]]; then
    "${ENV}/bin/python" "${REPO}/scripts/make_eval_model_manifest.py" \
        --features "${FEATURES}" \
        --model-contrasts "${MODEL_CONTRASTS}" \
        --out "${MODEL_LIST}" \
        --audit "${MANIFEST_AUDIT}"
fi

N_MODELS=${N_MODELS:-$(grep -vc '^[[:space:]]*#\|^[[:space:]]*$' "${MODEL_LIST}")}
if (( MODELS_PER_TASK <= 0 )); then
    echo "MODELS_PER_TASK must be positive, got ${MODELS_PER_TASK}" >&2
    exit 2
fi
N_MODEL_BATCHES=$(((N_MODELS + MODELS_PER_TASK - 1) / MODELS_PER_TASK))
N_SUBJECTS=${#SUBJECT_ARR[@]}
GLOBAL_ID=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))
TOTAL=$((N_SUBJECTS * N_MODEL_BATCHES))
if (( GLOBAL_ID >= TOTAL )); then
    echo "GLOBAL_ID=${GLOBAL_ID} >= TOTAL=${TOTAL}; exiting"
    exit 0
fi

BATCH_IDX=$((GLOBAL_ID % N_MODEL_BATCHES))
SUBJECT_IDX=$((GLOBAL_ID / N_MODEL_BATCHES))
SUBJECT=${SUBJECT_ARR[$SUBJECT_IDX]}
MODEL_START=$((BATCH_IDX * MODELS_PER_TASK))
MODEL_STOP=$((MODEL_START + MODELS_PER_TASK))
if (( MODEL_STOP > N_MODELS )); then
    MODEL_STOP=${N_MODELS}
fi

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== job/task: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}/${SLURM_ARRAY_TASK_ID}"
echo "=== global_id: ${GLOBAL_ID}/${TOTAL}"
echo "=== subject=${SUBJECT} odd/even model_batch=${BATCH_IDX}/${N_MODEL_BATCHES}"
echo "=== models_per_task=${MODELS_PER_TASK} model_idx_range=${MODEL_START}..$((MODEL_STOP - 1))/${N_MODELS}"
echo "=== voxel_set=${VOXEL_SET} brain_cache=${CONWELL_BRAIN_CACHE}"

ROI_ARGS=()
if [[ -n "${ROIS}" ]]; then
    read -r -a ROI_ARR <<< "${ROIS}"
    ROI_ARGS=(--rois "${ROI_ARR[@]}")
    echo "=== rois=${ROIS}"
fi

OUT_PATH="${OUT}/results_${SUBJECT}_batch${BATCH_IDX}.parquet"
VW_PATH="${OUT}/srpr_voxelwise/results_${SUBJECT}_batch${BATCH_IDX}.scores.npy"
if [[ "${SKIP_EXISTING}" == "1" && -s "${OUT_PATH}" ]]; then
    if [[ "${SRPR_VOXELWISE}" == "none" || -s "${VW_PATH}" ]]; then
        echo "=== skipping existing output: ${OUT_PATH}"
        exit 0
    fi
fi

EXTRA_ARGS=()
if [[ "${INCLUDE_OOD}" == "1" ]]; then
    EXTRA_ARGS+=(--include-ood)
fi
if [[ -n "${EXTRA_FEATURES_DIR}" ]]; then
    EXTRA_ARGS+=(--extra-features-dir "${EXTRA_FEATURES_DIR}")
fi

/usr/bin/time -v "${ENV}/bin/conwell-eval-splithalf" \
    --features "${FEATURES}" \
    --out "${OUT}" \
    --subjects "${SUBJECT}" \
    --voxel-set "${VOXEL_SET}" \
    "${ROI_ARGS[@]}" \
    --min-roi-voxels "${MIN_ROI_VOXELS}" \
    --ncsnr-threshold "${NCSNR_THRESHOLD}" \
    --model-list "${MODEL_LIST}" \
    --models-per-batch "${MODELS_PER_TASK}" \
    --model-batch "${BATCH_IDX}" \
    --srpr-voxelwise "${SRPR_VOXELWISE}" \
    "${EXTRA_ARGS[@]}"

echo "=== finished: $(date --iso-8601=seconds)"
