#!/bin/bash -l
#SBATCH --job-name=eval_min_nn
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=02:00:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/eval_min_nn_%A_%a.log

set -eo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
FEATURES=${FEATURES:-/ptmp/rothj/conwell_replication/features}
OUT=${OUT:-/ptmp/rothj/conwell_replication/eval_min_nn_otc}
MODEL_LIST=${MODEL_LIST:-/ptmp/rothj/conwell_replication/eval_model_manifest_replication.txt}
MODEL_CONTRASTS=${MODEL_CONTRASTS:-${REPO}/resources/model_contrasts.csv}
MANIFEST_AUDIT=${MANIFEST_AUDIT:-${MODEL_LIST%.txt}.audit.csv}
LOG_DIR=${LOG_DIR:-/ptmp/rothj/conwell_replication/.copy_logs}
TASK_OFFSET=${TASK_OFFSET:-0}
MODELS_PER_TASK=${MODELS_PER_TASK:-1}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SRPR_VOXELWISE=${SRPR_VOXELWISE:-all}
EXTRA_FEATURES_DIR=${EXTRA_FEATURES_DIR:-}
VOXEL_SET=${VOXEL_SET:-otc}
ROIS=${ROIS:-}
MIN_ROI_VOXELS=${MIN_ROI_VOXELS:-10}

SUBJECTS_STR=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}
POOL_MODES_STR=${POOL_MODES:-"shared subject"}
SPLITS_STR=${SPLITS:-"random_0 random_1 random_2 random_3 random_4 cluster_k5_0 cluster_k5_1 cluster_k5_2 cluster_k5_3 cluster_k5_4 tau ood"}
read -r -a SUBJECT_ARR <<< "${SUBJECTS_STR}"
read -r -a POOL_MODE_ARR <<< "${POOL_MODES_STR}"
read -r -a SPLIT_ARR <<< "${SPLITS_STR}"

export LAION_FMRI_ROOT=${LAION_FMRI_ROOT:-/ptmp/rothj/laion_fmri}
export CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE:-/ptmp/rothj/conwell_replication/brain_cache_otc}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "${OUT}" "${LOG_DIR}" "$(dirname "${MODEL_LIST}")"
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
N_POOL_MODES=${#POOL_MODE_ARR[@]}
N_SPLITS=${#SPLIT_ARR[@]}
GLOBAL_ID=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))
TOTAL=$((N_SUBJECTS * N_POOL_MODES * N_SPLITS * N_MODEL_BATCHES))
if (( GLOBAL_ID >= TOTAL )); then
    echo "GLOBAL_ID=${GLOBAL_ID} >= TOTAL=${TOTAL}; exiting"
    exit 0
fi

BATCH_IDX=$((GLOBAL_ID % N_MODEL_BATCHES))
TMP=$((GLOBAL_ID / N_MODEL_BATCHES))
SPLIT_IDX=$((TMP % N_SPLITS))
TMP=$((TMP / N_SPLITS))
POOL_MODE_IDX=$((TMP % N_POOL_MODES))
SUBJECT_IDX=$((TMP / N_POOL_MODES))
SUBJECT=${SUBJECT_ARR[$SUBJECT_IDX]}
POOL_MODE=${POOL_MODE_ARR[$POOL_MODE_IDX]}
SPLIT=${SPLIT_ARR[$SPLIT_IDX]}
case "${POOL_MODE}" in
    shared)
        POOL=shared
        ;;
    subject|full|subject_full)
        POOL=${SUBJECT}
        ;;
    *)
        POOL=${POOL_MODE}
        ;;
esac
MODEL_START=$((BATCH_IDX * MODELS_PER_TASK))
MODEL_STOP=$((MODEL_START + MODELS_PER_TASK))
if (( MODEL_STOP > N_MODELS )); then
    MODEL_STOP=${N_MODELS}
fi
OUT_PATH="${OUT}/results_${SUBJECT}_pool-${POOL}_${SPLIT}_v0_batch${BATCH_IDX}.parquet"
VW_PATH="${OUT}/srpr_voxelwise/results_${SUBJECT}_pool-${POOL}_${SPLIT}_v0_batch${BATCH_IDX}.scores.npy"

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== job/task: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}/${SLURM_ARRAY_TASK_ID}"
echo "=== global_id: ${GLOBAL_ID}/${TOTAL}"
echo "=== subject=${SUBJECT} pool_mode=${POOL_MODE} pool=${POOL} split=${SPLIT}"
echo "=== model_batch=${BATCH_IDX}/${N_MODEL_BATCHES} models_per_task=${MODELS_PER_TASK}"
echo "=== model_idx_range=${MODEL_START}..$((MODEL_STOP - 1))/${N_MODELS}"
echo "=== voxel_set=${VOXEL_SET} brain_cache=${CONWELL_BRAIN_CACHE}"

EXTRA_FEATURE_ARGS=()
if [[ -n "${EXTRA_FEATURES_DIR}" ]]; then
    EXTRA_FEATURE_ARGS=(--extra-features-dir "${EXTRA_FEATURES_DIR}")
    echo "=== extra_features_dir=${EXTRA_FEATURES_DIR}"
fi
ROI_ARGS=()
if [[ -n "${ROIS}" ]]; then
    read -r -a ROI_ARR <<< "${ROIS}"
    ROI_ARGS=(--rois "${ROI_ARR[@]}")
    echo "=== rois=${ROIS}"
fi

if [[ "${SKIP_EXISTING}" == "1" && -s "${OUT_PATH}" ]]; then
    if [[ "${SRPR_VOXELWISE}" == "none" || -s "${VW_PATH}" ]]; then
        echo "=== skipping existing output: ${OUT_PATH}"
        exit 0
    fi
fi

/usr/bin/time -v "${ENV}/bin/conwell-eval-min-nn" \
    --features "${FEATURES}" \
    --out "${OUT}" \
    --subjects "${SUBJECT}" \
    --pool "${POOL}" \
    --voxel-set "${VOXEL_SET}" \
    "${ROI_ARGS[@]}" \
    --min-roi-voxels "${MIN_ROI_VOXELS}" \
    --splits "${SPLIT}" \
    --model-list "${MODEL_LIST}" \
    --models-per-batch "${MODELS_PER_TASK}" \
    --model-batch "${BATCH_IDX}" \
    --skip-existing \
    --srpr-voxelwise "${SRPR_VOXELWISE}" \
    "${EXTRA_FEATURE_ARGS[@]}"

echo "=== finished: $(date --iso-8601=seconds)"
