#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
FEATURES=${FEATURES:-/ptmp/rothj/conwell_replication/features}
OUT=${OUT:-/ptmp/rothj/conwell_replication/eval_min_nn_array_deepnsd}
MODEL_LIST=${MODEL_LIST:-/ptmp/rothj/conwell_replication/eval_model_manifest_replication.txt}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
MODEL_CONTRASTS=${MODEL_CONTRASTS:-${REPO}/resources/model_contrasts.csv}
MANIFEST_AUDIT=${MANIFEST_AUDIT:-${MODEL_LIST%.txt}.audit.csv}
LAION_FMRI_ROOT=${LAION_FMRI_ROOT:-/ptmp/rothj/laion_fmri}
CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE:-/ptmp/rothj/conwell_replication/brain_cache}
EXTRA_FEATURES_DIR=${EXTRA_FEATURES_DIR:-}
LOG_DIR=${LOG_DIR:-/ptmp/rothj/conwell_replication/.copy_logs}
EVAL_SCRIPT=${EVAL_SCRIPT:-scripts/eval_min_nn_array_viper.sbatch}
PARTITION=${PARTITION:-general}
CHUNK_SIZE=${CHUNK_SIZE:-300}
MAX_RUNNING=${MAX_RUNNING:-8}
MODELS_PER_TASK=${MODELS_PER_TASK:-1}
TIME_LIMIT=${TIME_LIMIT:-}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SRPR_VOXELWISE=${SRPR_VOXELWISE:-none}
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi
TASK_OFFSET=${1:-0}

SUBJECTS_STR=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}
POOL_MODES_STR=${POOL_MODES:-"shared subject"}
SPLITS_STR=${SPLITS:-"random_0 random_1 random_2 random_3 random_4 cluster_k5_0 cluster_k5_1 cluster_k5_2 cluster_k5_3 cluster_k5_4 tau ood"}
read -r -a SUBJECT_ARR <<< "${SUBJECTS_STR}"
read -r -a POOL_MODE_ARR <<< "${POOL_MODES_STR}"
read -r -a SPLIT_ARR <<< "${SPLITS_STR}"

mkdir -p "$(dirname "${MODEL_LIST}")"
if [[ ! -s "${MODEL_LIST}" ]]; then
    "${ENV}/bin/python" "${REPO}/scripts/make_eval_model_manifest.py" \
        --features "${FEATURES}" \
        --model-contrasts "${MODEL_CONTRASTS}" \
        --out "${MODEL_LIST}" \
        --audit "${MANIFEST_AUDIT}"
fi

N_MODELS=$(grep -vc '^[[:space:]]*#\|^[[:space:]]*$' "${MODEL_LIST}")
if (( MODELS_PER_TASK <= 0 )); then
    echo "MODELS_PER_TASK must be positive, got ${MODELS_PER_TASK}" >&2
    exit 2
fi
N_MODEL_BATCHES=$(((N_MODELS + MODELS_PER_TASK - 1) / MODELS_PER_TASK))
TOTAL=$((${#SUBJECT_ARR[@]} * ${#POOL_MODE_ARR[@]} * ${#SPLIT_ARR[@]} * N_MODEL_BATCHES))
REMAINING=$((TOTAL - TASK_OFFSET))
if (( REMAINING <= 0 )); then
    echo "Nothing to submit: offset ${TASK_OFFSET} >= total ${TOTAL}"
    exit 0
fi

THIS_CHUNK=${CHUNK_SIZE}
if (( REMAINING < THIS_CHUNK )); then
    THIS_CHUNK=${REMAINING}
fi
ARRAY_LAST=$((THIS_CHUNK - 1))

echo "Submitting min_nn tasks ${TASK_OFFSET}..$((TASK_OFFSET + THIS_CHUNK - 1)) of ${TOTAL}"
echo "Subjects: ${SUBJECTS_STR}"
echo "Pool modes: ${POOL_MODES_STR}"
echo "Splits: ${SPLITS_STR}"
echo "Models: ${N_MODELS}; models per task: ${MODELS_PER_TASK}; model batches: ${N_MODEL_BATCHES}"
echo "Features: ${FEATURES}"
echo "Out: ${OUT}"
echo "Brain cache: ${CONWELL_BRAIN_CACHE}"
SBATCH_ARGS=(
    --partition="${PARTITION}" \
    --array=0-${ARRAY_LAST}%${MAX_RUNNING} \
    --output="${LOG_DIR}/eval_min_nn_%A_%a.log" \
    --export=ALL,REPO=${REPO},TASK_OFFSET=${TASK_OFFSET},N_MODELS=${N_MODELS},MODEL_LIST=${MODEL_LIST},MODEL_CONTRASTS=${MODEL_CONTRASTS},MANIFEST_AUDIT=${MANIFEST_AUDIT},FEATURES=${FEATURES},OUT=${OUT},ENV=${ENV},LAION_FMRI_ROOT=${LAION_FMRI_ROOT},CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE},EXTRA_FEATURES_DIR=${EXTRA_FEATURES_DIR},LOG_DIR=${LOG_DIR},SUBJECTS="${SUBJECTS_STR}",POOL_MODES="${POOL_MODES_STR}",SPLITS="${SPLITS_STR}",MODELS_PER_TASK=${MODELS_PER_TASK},SKIP_EXISTING=${SKIP_EXISTING},SRPR_VOXELWISE=${SRPR_VOXELWISE} \
)
if [[ -n "${TIME_LIMIT}" ]]; then
    SBATCH_ARGS+=(--time="${TIME_LIMIT}")
fi
if ${DRY_RUN}; then
    echo "[DRY RUN] sbatch ${SBATCH_ARGS[*]} ${EVAL_SCRIPT}"
    echo "Next offset: $((TASK_OFFSET + THIS_CHUNK))"
    exit 0
fi
sbatch "${SBATCH_ARGS[@]}" \
    "${EVAL_SCRIPT}"
echo "Next offset: $((TASK_OFFSET + THIS_CHUNK))"
