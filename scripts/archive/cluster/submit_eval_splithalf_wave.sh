#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
FEATURES=${FEATURES:-/ptmp/rothj/conwell_replication/features}
OUT=${OUT:-/ptmp/rothj/conwell_replication/eval_splithalf_array_deepnsd}
MODEL_LIST=${MODEL_LIST:-/ptmp/rothj/conwell_replication/eval_model_manifest_replication.txt}
MODEL_CONTRASTS=${MODEL_CONTRASTS:-${REPO}/resources/model_contrasts.csv}
MANIFEST_AUDIT=${MANIFEST_AUDIT:-${MODEL_LIST%.txt}.audit.csv}
CHUNK_SIZE=${CHUNK_SIZE:-300}
MAX_RUNNING=${MAX_RUNNING:-8}
MODELS_PER_TASK=${MODELS_PER_TASK:-1}
TIME_LIMIT=${TIME_LIMIT:-}
SRPR_VOXELWISE=${SRPR_VOXELWISE:-none}
SKIP_EXISTING=${SKIP_EXISTING:-1}
TASK_OFFSET=${1:-0}

SUBJECTS_STR=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}
read -r -a SUBJECT_ARR <<< "${SUBJECTS_STR}"

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
TOTAL=$((${#SUBJECT_ARR[@]} * N_MODEL_BATCHES))
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

echo "Submitting split-half tasks ${TASK_OFFSET}..$((TASK_OFFSET + THIS_CHUNK - 1)) of ${TOTAL}"
echo "Subjects: ${SUBJECTS_STR}"
echo "Models: ${N_MODELS}; models per task: ${MODELS_PER_TASK}; model batches per subject: ${N_MODEL_BATCHES}"
SBATCH_ARGS=(
    --array=0-${ARRAY_LAST}%${MAX_RUNNING} \
    --export=ALL,REPO=${REPO},TASK_OFFSET=${TASK_OFFSET},N_MODELS=${N_MODELS},MODEL_LIST=${MODEL_LIST},MODEL_CONTRASTS=${MODEL_CONTRASTS},MANIFEST_AUDIT=${MANIFEST_AUDIT},FEATURES=${FEATURES},OUT=${OUT},ENV=${ENV},SUBJECTS="${SUBJECTS_STR}",MODELS_PER_TASK=${MODELS_PER_TASK},SRPR_VOXELWISE=${SRPR_VOXELWISE},SKIP_EXISTING=${SKIP_EXISTING} \
)
if [[ -n "${TIME_LIMIT}" ]]; then
    SBATCH_ARGS+=(--time="${TIME_LIMIT}")
fi
sbatch "${SBATCH_ARGS[@]}" \
    scripts/eval_splithalf_array_viper.sbatch
echo "Next offset: $((TASK_OFFSET + THIS_CHUNK))"
