#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
FEATURES=${FEATURES:-/ptmp/rothj/conwell_replication/features}
MODEL_LIST=${MODEL_LIST:-/ptmp/rothj/conwell_replication/eval_smoketest_first5_models.txt}
FULL_MODEL_LIST=${FULL_MODEL_LIST:-/ptmp/rothj/conwell_replication/eval_model_manifest_replication.txt}
MODEL_CONTRASTS=${MODEL_CONTRASTS:-${REPO}/resources/model_contrasts.csv}
MANIFEST_AUDIT=${MANIFEST_AUDIT:-${FULL_MODEL_LIST%.txt}.audit.csv}
OUT=${OUT:-/ptmp/rothj/conwell_replication/eval_smoketest_array}
MAX_RUNNING=${MAX_RUNNING:-8}
SUBJECTS_STR=${SUBJECTS:-"sub-01"}
SPLITS_STR=${SPLITS:-"random_0 random_1 random_2 random_3 random_4 cluster_k5_0 cluster_k5_1 cluster_k5_2 cluster_k5_3 cluster_k5_4 tau"}

mkdir -p "$(dirname "${MODEL_LIST}")"
if [[ ! -s "${FULL_MODEL_LIST}" ]]; then
    "${ENV}/bin/python" "${REPO}/scripts/make_eval_model_manifest.py" \
        --features "${FEATURES}" \
        --model-contrasts "${MODEL_CONTRASTS}" \
        --out "${FULL_MODEL_LIST}" \
        --audit "${MANIFEST_AUDIT}"
fi
grep -v '^[[:space:]]*#\|^[[:space:]]*$' "${FULL_MODEL_LIST}" | head -5 > "${MODEL_LIST}"

N_MODELS=$(grep -vc '^[[:space:]]*#\|^[[:space:]]*$' "${MODEL_LIST}")
read -r -a SPLIT_ARR <<< "${SPLITS_STR}"
TOTAL=$((N_MODELS * ${#SPLIT_ARR[@]}))
ARRAY_LAST=$((TOTAL - 1))

echo "Submitting smoke array: ${TOTAL} tasks (${N_MODELS} models x ${#SPLIT_ARR[@]} splits)"
sbatch \
    --array=0-${ARRAY_LAST}%${MAX_RUNNING} \
    --export=ALL,REPO=${REPO},TASK_OFFSET=0,N_MODELS=${N_MODELS},MODEL_LIST=${MODEL_LIST},MODEL_CONTRASTS=${MODEL_CONTRASTS},MANIFEST_AUDIT=${MANIFEST_AUDIT},FEATURES=${FEATURES},OUT=${OUT},ENV=${ENV},SUBJECTS="${SUBJECTS_STR}",SPLITS="${SPLITS_STR}" \
    scripts/eval_min_nn_array_viper.sbatch
