#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
FEATURES=${FEATURES:-/ptmp/rothj/conwell_replication/features}
OUT=${OUT:-/ptmp/rothj/conwell_replication/eval_splithalf_otc}
MODEL_LIST=${MODEL_LIST:-/ptmp/rothj/conwell_replication/eval_model_manifest_replication.txt}
MODEL_CONTRASTS=${MODEL_CONTRASTS:-${REPO}/resources/model_contrasts.csv}
MANIFEST_AUDIT=${MANIFEST_AUDIT:-${MODEL_LIST%.txt}.audit.csv}
LOG_DIR=${LOG_DIR:-/ptmp/rothj/conwell_replication/.copy_logs}
EVAL_SCRIPT=${EVAL_SCRIPT:-${REPO}/scripts/eval_splithalf_array_viper.sbatch}
LAION_FMRI_ROOT=${LAION_FMRI_ROOT:-/ptmp/rothj/laion_fmri}
CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE:-/ptmp/rothj/conwell_replication/brain_cache_otc}
VOXEL_SET=${VOXEL_SET:-otc}
ROIS=${ROIS:-}
MIN_ROI_VOXELS=${MIN_ROI_VOXELS:-10}

SUBJECTS=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}
MODELS_PER_TASK=${MODELS_PER_TASK:-5}
CHUNK_SIZE=${CHUNK_SIZE:-30}
MAX_RUNNING_PER_ARRAY=${MAX_RUNNING_PER_ARRAY:-8}
MAX_ACTIVE_ARRAYS=${MAX_ACTIVE_ARRAYS:-2}
MAX_SUBMITTED_ELEMENTS=${MAX_SUBMITTED_ELEMENTS:-80}
TIME_LIMIT=${TIME_LIMIT:-02:00:00}
POLL_SECONDS=${POLL_SECONDS:-60}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SRPR_VOXELWISE=${SRPR_VOXELWISE:-all}

MODEL_LIST_STEM=$(basename "${MODEL_LIST%.*}")
STATE_DIR=${STATE_DIR:-${OUT}/.autosubmit_splithalf_${MODEL_LIST_STEM}}
STATE_FILE=${STATE_FILE:-${STATE_DIR}/next_offset.txt}
LEDGER=${LEDGER:-${STATE_DIR}/jobs.tsv}
LOCK_FILE=${LOCK_FILE:-${STATE_DIR}/lock}
FIRST_OFFSET=${FIRST_OFFSET:-0}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${OUT}" "${LOG_DIR}" "$(dirname "${MODEL_LIST}")" "${STATE_DIR}"
cd "${REPO}"

if [[ ! -s "${MODEL_LIST}" ]]; then
    "${ENV}/bin/python" "${REPO}/scripts/make_eval_model_manifest.py" \
        --features "${FEATURES}" \
        --model-contrasts "${MODEL_CONTRASTS}" \
        --out "${MODEL_LIST}" \
        --audit "${MANIFEST_AUDIT}"
fi

read -r -a SUBJECT_ARR <<< "${SUBJECTS}"
N_MODELS=$(grep -vc '^[[:space:]]*#\|^[[:space:]]*$' "${MODEL_LIST}")
N_MODEL_BATCHES=$(((N_MODELS + MODELS_PER_TASK - 1) / MODELS_PER_TASK))
TOTAL=$((${#SUBJECT_ARR[@]} * N_MODEL_BATCHES))

if [[ ! -s "${STATE_FILE}" ]]; then
    echo "${FIRST_OFFSET}" > "${STATE_FILE}"
fi
if [[ ! -s "${LEDGER}" ]]; then
    printf "timestamp\tjob_id\toffset\tchunk_size\n" > "${LEDGER}"
fi

count_active_eval_elements() {
    squeue -h -r -u "${USER}" -n eval_splithalf -o '%i' | wc -l
}

count_active_eval_arrays() {
    squeue -h -r -u "${USER}" -n eval_splithalf -o '%i' \
        | awk -F_ '{print $1}' | sort -u | wc -l
}

count_active_user_elements() {
    squeue -h -r -u "${USER}" -o '%i' | wc -l
}

submit_chunk() {
    local offset=$1
    local chunk=$2
    local last=$((chunk - 1))

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY RUN] submit offset=${offset} chunk=${chunk}"
        return 0
    fi

    local jid
    if ! jid=$(sbatch \
        --parsable \
        --array="0-${last}%${MAX_RUNNING_PER_ARRAY}" \
        --time="${TIME_LIMIT}" \
        --output="${LOG_DIR}/eval_splithalf_%A_%a.log" \
        --export=ALL,REPO="${REPO}",TASK_OFFSET="${offset}",N_MODELS="${N_MODELS}",MODEL_LIST="${MODEL_LIST}",MODEL_CONTRASTS="${MODEL_CONTRASTS}",MANIFEST_AUDIT="${MANIFEST_AUDIT}",FEATURES="${FEATURES}",OUT="${OUT}",ENV="${ENV}",LAION_FMRI_ROOT="${LAION_FMRI_ROOT}",CONWELL_BRAIN_CACHE="${CONWELL_BRAIN_CACHE}",SUBJECTS="${SUBJECTS}",MODELS_PER_TASK="${MODELS_PER_TASK}",SKIP_EXISTING="${SKIP_EXISTING}",SRPR_VOXELWISE="${SRPR_VOXELWISE}",VOXEL_SET="${VOXEL_SET}",ROIS="${ROIS}",MIN_ROI_VOXELS="${MIN_ROI_VOXELS}" \
        "${EVAL_SCRIPT}" 2>&1); then
        echo "submit failed for offset=${offset} chunk=${chunk}: ${jid}"
        return 1
    fi
    jid=${jid%%;*}
    printf "%s\t%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "${jid}" "${offset}" "${chunk}" >> "${LEDGER}"
    echo "submitted job=${jid} offset=${offset} chunk=${chunk}"
}

main_loop() {
    echo "=== auto split-half submitter started $(date --iso-8601=seconds)"
    echo "repo=${REPO}"
    echo "out=${OUT}"
    echo "brain_cache=${CONWELL_BRAIN_CACHE}"
    echo "voxel_set=${VOXEL_SET}"
    echo "rois=${ROIS:-<default>}"
    echo "models=${N_MODELS}; models_per_task=${MODELS_PER_TASK}; model_batches=${N_MODEL_BATCHES}"
    echo "subjects=${SUBJECTS}"
    echo "total_tasks=${TOTAL}; chunk_size=${CHUNK_SIZE}; arrays<=${MAX_ACTIVE_ARRAYS}; per_array=%${MAX_RUNNING_PER_ARRAY}; submitted_elements<=${MAX_SUBMITTED_ELEMENTS}"

    while true; do
        local next active_eval active_arrays active_user capacity chunk remaining
        local submitted_arrays submitted_elements
        next=$(<"${STATE_FILE}")
        active_eval=$(count_active_eval_elements | tr -d ' ')
        active_arrays=$(count_active_eval_arrays | tr -d ' ')
        active_user=$(count_active_user_elements | tr -d ' ')
        submitted_arrays=0
        submitted_elements=0

        echo "[$(date --iso-8601=seconds)] next=${next}/${TOTAL} active_eval_elements=${active_eval} active_eval_arrays=${active_arrays} active_user_elements=${active_user}"

        while (( next < TOTAL )); do
            capacity=$((MAX_SUBMITTED_ELEMENTS - active_user - submitted_elements))
            if (( active_arrays + submitted_arrays >= MAX_ACTIVE_ARRAYS || capacity <= 0 )); then
                break
            fi

            chunk=${CHUNK_SIZE}
            if (( chunk > capacity )); then
                chunk=${capacity}
            fi
            remaining=$((TOTAL - next))
            if (( chunk > remaining )); then
                chunk=${remaining}
            fi
            if (( chunk <= 0 )); then
                break
            fi

            if ! submit_chunk "${next}" "${chunk}"; then
                break
            fi
            submitted_arrays=$((submitted_arrays + 1))
            submitted_elements=$((submitted_elements + chunk))
            next=$((next + chunk))
            echo "${next}" > "${STATE_FILE}"
        done

        active_eval=$(count_active_eval_elements | tr -d ' ')
        next=$(<"${STATE_FILE}")
        if (( next >= TOTAL && active_eval == 0 )); then
            echo "=== all submitted tasks have left the queue $(date --iso-8601=seconds)"
            break
        fi
        sleep "${POLL_SECONDS}"
    done
}

(
    flock -n 9 || {
        echo "Another auto_submit_eval_splithalf.sh appears to be running: ${LOCK_FILE}" >&2
        exit 1
    }
    main_loop
) 9>"${LOCK_FILE}"
