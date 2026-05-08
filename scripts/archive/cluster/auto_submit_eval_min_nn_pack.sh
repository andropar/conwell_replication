#!/usr/bin/env bash
# Rolling submitter for the min-nn PACK eval. Keeps one array of CHUNK_SIZE
# packs submitted at a time; submits the next chunk when the current chunk
# clears (or the user's submit cap drops below MAX_SUBMITTED_ELEMENTS).
#
# Designed to be run from a slurm job (not the login node).
set -euo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
FEATURES=${FEATURES:-/ptmp/rothj/conwell_replication/features}
EXTRA_FEATURES_DIR=${EXTRA_FEATURES_DIR:-/ptmp/rothj/conwell_replication/features_ood}
OUT=${OUT:-/ptmp/rothj/conwell_replication/eval_min_nn_otc}
MODEL_LIST=${MODEL_LIST:-/ptmp/rothj/conwell_replication/eval_model_manifest_replication.txt}
MODEL_CONTRASTS=${MODEL_CONTRASTS:-${REPO}/resources/model_contrasts.csv}
LOG_DIR=${LOG_DIR:-/ptmp/rothj/conwell_replication/.copy_logs}
LAION_FMRI_ROOT=${LAION_FMRI_ROOT:-/ptmp/rothj/laion_fmri}
CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE:-/ptmp/rothj/conwell_replication/brain_cache_otc}
PACK_SCRIPT=${PACK_SCRIPT:-${REPO}/scripts/eval_min_nn_pack_viper.sbatch}
TASK_SCRIPT=${TASK_SCRIPT:-${REPO}/scripts/eval_min_nn_array_viper.sbatch}
VOXEL_SET=${VOXEL_SET:-otc}
SRPR_VOXELWISE=${SRPR_VOXELWISE:-all}
SKIP_EXISTING=${SKIP_EXISTING:-1}

SUBJECTS=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}
POOL_MODES=${POOL_MODES:-"shared subject"}
SPLITS=${SPLITS:-"random_0 random_1 random_2 random_3 random_4 cluster_k5_0 cluster_k5_1 cluster_k5_2 cluster_k5_3 cluster_k5_4 tau ood"}
PACK_SIZE=${PACK_SIZE:-8}

CHUNK_SIZE=${CHUNK_SIZE:-200}
MAX_RUNNING=${MAX_RUNNING:-16}
POLL_SECONDS=${POLL_SECONDS:-90}

read -r -a SUBJECT_ARR <<< "${SUBJECTS}"
read -r -a POOL_MODE_ARR <<< "${POOL_MODES}"
read -r -a SPLIT_ARR <<< "${SPLITS}"

N_MODELS=$(grep -vc '^[[:space:]]*#\|^[[:space:]]*$' "${MODEL_LIST}")
TOTAL_INNER=$((${#SUBJECT_ARR[@]} * ${#POOL_MODE_ARR[@]} * ${#SPLIT_ARR[@]} * N_MODELS))
TOTAL_PACKS=$(( (TOTAL_INNER + PACK_SIZE - 1) / PACK_SIZE ))

STATE_DIR=${STATE_DIR:-${OUT}/.autosubmit_pack}
STATE_FILE=${STATE_FILE:-${STATE_DIR}/next_pack.txt}
LEDGER=${LEDGER:-${STATE_DIR}/jobs.tsv}
mkdir -p "${OUT}" "${STATE_DIR}" "${LOG_DIR}"
[[ -s "${STATE_FILE}" ]] || echo 0 > "${STATE_FILE}"
[[ -s "${LEDGER}" ]] || printf "timestamp\tjob_id\tpack_offset\tn_packs\n" > "${LEDGER}"

echo "=== auto submit pack started $(date --iso-8601=seconds)"
echo "TOTAL_INNER=${TOTAL_INNER} TOTAL_PACKS=${TOTAL_PACKS} CHUNK=${CHUNK_SIZE} %${MAX_RUNNING}"

count_active() {
    squeue -h -r -u "${USER}" -n eval_min_nn_pack -o '%i' 2>/dev/null | wc -l
}

while true; do
    next=$(<"${STATE_FILE}")
    if (( next >= TOTAL_PACKS )); then
        if (( $(count_active) == 0 )); then
            echo "[$(date +%H:%M:%S)] all submitted and queue drained"
            break
        fi
    fi
    active=$(count_active)
    if (( active > 0 )); then
        echo "[$(date +%H:%M:%S)] ${active} pack tasks still in queue (next_pack=${next}/${TOTAL_PACKS}); sleeping"
        sleep "${POLL_SECONDS}"
        continue
    fi

    chunk=${CHUNK_SIZE}
    remaining=$((TOTAL_PACKS - next))
    if (( chunk > remaining )); then chunk=${remaining}; fi
    if (( chunk <= 0 )); then sleep "${POLL_SECONDS}"; continue; fi

    last=$((chunk - 1))
    pack_offset=${next}
    inner_offset=$((pack_offset * PACK_SIZE))
    if jid=$(sbatch --parsable \
        --array="0-${last}%${MAX_RUNNING}" \
        --export=ALL,REPO="${REPO}",TASK_OFFSET="${inner_offset}",OUT="${OUT}",FEATURES="${FEATURES}",EXTRA_FEATURES_DIR="${EXTRA_FEATURES_DIR}",MODEL_LIST="${MODEL_LIST}",MODEL_CONTRASTS="${MODEL_CONTRASTS}",ENV="${ENV}",SKIP_EXISTING="${SKIP_EXISTING}",SRPR_VOXELWISE="${SRPR_VOXELWISE}",VOXEL_SET="${VOXEL_SET}",LAION_FMRI_ROOT="${LAION_FMRI_ROOT}",CONWELL_BRAIN_CACHE="${CONWELL_BRAIN_CACHE}",TASK_SCRIPT="${TASK_SCRIPT}",SUBJECTS="${SUBJECTS}",POOL_MODES="${POOL_MODES}",SPLITS="${SPLITS}" \
        "${PACK_SCRIPT}" 2>&1); then
        printf "%s\t%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "${jid}" "${pack_offset}" "${chunk}" >> "${LEDGER}"
        next=$((next + chunk))
        echo "${next}" > "${STATE_FILE}"
        echo "[$(date +%H:%M:%S)] submitted ${jid} packs ${pack_offset}..$((pack_offset + chunk - 1)) (next=${next}/${TOTAL_PACKS})"
    else
        echo "[$(date +%H:%M:%S)] submit failed: ${jid}"
    fi
    sleep "${POLL_SECONDS}"
done
echo "=== auto submit pack finished $(date --iso-8601=seconds)"
