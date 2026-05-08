#!/bin/bash
# Submit the full feature-extraction sweep on Raven as one slurm array.
#
# Usage:
#   bash scripts/submit_extract_full_raven.sh                 # core 212 models, throttle 16
#   bash scripts/submit_extract_full_raven.sh --dry-run       # just print plan
#   THROTTLE=8 bash scripts/submit_extract_full_raven.sh
#   MODELS_CSV=resources/conwell_model_list.csv bash ...      # full 335 (will fail on extras-needing models)
#
# Run on a Raven login node, NOT Viper.

set -eo pipefail

REPO=/raven/u/rothj/conwell_replication
OUT_DIR="${OUT_DIR:-/raven/ptmp/rothj/conwell_replication/features}"

# Default to the full 335-model Conwell registry. Models whose source needs
# unavailable extras (detectron2, vissl, slip weights, tensorflow, ...) will
# fail and continue per-model — the array is idempotent, so a second pass
# after installing extras only re-runs the missing ones.
MODELS_CSV="${MODELS_CSV:-${REPO}/resources/conwell_model_list.csv}"
POOL_CSV="${POOL_CSV:-${REPO}/features/stimulus_pool.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"
THROTTLE="${THROTTLE:-16}"     # max simultaneous A100 tasks; bump up if the queue is empty
# Raven imposes MaxSubmit=300 per user; submit in chunks no larger than this,
# chained by --dependency=afterany so total queue depth stays under the cap.
CHUNK_SIZE="${CHUNK_SIZE:-200}"

DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
    esac
done

N=$(/raven/u/rothj/conda-envs/conwell_replication/bin/python -c "import pandas as pd; print(len(pd.read_csv('${MODELS_CSV}')))")
LAST=$((N - 1))

mkdir -p "${OUT_DIR}"

echo "=============================================="
echo " Full feature extraction sweep"
echo "=============================================="
echo "  models csv:  ${MODELS_CSV}     (${N} models)"
echo "  pool csv:    ${POOL_CSV}"
echo "  out dir:     ${OUT_DIR}"
echo "  batch size:  ${BATCH_SIZE}"
echo "  throttle:    %${THROTTLE} concurrent per chunk"
echo "  chunk size:  ${CHUNK_SIZE} (MaxSubmit=300 cap, chained via afterany)"
echo "  per-task:    1xA100, 18 cpus, 125 GB mem, 1 h time limit"
echo

if $DRY_RUN; then
    echo "[DRY RUN] would submit chunks:"
    START=0
    while [ "${START}" -lt "${N}" ]; do
        END=$((START + CHUNK_SIZE - 1))
        if [ "${END}" -ge "${N}" ]; then END=$((N - 1)); fi
        echo "  ${START}-${END}%${THROTTLE}"
        START=$((END + 1))
    done
    exit 0
fi

PREV_ID=""
ALL_IDS=()
START=0
while [ "${START}" -lt "${N}" ]; do
    END=$((START + CHUNK_SIZE - 1))
    if [ "${END}" -ge "${N}" ]; then END=$((N - 1)); fi

    DEP_FLAG=()
    if [ -n "${PREV_ID}" ]; then
        DEP_FLAG=(--dependency="afterany:${PREV_ID}")
    fi

    CHUNK_ID=$(sbatch \
        --parsable \
        --array="${START}-${END}%${THROTTLE}" \
        --export=ALL,MODELS_CSV="${MODELS_CSV}",POOL_CSV="${POOL_CSV}",OUT_DIR="${OUT_DIR}",BATCH_SIZE="${BATCH_SIZE}" \
        "${DEP_FLAG[@]}" \
        "${REPO}/scripts/_extract_array.sbatch")

    if [ -n "${PREV_ID}" ]; then
        echo "Submitted chunk ${START}-${END}: job ${CHUNK_ID}  (dep afterany ${PREV_ID})"
    else
        echo "Submitted chunk ${START}-${END}: job ${CHUNK_ID}"
    fi

    ALL_IDS+=("${CHUNK_ID}")
    PREV_ID="${CHUNK_ID}"
    START=$((END + 1))
done

echo
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  ls ${OUT_DIR}/*.h5 | wc -l                                       # done count"
echo "  tail -f ${OUT_DIR}/slurm_<JID>_<task>.log                        # one task"
echo "  cat ${OUT_DIR}/extraction_failures.csv                           # what failed"
echo
echo "Re-running this script after failures is safe — existing .h5s are skipped."
