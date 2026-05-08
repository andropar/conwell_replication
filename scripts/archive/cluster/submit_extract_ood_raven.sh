#!/bin/bash
# Submit OOD-pool feature extraction on Raven for the 118 replication models.
#
# Run on a Raven login node:
#   bash scripts/submit_extract_ood_raven.sh
#   bash scripts/submit_extract_ood_raven.sh --dry-run     # print plan only
#   THROTTLE=8 bash scripts/submit_extract_ood_raven.sh
#
# Inputs that must already exist on Raven:
#   - resources/conwell_model_list_replication.csv   (118 rows; built from full)
#   - /ptmp/rothj/conwell_replication/features_ood/stimulus_pool_ood.csv (372 rows)
#   - OOD stimulus PNGs at /ptmp/rothj/.../stimuli/image_sets/deepvision_shared/
#
# Idempotent — existing .h5s are skipped by _extract_array.sbatch.

set -eo pipefail

REPO=/raven/u/rothj/conwell_replication
ENV=/raven/u/rothj/conda-envs/conwell_replication
OUT_DIR="${OUT_DIR:-/raven/ptmp/rothj/conwell_replication/features_ood}"
MODELS_CSV="${MODELS_CSV:-${REPO}/resources/conwell_model_list_replication.csv}"
POOL_CSV="${POOL_CSV:-/raven/ptmp/rothj/conwell_replication/features_ood/stimulus_pool_ood.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"
THROTTLE="${THROTTLE:-16}"

DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
    esac
done

if [ ! -s "${MODELS_CSV}" ]; then
    echo "ERROR: MODELS_CSV not found at ${MODELS_CSV}" >&2
    exit 2
fi
if [ ! -s "${POOL_CSV}" ]; then
    echo "ERROR: POOL_CSV not found at ${POOL_CSV}" >&2
    exit 2
fi

N=$("${ENV}/bin/python" -c "import pandas as pd; print(len(pd.read_csv('${MODELS_CSV}')))")
LAST=$((N - 1))

mkdir -p "${OUT_DIR}"

echo "=============================================="
echo " OOD-pool feature extraction (replication 118)"
echo "=============================================="
echo "  models csv:  ${MODELS_CSV}     (${N} models)"
echo "  pool csv:    ${POOL_CSV}       ($(($(wc -l < "${POOL_CSV}") - 1)) images)"
echo "  out dir:     ${OUT_DIR}"
echo "  batch size:  ${BATCH_SIZE}"
echo "  throttle:    %${THROTTLE} concurrent A100 tasks"
echo "  per-task:    1xA100, 18 cpus, 120 GB mem, 1 h time limit"
echo

if $DRY_RUN; then
    echo "[DRY RUN] would submit one array 0-${LAST}%${THROTTLE}"
    exit 0
fi

JOB_ID=$(sbatch \
    --parsable \
    --array="0-${LAST}%${THROTTLE}" \
    --export=ALL,MODELS_CSV="${MODELS_CSV}",POOL_CSV="${POOL_CSV}",OUT_DIR="${OUT_DIR}",BATCH_SIZE="${BATCH_SIZE}" \
    "${REPO}/scripts/_extract_array.sbatch")

echo "Submitted array 0-${LAST}: job ${JOB_ID}"
echo
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  ls ${OUT_DIR}/*.h5 | wc -l                      # done count"
echo "  tail -f ${OUT_DIR}/slurm_<JID>_<task>.log       # one task"
