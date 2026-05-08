#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export REPO=${REPO:-/raven/u/rothj/conwell_replication}
export ENV=${ENV:-/raven/u/rothj/conda-envs/conwell_replication}
export OUT=${OUT:-/raven/ptmp/rothj/conwell_replication/eval_splithalf_streamed_otc}
export MODELS_CSV=${MODELS_CSV:-${REPO}/resources/conwell_model_list_replication.csv}
export MODEL_LIST=${MODEL_LIST:-}
export POOL_CSV=${POOL_CSV:-${REPO}/features/stimulus_pool.csv}
export LOG_DIR=${LOG_DIR:-/raven/ptmp/rothj/conwell_replication/.copy_logs}
export EVAL_SCRIPT=${EVAL_SCRIPT:-${SCRIPT_DIR}/eval_splithalf_streamed_array_raven.sbatch}
export LAION_FMRI_ROOT=${LAION_FMRI_ROOT:-/raven/ptmp/rothj/laion_fmri}
export CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE:-/raven/ptmp/rothj/conwell_replication/brain_cache_otc}

export SUBJECTS=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}
export MODELS_PER_TASK=${MODELS_PER_TASK:-1}
export CHUNK_SIZE=${CHUNK_SIZE:-50}
export MAX_RUNNING=${MAX_RUNNING:-4}
export TIME_LIMIT=${TIME_LIMIT:-24:00:00}
export SKIP_EXISTING=${SKIP_EXISTING:-1}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if (( $# > 0 )); then
    echo "Unknown arguments: $*" >&2
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

mkdir -p "${OUT}" "${LOG_DIR}"
cd "${REPO}"

if (( MODELS_PER_TASK <= 0 )); then
    echo "MODELS_PER_TASK must be positive, got ${MODELS_PER_TASK}" >&2
    exit 2
fi
if (( CHUNK_SIZE <= 0 )); then
    echo "CHUNK_SIZE must be positive, got ${CHUNK_SIZE}" >&2
    exit 2
fi
if (( MAX_RUNNING <= 0 )); then
    echo "MAX_RUNNING must be positive, got ${MAX_RUNNING}" >&2
    exit 2
fi

if [[ -n "${MODEL_LIST}" ]]; then
    if [[ ! -s "${MODEL_LIST}" ]]; then
        echo "MODEL_LIST does not exist or is empty: ${MODEL_LIST}" >&2
        exit 2
    fi
    N_MODELS=$(grep -vc '^[[:space:]]*#\|^[[:space:]]*$' "${MODEL_LIST}")
    MODEL_SOURCE="${MODEL_LIST}"
else
    if [[ ! -s "${MODELS_CSV}" ]]; then
        echo "MODELS_CSV does not exist or is empty: ${MODELS_CSV}" >&2
        exit 2
    fi
    N_MODELS=$("${ENV}/bin/python" - "${MODELS_CSV}" <<'PY'
import sys
import pandas as pd

print(len(pd.read_csv(sys.argv[1])))
PY
)
    MODEL_SOURCE="${MODELS_CSV}"
fi

N_BATCHES=$(((N_MODELS + MODELS_PER_TASK - 1) / MODELS_PER_TASK))

echo "=== model source: ${MODEL_SOURCE}"
echo "=== n_models=${N_MODELS} models_per_task=${MODELS_PER_TASK} n_batches=${N_BATCHES}"
echo "=== chunks: size=${CHUNK_SIZE} max_running=${MAX_RUNNING} time_limit=${TIME_LIMIT}"
echo "=== out=${OUT}"

for (( start = 0; start < N_BATCHES; start += CHUNK_SIZE )); do
    stop=$((start + CHUNK_SIZE))
    if (( stop > N_BATCHES )); then
        stop=${N_BATCHES}
    fi
    last_local=$((stop - start - 1))
    array_spec="0-${last_local}%${MAX_RUNNING}"
    cmd=(
        sbatch
        --time "${TIME_LIMIT}"
        --array "${array_spec}"
        --export "ALL,TASK_OFFSET=${start}"
        "${EVAL_SCRIPT}"
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN: TASK_OFFSET=%s ' "${start}"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        echo "=== submitting batches ${start}..$((stop - 1)) with array ${array_spec}"
        "${cmd[@]}"
    fi
done
