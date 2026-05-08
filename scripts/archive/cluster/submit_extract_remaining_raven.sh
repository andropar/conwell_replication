#!/bin/bash
# Submit any not-yet-submitted, not-yet-done extraction tasks, respecting
# the user's MaxSubmit=300 cap on Raven. Polls in a loop until all 335 are
# either queued or finished.
#
# Idempotent and safe to run alongside an existing chunk1 array — it just
# fills capacity as chunk1 tasks complete.
#
# Run on a Raven login node, in screen if you want to leave it running:
#   screen -dmS extract_filler bash scripts/submit_extract_remaining_raven.sh

set -eo pipefail

REPO=/raven/u/rothj/conwell_replication
ENV=/raven/u/rothj/conda-envs/conwell_replication
OUT_DIR="${OUT_DIR:-/raven/ptmp/rothj/conwell_replication/features}"
MODELS_CSV="${MODELS_CSV:-${REPO}/resources/conwell_model_list.csv}"
POOL_CSV="${POOL_CSV:-${REPO}/features/stimulus_pool.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"
THROTTLE="${THROTTLE:-16}"
MAX_SUBMIT_CAP=300                      # per-user MaxSubmit on Raven
SAFETY_MARGIN=5                         # leave a few slots for other small jobs
POLL_SECONDS="${POLL_SECONDS:-60}"

N=$("${ENV}/bin/python" -c "import pandas as pd; print(len(pd.read_csv('${MODELS_CSV}')))")
echo "Total models in registry: ${N}"

list_remaining_indices() {
    "${ENV}/bin/python" - <<PY
import pandas as pd, os
df = pd.read_csv('${MODELS_CSV}')
out_dir = '${OUT_DIR}'
remaining = []
for i, row in df.iterrows():
    h5 = os.path.join(out_dir, row['option_key'].replace('/', '-') + '.h5')
    if not os.path.exists(h5):
        remaining.append(i)
print(' '.join(str(i) for i in remaining))
PY
}

count_in_queue() {
    # Number of jobs (array elements counted individually) for this user
    squeue -u "$USER" -h -t pending,running -r 2>/dev/null | wc -l
}

# Compress a sorted list of indices into "%i" sbatch array format.
# Slurm array supports a comma-separated list of ranges like "1,3,5-9,11".
compress_to_array_spec() {
    local idxs=("$@")
    [ "${#idxs[@]}" -eq 0 ] && { echo ""; return; }
    local out="" run_start="${idxs[0]}" run_end="${idxs[0]}"
    for i in "${idxs[@]:1}"; do
        if [ "${i}" -eq "$((run_end + 1))" ]; then
            run_end="${i}"
        else
            if [ "${run_start}" -eq "${run_end}" ]; then out="${out},${run_start}"
            else out="${out},${run_start}-${run_end}"; fi
            run_start="${i}"; run_end="${i}"
        fi
    done
    if [ "${run_start}" -eq "${run_end}" ]; then out="${out},${run_start}"
    else out="${out},${run_start}-${run_end}"; fi
    echo "${out#,}"
}

# Tracks indices that have been submitted at least once during *this* filler
# run. If a task starts, fails fast, and exits without producing an .h5,
# we don't want the next poll to keep resubmitting it — that wastes
# slurm submit quota and starves the queue of fresh attempts. One shot
# per filler run; user re-runs the filler if they want failed indices
# retried (e.g., after installing missing extras).
declare -A TRIED_IDXS

while true; do
    REMAINING=( $(list_remaining_indices) )
    if [ "${#REMAINING[@]}" -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] All ${N} models have .h5 files — nothing left to submit."
        exit 0
    fi

    IN_QUEUE=$(count_in_queue)
    FREE=$((MAX_SUBMIT_CAP - IN_QUEUE - SAFETY_MARGIN))
    echo "[$(date +%H:%M:%S)] remaining=${#REMAINING[@]}  in_queue=${IN_QUEUE}  free=${FREE}"

    if [ "${FREE}" -le 0 ]; then
        echo "  queue saturated; sleeping ${POLL_SECONDS}s"
        sleep "${POLL_SECONDS}"
        continue
    fi

    # Filter out indices that are already submitted (in squeue under our task name).
    # Cheap heuristic: sbatch with array spec — slurm will reject duplicates by
    # erroring; instead, we just only submit indices whose .h5 doesn't exist
    # AND aren't currently running. The squeue check below handles "not running."
    # When squeue is empty, grep returns 1 and pipefail would kill the
    # script. `|| true` keeps the pipeline status zero so we just get an
    # empty RUNNING_IDXS, which is exactly what we want.
    RUNNING_IDXS=$(squeue -u "$USER" -h -t pending,running -o "%K" -r 2>/dev/null | { grep -E "^[0-9]+$" || true; } | sort -u)
    NEEDS_SUBMIT=()
    SKIPPED_TRIED=0
    for i in "${REMAINING[@]}"; do
        # already in queue: skip
        if grep -qx "${i}" <<< "${RUNNING_IDXS}"; then
            continue
        fi
        # already attempted in this filler run and didn't produce an .h5:
        # don't loop on it. Re-run the filler from scratch to retry.
        if [ -n "${TRIED_IDXS[$i]:-}" ]; then
            SKIPPED_TRIED=$((SKIPPED_TRIED + 1))
            continue
        fi
        NEEDS_SUBMIT+=("${i}")
    done

    if [ "${#NEEDS_SUBMIT[@]}" -eq 0 ]; then
        if [ "${SKIPPED_TRIED}" -gt 0 ]; then
            echo "  ${SKIPPED_TRIED} indices already attempted in this run (no .h5); not resubmitting. Restart filler to retry."
        else
            echo "  all remaining are already in queue; sleeping ${POLL_SECONDS}s"
        fi
        sleep "${POLL_SECONDS}"
        continue
    fi

    # Take a slice that fits in FREE
    SLICE=( "${NEEDS_SUBMIT[@]:0:${FREE}}" )
    SPEC=$(compress_to_array_spec "${SLICE[@]}")

    # sbatch may reject (queue full, partition restrictions, etc.). We
    # don't want one failed submit to kill the whole filler — capture
    # stderr, log it, and keep polling. set -e doesn't trip because
    # `JOB_ID=$(...) || true` keeps the overall command status zero.
    SBATCH_ERR=$(mktemp)
    JOB_ID=$(sbatch \
        --parsable \
        --array="${SPEC}%${THROTTLE}" \
        --export=ALL,MODELS_CSV="${MODELS_CSV}",POOL_CSV="${POOL_CSV}",OUT_DIR="${OUT_DIR}",BATCH_SIZE="${BATCH_SIZE}" \
        "${REPO}/scripts/_extract_array.sbatch" 2>"${SBATCH_ERR}") || JOB_ID=""

    if [ -n "${JOB_ID}" ]; then
        echo "  submitted ${#SLICE[@]} tasks as job ${JOB_ID} (array spec: ${SPEC})"
        # Mark these indices as tried this run so we don't resubmit them on
        # the next poll if they fail fast.
        for i in "${SLICE[@]}"; do TRIED_IDXS[$i]=1; done
    else
        echo "  sbatch FAILED:"
        sed 's/^/    /' "${SBATCH_ERR}"
        echo "  will retry next poll (sbatch error is presumed transient)"
    fi
    rm -f "${SBATCH_ERR}"

    sleep "${POLL_SECONDS}"
done
