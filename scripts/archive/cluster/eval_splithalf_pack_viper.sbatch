#!/bin/bash -l
#SBATCH --job-name=eval_splithalf_pack
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=128000
#SBATCH --time=02:00:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/eval_splithalf_pack_%A_%a.log

set -euo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
TASK_SCRIPT=${TASK_SCRIPT:-${REPO}/scripts/eval_splithalf_array_viper.sbatch}
LOG_DIR=${LOG_DIR:-/ptmp/rothj/conwell_replication/.copy_logs}

PACK_SIZE=${PACK_SIZE:-8}
INNER_CPUS=${INNER_CPUS:-4}
TASK_OFFSET=${TASK_OFFSET:-0}

mkdir -p "${LOG_DIR}"

PACK_ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}
FIRST_GLOBAL_ID=$((TASK_OFFSET + PACK_ARRAY_ID * PACK_SIZE))

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== job/task: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}/${PACK_ARRAY_ID}"
echo "=== partition=${SLURM_JOB_PARTITION:-unknown} cpus=${SLURM_CPUS_PER_TASK:-unknown}"
echo "=== first_global_id=${FIRST_GLOBAL_ID} pack_size=${PACK_SIZE} inner_cpus=${INNER_CPUS}"
echo "=== task_script=${TASK_SCRIPT}"

status=0
pids=()
for ((i = 0; i < PACK_SIZE; i++)); do
    global_id=$((FIRST_GLOBAL_ID + i))
    log="${LOG_DIR}/eval_splithalf_pack_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${PACK_ARRAY_ID}_${global_id}.log"
    (
        export TASK_OFFSET="${global_id}"
        export SLURM_ARRAY_TASK_ID=0
        export SLURM_CPUS_PER_TASK="${INNER_CPUS}"
        export OMP_NUM_THREADS="${INNER_CPUS}"
        export MKL_NUM_THREADS="${INNER_CPUS}"
        export OPENBLAS_NUM_THREADS="${INNER_CPUS}"
        export MODELS_PER_TASK=1
        bash "${TASK_SCRIPT}"
    ) >"${log}" 2>&1 &
    pids+=("$!")
    echo "=== launched global_id=${global_id} pid=${pids[-1]} log=${log}"
done

for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

echo "=== finished: $(date --iso-8601=seconds) status=${status}"
exit "${status}"
