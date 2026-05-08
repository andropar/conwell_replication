#!/bin/bash -l
#SBATCH --job-name=conwell_extract
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120000
#SBATCH --time=01:00:00
#SBATCH --output=/raven/ptmp/rothj/conwell_replication/features/slurm_%A_%a.log

# One model per array task. ${SLURM_ARRAY_TASK_ID} indexes a model in
# ${MODELS_CSV}; the extractor's --start/--end slice picks that single row.
# The .h5 path is option_key-derived, so tasks never collide on output.
# Idempotent: if the .h5 already exists, the model is skipped (args.force
# is off by default).
#
# CONFIG (set via env when sbatching):
#   MODELS_CSV  defaults to resources/conwell_model_list_core.csv
#   POOL_CSV    defaults to features/stimulus_pool.csv
#   OUT_DIR     defaults to features/
#   BATCH_SIZE  defaults to 64
# See submit_extract_full_raven.sh for the wrapping launcher.

set -eo pipefail

ENV=/raven/u/rothj/conda-envs/conwell_replication
REPO=/raven/u/rothj/conwell_replication

MODELS_CSV="${MODELS_CSV:-${REPO}/resources/conwell_model_list_core.csv}"
POOL_CSV="${POOL_CSV:-${REPO}/features/stimulus_pool.csv}"
OUT_DIR="${OUT_DIR:-/raven/ptmp/rothj/conwell_replication/features}"
BATCH_SIZE="${BATCH_SIZE:-64}"

export LAION_FMRI_ROOT=/raven/ptmp/rothj/laion_fmri
export CONWELL_STIMULI_ROOT=/raven/ptmp/rothj/conwell_replication/stimuli/image_sets

mkdir -p "${OUT_DIR}"

IDX="${SLURM_ARRAY_TASK_ID}"
NEXT=$((IDX + 1))

echo "=== array task ${IDX}, node $(hostname)"
echo "=== models csv:  ${MODELS_CSV}"
echo "=== pool csv:    ${POOL_CSV}"
echo "=== out dir:     ${OUT_DIR}"
echo "=== model row ${IDX}:"
"${ENV}/bin/python" -c "
import pandas as pd
df = pd.read_csv('${MODELS_CSV}')
if ${IDX} >= len(df):
    raise SystemExit(f'array index ${IDX} >= len(df) = {len(df)}')
row = df.iloc[${IDX}]
print(f'  option_key:   {row[\"option_key\"]}')
print(f'  model_source: {row.get(\"model_source\", \"?\")}')
"

echo "=== nvidia-smi:"
nvidia-smi -L 2>&1 | head

"${ENV}/bin/conwell-extract" \
    --models "${MODELS_CSV}" \
    --pool   "${POOL_CSV}" \
    --out    "${OUT_DIR}" \
    --start  "${IDX}" \
    --end    "${NEXT}" \
    --batch-size "${BATCH_SIZE}" \
    --gpu 0
