#!/bin/bash
# Smoke test: extract features for one model on 100 images on a single
# AMD APU GPU on Viper, to confirm the ROCm 7.2 torch + DeepNSD path works
# end-to-end before committing the full 335-model run.
#
# Usage:
#   bash submit_extract_smoketest.sh
#   bash submit_extract_smoketest.sh --dry-run

set -euo pipefail

CONDA_ENV="/u/rothj/conda-envs/conwell_replication"
REPO_DIR="/u/rothj/conwell_replication"
# Per MPCDF Viper-GPU docs: use --constraint=apu + --gres=gpu:N; the submit
# filter routes to the right partition. Single-APU job = gpu:1, 24 CPUs, 110 GB.
TIME="00:30:00"
OUT_DIR="/ptmp/rothj/conwell_replication/features_smoketest"

DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
    esac
done

mkdir -p "${OUT_DIR}"

CMD=$(cat <<EOF
module load anaconda/3/2023.03
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${REPO_DIR}

echo "=== node: \$(hostname)"
echo "=== ROCm visible devices: \${ROCR_VISIBLE_DEVICES:-unset} / HIP: \${HIP_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('torch', torch.__version__, 'gpu_available:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"

conwell-extract \\
    --models ${REPO_DIR}/resources/conwell_model_list_smoketest.csv \\
    --pool   ${REPO_DIR}/features/stimulus_pool_smoketest.csv \\
    --out    ${OUT_DIR} \\
    --batch-size 32 \\
    --gpu 0
EOF
)

if $DRY_RUN; then
    echo "[DRY RUN] would sbatch with:"
    echo "  partition: ${PARTITION}"
    echo "  time:      ${TIME}"
    echo "  gres:      gpu:1"
    echo "  out:       ${OUT_DIR}"
    echo "--- wrapped command ---"
    echo "${CMD}"
    exit 0
fi

JOB_ID=$(sbatch \
    --parsable \
    --constraint=apu \
    --job-name="conwell_smoke" \
    --gres=gpu:1 \
    --cpus-per-task=24 \
    --mem=110000 \
    --time="${TIME}" \
    --output="${OUT_DIR}/slurm_%j.log" \
    --export=ALL \
    --wrap="${CMD}")

echo "Submitted smoke test: job ${JOB_ID}"
echo "  log:  ${OUT_DIR}/slurm_${JOB_ID}.log"
echo "  monitor: squeue -j ${JOB_ID}; tail -f ${OUT_DIR}/slurm_${JOB_ID}.log"
