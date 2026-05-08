#!/bin/bash -l
#SBATCH --job-name=conwell_smoke
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=00:30:00
#SBATCH --output=/raven/ptmp/rothj/conwell_replication/features_smoketest/slurm_%j.log

# Smoke test: alexnet x 100 images on a single A100. Confirms the CUDA
# torch + DeepNSD hook chain works end-to-end on Raven.
#
# We call the env's binaries directly instead of `conda activate`, which
# avoids subtle PATH/PYTHONPATH issues when the user's login shell already
# has (base) conda active.

set -eo pipefail

ENV=/raven/u/rothj/conda-envs/conwell_replication
REPO=/raven/u/rothj/conwell_replication
OUT=/raven/ptmp/rothj/conwell_replication/features_smoketest

export LAION_FMRI_ROOT=/raven/ptmp/rothj/laion_fmri
export CONWELL_STIMULI_ROOT=/raven/ptmp/rothj/conwell_replication/stimuli/image_sets

mkdir -p "${OUT}"

echo "=== node: $(hostname)"
echo "=== nvidia-smi:"
nvidia-smi -L 2>&1 | head

echo "=== env's python sees torch + an A100:"
"${ENV}/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda compiled:", torch.version.cuda)
print("cuda runtime available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

echo "=== running conwell-extract:"
"${ENV}/bin/conwell-extract" \
    --models "${REPO}/resources/conwell_model_list_smoketest.csv" \
    --pool   "${REPO}/features/stimulus_pool_smoketest.csv" \
    --out    "${OUT}" \
    --batch-size 64 \
    --gpu 0

echo "=== output files:"
ls -la "${OUT}"
