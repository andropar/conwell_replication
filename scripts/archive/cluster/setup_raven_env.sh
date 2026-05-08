#!/bin/bash
# Setup the conwell_replication conda env on Raven.
# Run this ON RAVEN (i.e. after `ssh raven01.mpcdf.mpg.de`), not on Viper —
# the Python interpreter and CUDA/PyTorch wheels are cluster-specific.
#
# Mirrors the Viper setup but with NVIDIA CUDA torch instead of ROCm:
#   /raven/u/rothj/conda-envs/conwell_replication
#   torch == 2.11.0 (CUDA build for Raven A100s)
#   LAION_FMRI_ROOT       = /raven/ptmp/rothj/laion_fmri
#   CONWELL_STIMULI_ROOT  = /raven/ptmp/rothj/conwell_replication/stimuli/image_sets
#
# We *don't* `conda activate` here — Raven's default shell already has
# anaconda/3/2021.11 active, and nested activation runs unrelated
# deactivate hooks (e.g. proj4-deactivate.sh) that can return non-zero
# and trip set -e. Calling the env's own pip/python binaries directly
# avoids that whole chain.

set -eo pipefail

ENV_PATH=/raven/u/rothj/conda-envs/conwell_replication
LAION_FMRI_REPO=/raven/u/rothj/LAION-fMRI
CONWELL_REPO=/raven/u/rothj/conwell_replication

ENV_PIP="${ENV_PATH}/bin/pip"
ENV_PYTHON="${ENV_PATH}/bin/python"

echo "=== Loading anaconda module"
module load anaconda/3/2023.03

if [ ! -x "${ENV_PYTHON}" ]; then
    echo "=== Creating env at ${ENV_PATH}"
    conda create -y -p "${ENV_PATH}" python=3.11 pip
else
    echo "=== Reusing existing env at ${ENV_PATH}"
fi

echo "=== Installing laion_fmri (editable)"
"${ENV_PIP}" install -e "${LAION_FMRI_REPO}"

echo "=== Installing conwell_replication (editable, core deps incl. CUDA torch)"
"${ENV_PIP}" install -e "${CONWELL_REPO}"

echo "=== Verifying torch is CUDA-built (not ROCm)"
"${ENV_PYTHON}" -c "
import torch
assert '+rocm' not in torch.__version__, f'wrong torch: {torch.__version__}'
print('torch:', torch.__version__)
print('cuda compiled:', torch.version.cuda)
print('cuda runtime available:', torch.cuda.is_available())
"

echo "=== Wiring activate/deactivate hooks for env vars"
ACTIVATE_DIR="${ENV_PATH}/etc/conda/activate.d"
DEACTIVATE_DIR="${ENV_PATH}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/conwell_env.sh" <<'EOF'
export LAION_FMRI_ROOT=/raven/ptmp/rothj/laion_fmri
export CONWELL_STIMULI_ROOT=/raven/ptmp/rothj/conwell_replication/stimuli/image_sets
EOF

cat > "${DEACTIVATE_DIR}/conwell_env.sh" <<'EOF'
unset LAION_FMRI_ROOT
unset CONWELL_STIMULI_ROOT
EOF

echo "=== DONE."
echo "To use the env (works even with anaconda/3/2021.11 already loaded):"
echo "  module load anaconda/3/2023.03"
echo "  conda activate ${ENV_PATH}"
echo "  echo \$LAION_FMRI_ROOT \$CONWELL_STIMULI_ROOT"
