#!/usr/bin/env bash
set -euo pipefail

# Raven defaults. Override any of these via env if the path aliases differ.
export REPO=${REPO:-/raven/u/rothj/conwell_replication}
export ENV=${ENV:-/raven/u/rothj/conda-envs/conwell_replication}
export FEATURES=${FEATURES:-/raven/ptmp/rothj/conwell_replication/features}
export OUT=${OUT:-/raven/ptmp/rothj/conwell_replication/eval_min_nn_array_deepnsd}
export MODEL_LIST=${MODEL_LIST:-/raven/ptmp/rothj/conwell_replication/eval_model_manifest_replication.txt}
export MODEL_CONTRASTS=${MODEL_CONTRASTS:-${REPO}/resources/model_contrasts.csv}
export MANIFEST_AUDIT=${MANIFEST_AUDIT:-${MODEL_LIST%.txt}.audit.csv}
export LOG_DIR=${LOG_DIR:-/raven/ptmp/rothj/conwell_replication/.copy_logs}
export LAION_FMRI_ROOT=${LAION_FMRI_ROOT:-/raven/ptmp/rothj/laion_fmri}
export CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE:-/raven/ptmp/rothj/conwell_replication/brain_cache}
export EXTRA_FEATURES_DIR=${EXTRA_FEATURES_DIR:-}
export EVAL_SCRIPT=${EVAL_SCRIPT:-${REPO}/scripts/eval_min_nn_array_viper.sbatch}

# Full design: for each subject, evaluate both the shared pool and the
# subject's full regular pool (shared + unique), across all 12 LAION splits.
export SUBJECTS=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}
export POOL_MODES=${POOL_MODES:-"shared subject"}
export SPLITS=${SPLITS:-"random_0 random_1 random_2 random_3 random_4 cluster_k5_0 cluster_k5_1 cluster_k5_2 cluster_k5_3 cluster_k5_4 tau ood"}

# Keep submitted array elements below MaxSubmitJobs by using waves. Use
# MODELS_PER_TASK=5 for the first profile/full run; increase only after timing.
export MODELS_PER_TASK=${MODELS_PER_TASK:-5}
export CHUNK_SIZE=${CHUNK_SIZE:-300}
export MAX_RUNNING=${MAX_RUNNING:-8}
export TIME_LIMIT=${TIME_LIMIT:-02:00:00}
export SKIP_EXISTING=${SKIP_EXISTING:-1}
export SRPR_VOXELWISE=${SRPR_VOXELWISE:-none}

mkdir -p "${OUT}" "${LOG_DIR}" "$(dirname "${MODEL_LIST}")"
cd "${REPO}"

exec "${REPO}/scripts/submit_eval_min_nn_wave.sh" "$@"
