#!/bin/bash -l
#SBATCH --job-name=eval_sh_stream
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=119000
#SBATCH --time=24:00:00
#SBATCH --output=/raven/ptmp/rothj/conwell_replication/.copy_logs/eval_splithalf_streamed_%A_%a.log

set -euo pipefail

REPO=${REPO:-/raven/u/rothj/conwell_replication}
ENV=${ENV:-/raven/u/rothj/conda-envs/conwell_replication}
OUT=${OUT:-/raven/ptmp/rothj/conwell_replication/eval_splithalf_streamed_otc}
MODELS_CSV=${MODELS_CSV:-${REPO}/resources/conwell_model_list_replication.csv}
MODEL_LIST=${MODEL_LIST:-}
POOL_CSV=${POOL_CSV:-${REPO}/features/stimulus_pool.csv}
TASK_OFFSET=${TASK_OFFSET:-0}
MODELS_PER_TASK=${MODELS_PER_TASK:-1}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SUBJECTS=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}
VOXEL_SET=${VOXEL_SET:-otc}
ROIS=${ROIS:-}
MIN_ROI_VOXELS=${MIN_ROI_VOXELS:-10}
NCSNR_THRESHOLD=${NCSNR_THRESHOLD:-0.2}
INCLUDE_OOD=${INCLUDE_OOD:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
LAYER_CHUNK_SIZE=${LAYER_CHUNK_SIZE:-8}
N_PROJECTIONS=${N_PROJECTIONS:-5960}
SRP_SEED=${SRP_SEED:-0}
NO_CRSA=${NO_CRSA:-0}
NO_WRSA=${NO_WRSA:-0}
NO_SRPR=${NO_SRPR:-0}
DRY_RUN=${DRY_RUN:-0}
REQUIRE_BRAIN_CACHE=${REQUIRE_BRAIN_CACHE:-1}
IMAGE_ROOT_MAPS=${IMAGE_ROOT_MAPS-"/ptmp/rothj/=/raven/ptmp/rothj/"}

export LAION_FMRI_ROOT=${LAION_FMRI_ROOT:-/raven/ptmp/rothj/laion_fmri}
export CONWELL_BRAIN_CACHE=${CONWELL_BRAIN_CACHE:-/raven/ptmp/rothj/conwell_replication/brain_cache_otc}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-18}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-18}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-18}
export PYTHONUNBUFFERED=1

mkdir -p "${OUT}" /raven/ptmp/rothj/conwell_replication/.copy_logs
cd "${REPO}"

if (( MODELS_PER_TASK <= 0 )); then
    echo "MODELS_PER_TASK must be positive, got ${MODELS_PER_TASK}" >&2
    exit 2
fi

MODEL_ARGS=()
if [[ -n "${MODEL_LIST}" ]]; then
    if [[ ! -s "${MODEL_LIST}" ]]; then
        echo "MODEL_LIST does not exist or is empty: ${MODEL_LIST}" >&2
        exit 2
    fi
    N_MODELS=$(grep -vc '^[[:space:]]*#\|^[[:space:]]*$' "${MODEL_LIST}")
    MODEL_ARGS=(--model-list "${MODEL_LIST}")
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
    MODEL_ARGS=(--models "${MODELS_CSV}")
fi

N_MODEL_BATCHES=$(((N_MODELS + MODELS_PER_TASK - 1) / MODELS_PER_TASK))
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}
GLOBAL_ID=$((TASK_OFFSET + ARRAY_ID))
if (( GLOBAL_ID >= N_MODEL_BATCHES )); then
    echo "GLOBAL_ID=${GLOBAL_ID} >= N_MODEL_BATCHES=${N_MODEL_BATCHES}; exiting"
    exit 0
fi

BATCH_IDX=${GLOBAL_ID}
MODEL_START=$((BATCH_IDX * MODELS_PER_TASK))
MODEL_STOP=$((MODEL_START + MODELS_PER_TASK))
if (( MODEL_STOP > N_MODELS )); then
    MODEL_STOP=${N_MODELS}
fi

read -r -a SUBJECT_ARR <<< "${SUBJECTS}"
if [[ "${REQUIRE_BRAIN_CACHE}" == "1" ]]; then
    "${ENV}/bin/python" - "${CONWELL_BRAIN_CACHE}" "${VOXEL_SET}" "${SUBJECT_ARR[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
voxel_set = sys.argv[2]
subjects = sys.argv[3:]
bad = []
for subject in subjects:
    subject_dir = root / subject
    info_path = subject_dir / "cache_info.json"
    betas_path = subject_dir / "betas.npy"
    image_ids_path = subject_dir / "image_ids.npy"
    if not betas_path.exists() or not image_ids_path.exists() or not info_path.exists():
        bad.append(f"{subject}: missing betas.npy, image_ids.npy, or cache_info.json")
        continue
    info = json.loads(info_path.read_text())
    if info.get("voxel_set") != voxel_set:
        bad.append(
            f"{subject}: cache voxel_set={info.get('voxel_set')!r}, expected {voxel_set!r}"
        )
if bad:
    print("Brain cache preflight failed:", file=sys.stderr)
    for item in bad:
        print(f"  - {item}", file=sys.stderr)
    print(
        "Build it first with scripts/archive/cluster/cache_brain_otc_raven.sbatch "
        "or set REQUIRE_BRAIN_CACHE=0 to allow slow raw loading.",
        file=sys.stderr,
    )
    sys.exit(2)
PY
fi

ROI_ARGS=()
if [[ -n "${ROIS}" ]]; then
    read -r -a ROI_ARR <<< "${ROIS}"
    ROI_ARGS=(--rois "${ROI_ARR[@]}")
fi

EXTRA_ARGS=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-existing)
fi
if [[ "${INCLUDE_OOD}" == "1" ]]; then
    EXTRA_ARGS+=(--include-ood)
fi
if [[ "${NO_CRSA}" == "1" ]]; then
    EXTRA_ARGS+=(--no-crsa)
fi
if [[ "${NO_WRSA}" == "1" ]]; then
    EXTRA_ARGS+=(--no-wrsa)
fi
if [[ "${NO_SRPR}" == "1" ]]; then
    EXTRA_ARGS+=(--no-srpr)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi
if [[ -n "${IMAGE_ROOT_MAPS}" ]]; then
    read -r -a IMAGE_ROOT_MAP_ARR <<< "${IMAGE_ROOT_MAPS}"
    for image_root_map in "${IMAGE_ROOT_MAP_ARR[@]}"; do
        EXTRA_ARGS+=(--image-root-map "${image_root_map}")
    done
fi

OUT_PATH="${OUT}/results_streamed_batch${BATCH_IDX}.parquet"
if [[ "${SKIP_EXISTING}" == "1" && -s "${OUT_PATH}" ]]; then
    echo "=== skipping existing output: ${OUT_PATH}"
    exit 0
fi

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== job/task: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}/${ARRAY_ID}"
echo "=== model_batch=${BATCH_IDX}/${N_MODEL_BATCHES}"
echo "=== models_per_task=${MODELS_PER_TASK} model_idx_range=${MODEL_START}..$((MODEL_STOP - 1))/${N_MODELS}"
echo "=== models_csv=${MODELS_CSV}"
echo "=== model_list=${MODEL_LIST:-<none>}"
echo "=== pool_csv=${POOL_CSV}"
echo "=== subjects=${SUBJECTS}"
echo "=== voxel_set=${VOXEL_SET} brain_cache=${CONWELL_BRAIN_CACHE}"
echo "=== image_root_maps=${IMAGE_ROOT_MAPS:-<none>}"
echo "=== nvidia-smi:"
nvidia-smi -L 2>&1 | head || true

/usr/bin/time -v "${ENV}/bin/python" -m conwell_replication.eval.rsa_splithalf_streamed \
    "${MODEL_ARGS[@]}" \
    --pool-csv "${POOL_CSV}" \
    --out "${OUT}" \
    --subjects "${SUBJECT_ARR[@]}" \
    --voxel-set "${VOXEL_SET}" \
    "${ROI_ARGS[@]}" \
    --min-roi-voxels "${MIN_ROI_VOXELS}" \
    --ncsnr-threshold "${NCSNR_THRESHOLD}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --layer-chunk-size "${LAYER_CHUNK_SIZE}" \
    --n-projections "${N_PROJECTIONS}" \
    --srp-seed "${SRP_SEED}" \
    --model-batch "${BATCH_IDX}" \
    --models-per-batch "${MODELS_PER_TASK}" \
    --gpu 0 \
    "${EXTRA_ARGS[@]}"

echo "=== finished: $(date --iso-8601=seconds)"
