#!/bin/bash -l
#SBATCH --job-name=rsync_deepvision_old
#SBATCH --partition=datatransfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --time=01:00:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/rsync_deepvision_old_%j.log

set -eo pipefail

SRC_ROOT=${SRC_ROOT:-/raven/ptmp/rothj/deepvision_fmri}
DST_ROOT=${DST_ROOT:-/ptmp/rothj/deepvision_fmri_old}
SUBJECTS_STR=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}

mkdir -p "${DST_ROOT}/cache/voxel_sets/deepvision_shared_hlvis/finalinterp" \
         "${DST_ROOT}/cache/image_sets" \
         "${DST_ROOT}/derivatives/functional/1sTR_1pt5mm/atlas" \
         /ptmp/rothj/conwell_replication/.copy_logs

echo "=== started: $(date --iso-8601=seconds) on $(hostname)"
echo "=== SRC_ROOT=${SRC_ROOT} DST_ROOT=${DST_ROOT}"

# Per-subject hlvis betas. Skip subjects that aren't cached in OLD root.
read -r -a SUBJECT_ARR <<< "${SUBJECTS_STR}"
for sub in "${SUBJECT_ARR[@]}"; do
    SRC="${SRC_ROOT}/cache/voxel_sets/deepvision_shared_hlvis/finalinterp/${sub}/"
    DST="${DST_ROOT}/cache/voxel_sets/deepvision_shared_hlvis/finalinterp/${sub}/"
    if [ ! -d "${SRC}" ]; then
        echo "--- ${sub}: SKIP (not cached upstream)"
        continue
    fi
    mkdir -p "${DST}"
    echo "--- ${sub} hlvis betas"
    rsync -ah --no-perms --info=stats2 "${SRC}" "${DST}"
done

# Per-subject hlvis_p20 ROI masks + T1 brain (~10 MB each, where available).
for sub in "${SUBJECT_ARR[@]}"; do
    SRC="${SRC_ROOT}/derivatives/functional/1sTR_1pt5mm/atlas/${sub}/"
    DST="${DST_ROOT}/derivatives/functional/1sTR_1pt5mm/atlas/${sub}/"
    if [ ! -d "${SRC}" ]; then
        echo "--- ${sub} ROI masks: SKIP (no atlas dir)"
        continue
    fi
    mkdir -p "${DST}"
    echo "--- ${sub} ROI masks"
    rsync -ah --no-perms --include='hlvis_p20_mask_func.nii.gz' \
                       --include='HCPMMP1_func.nii.gz' \
                       --include='*T1w_brain*.nii.gz' \
                       --exclude='*' \
                       "${SRC}" "${DST}"
done

# Image set + metadata files.
echo "--- image_sets metadata"
rsync -ah --no-perms \
      "${SRC_ROOT}/cache/image_sets/deepvision_shared.csv" \
      "${DST_ROOT}/cache/image_sets/"
for f in metadata_p01.csv metadata_p02.csv metadata_p03.csv metadata_p04.csv metadata_p05.csv; do
    if [ -f "${SRC_ROOT}/${f}" ]; then
        rsync -ah --no-perms "${SRC_ROOT}/${f}" "${DST_ROOT}/${f}"
    fi
done

echo "=== done: $(date --iso-8601=seconds)"
echo "=== final size:"
du -sh "${DST_ROOT}"
echo "=== per-subject:"
for sub in "${SUBJECT_ARR[@]}"; do
    du -sh "${DST_ROOT}/cache/voxel_sets/deepvision_shared_hlvis/finalinterp/${sub}" 2>/dev/null
done
