#!/bin/bash -l
#SBATCH --job-name=rsync_features_ood
#SBATCH --partition=datatransfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --time=00:30:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/rsync_features_ood_%j.log

set -eo pipefail

SRC=${SRC:-/raven/ptmp/rothj/conwell_replication/features_ood/}
DST=${DST:-/ptmp/rothj/conwell_replication/features_ood/}

mkdir -p "${DST}" /ptmp/rothj/conwell_replication/.copy_logs

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== SRC=${SRC}"
echo "=== DST=${DST}"

rsync -avh --no-perms --info=progress2 "${SRC}" "${DST}"

echo "=== verify"
echo "  src h5 count: $(ls "${SRC}"*.h5 2>/dev/null | wc -l)"
echo "  dst h5 count: $(ls "${DST}"*.h5 2>/dev/null | wc -l)"
du -sh "${DST}"

echo "=== finished: $(date --iso-8601=seconds)"
