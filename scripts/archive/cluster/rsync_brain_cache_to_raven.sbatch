#!/bin/bash -l
#SBATCH --job-name=rsync_brain_cache
#SBATCH --partition=datatransfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --time=06:00:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/rsync_brain_cache_%j.log

set -eo pipefail

SRC=${SRC:-/ptmp/rothj/conwell_replication/brain_cache/}
DST=${DST:-/raven/ptmp/rothj/conwell_replication/brain_cache/}

mkdir -p "${DST}" /ptmp/rothj/conwell_replication/.copy_logs

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== src: ${SRC}"
echo "=== dst: ${DST}"
du -sh "${SRC}" || true

rsync -a \
    --partial \
    --append-verify \
    --human-readable \
    --info=progress2 \
    --stats \
    "${SRC}" "${DST}"

echo "=== destination:"
du -sh "${DST}" || true
find "${DST}" -maxdepth 2 -type f -printf '%p %s\n' | sort
echo "=== finished: $(date --iso-8601=seconds)"
