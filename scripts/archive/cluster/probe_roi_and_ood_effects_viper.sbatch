#!/bin/bash -l
#SBATCH --job-name=probe_roi_ood
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --time=00:30:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/probe_roi_ood_%j.log

set -eo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}

mkdir -p /ptmp/rothj/conwell_replication/.copy_logs

echo "=== started: $(date --iso-8601=seconds) on $(hostname)"

"${ENV}/bin/python" "${REPO}/scripts/probe_roi_and_ood_effects.py"

echo "=== done: $(date --iso-8601=seconds)"
