#!/bin/bash -l
#SBATCH --job-name=plot_min_nn_all
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=125000
#SBATCH --time=02:00:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/plot_min_nn_all_%j.log

set -eo pipefail

mkdir -p /ptmp/rothj/conwell_replication/.copy_logs
echo "=== started: $(date --iso-8601=seconds) on $(hostname)"
bash /u/rothj/conwell_replication/scripts/plot_min_nn_all_splits.sh
echo "=== finished: $(date --iso-8601=seconds)"
