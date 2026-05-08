#!/bin/bash -l
#SBATCH --job-name=auto_submit_min_nn_pack
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=4000
#SBATCH --time=24:00:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/auto_submit_min_nn_pack_%j.log

set -eo pipefail

bash /u/rothj/conwell_replication/scripts/auto_submit_eval_min_nn_pack.sh
