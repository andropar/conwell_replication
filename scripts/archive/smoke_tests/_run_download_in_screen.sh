#!/bin/bash
# Wrapper that the screen session executes: load conda env, then run the
# downloader. Kept separate so the screen-detach logic stays simple.
set -e
source /etc/profile.d/modules.sh
module load anaconda/3/2023.03
source activate /u/rothj/conda-envs/conwell_replication
exec python /u/rothj/conwell_replication/scripts/download_laion_fmri.py
