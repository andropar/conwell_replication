#!/bin/bash
# Submit the smoke-test sbatch on Raven.
# Run on a Raven login node, NOT Viper (slurm controllers are siloed).

set -eo pipefail

REPO=/raven/u/rothj/conwell_replication
OUT=/raven/ptmp/rothj/conwell_replication/features_smoketest

mkdir -p "${OUT}"

JOB_ID=$(sbatch --parsable "${REPO}/scripts/_smoketest_job.sbatch")

echo "Submitted smoke test: job ${JOB_ID}"
echo "  log:     ${OUT}/slurm_${JOB_ID}.log"
echo "  monitor: squeue -j ${JOB_ID}"
echo "  watch:   tail -f ${OUT}/slurm_${JOB_ID}.log"
