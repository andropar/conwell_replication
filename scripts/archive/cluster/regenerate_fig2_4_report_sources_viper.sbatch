#!/bin/bash -l
# Regenerate the report source figures after plot-style changes.
# This does not refit models; it only rereads completed result shards.
#SBATCH --job-name=regen_fig2_4
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/regen_fig2_4_%j.log

set -euo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
NOISE_CEILING=${NOISE_CEILING:-/ptmp/rothj/conwell_replication/noise_ceiling/noise_ceiling_ncsnr.csv}
REGION=${REGION:-otc}
NC_METRIC=${NC_METRIC:-rsa}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p /ptmp/rothj/conwell_replication/.copy_logs

plot_one() {
    local results_dir=$1
    local out_dir=$2
    local ood_filter=${3:-}
    echo "=== plotting ${out_dir}"
    local args=(
        --results-dir "${results_dir}"
        --noise-ceiling "${NOISE_CEILING}"
        --region "${REGION}"
        --nc-metric "${NC_METRIC}"
        --out-dir "${out_dir}"
    )
    if [[ -n "${ood_filter}" ]]; then
        args+=(--filter-ood-type "${ood_filter}")
    fi
    "${ENV}/bin/python" "${REPO}/scripts/plot_splithalf_results.py" "${args[@]}" | tail -25
}

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"

plot_one /ptmp/rothj/conwell_replication/eval_splithalf_otc_ncsnr \
    "${REPO}/figures/splithalf_results"
plot_one /ptmp/rothj/conwell_replication/eval_splithalf_otc_ncsnr_ood \
    "${REPO}/figures/splithalf_results_ood"

AGG=/ptmp/rothj/conwell_replication/eval_min_nn_otc_full_links_agg
LINKS=/ptmp/rothj/conwell_replication/eval_min_nn_otc_full_links
OUT=${REPO}/figures/min_nn_results

for combo in shared_random_agg subject_random_agg shared_cluster_k5_agg subject_cluster_k5_agg; do
    plot_one "${AGG}/${combo}" "${OUT}/${combo}"
done

for combo in shared_tau subject_tau; do
    plot_one "${LINKS}/${combo}" "${OUT}/${combo}"
done

for combo in shared_ood subject_ood; do
    plot_one "${LINKS}/${combo}" "${OUT}/${combo}" "none,all"
done

echo "=== finished: $(date --iso-8601=seconds)"
