#!/bin/bash
# Render fig2/3/4 for each (pool, split) combo in the min-nn results.
# Skips combos with no rows; safe to re-run as more shards land.
set -eo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
RESULTS_DIR=${RESULTS_DIR:-/ptmp/rothj/conwell_replication/eval_min_nn_otc}
NOISE_CEILING=${NOISE_CEILING:-/ptmp/rothj/conwell_replication/noise_ceiling/noise_ceiling_ncsnr.csv}
OUT_BASE=${OUT_BASE:-${REPO}/figures/min_nn_results}
REGION=${REGION:-otc}
NC_METRIC=${NC_METRIC:-rsa}

POOLS=${POOLS:-"shared subject"}
SPLITS=${SPLITS:-"random_0 random_1 random_2 random_3 random_4 cluster_k5_0 cluster_k5_1 cluster_k5_2 cluster_k5_3 cluster_k5_4 tau ood"}

mkdir -p "${OUT_BASE}"

for pool in ${POOLS}; do
    for split in ${SPLITS}; do
        out_dir="${OUT_BASE}/${pool}_${split}"
        mkdir -p "${out_dir}"
        # OOD split has many test subsets; keep null (train) + 'all' (aggregate test).
        if [[ "${split}" == "ood" ]]; then
            ood_filter="none,all"
        else
            ood_filter="none"
        fi
        echo "=== ${pool} / ${split} (ood_filter=${ood_filter}) -> ${out_dir}"
        if ! "${ENV}/bin/python" "${REPO}/scripts/plot_splithalf_results.py" \
            --results-dir "${RESULTS_DIR}" \
            --noise-ceiling "${NOISE_CEILING}" \
            --region "${REGION}" \
            --nc-metric "${NC_METRIC}" \
            --filter-pool "${pool}" \
            --filter-split "${split}" \
            --filter-ood-type "${ood_filter}" \
            --out-dir "${out_dir}" 2>&1 | tail -25; then
            echo "  WARN: plotting failed for ${pool}/${split}"
        fi
    done
done
echo "=== all done"
