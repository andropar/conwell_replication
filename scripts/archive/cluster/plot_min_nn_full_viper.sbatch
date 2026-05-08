#!/bin/bash -l
# Plot all complete min-nn split variants from ROI-wise raw result shards.
#
# This uses per-combo symlink directories so the plotting script only reads the
# shards for one (pool x split) combination at a time. Subject-pool plots combine
# pool=sub-01, pool=sub-03, ... into one "subject" comparison panel.
#SBATCH --job-name=plot_min_nn_full
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=16000
#SBATCH --time=04:00:00
#SBATCH --output=/ptmp/rothj/conwell_replication/.copy_logs/plot_min_nn_full_%j.log

set -euo pipefail

REPO=${REPO:-/u/rothj/conwell_replication}
ENV=${ENV:-/u/rothj/conda-envs/conwell_replication}
RESULTS_DIR=${RESULTS_DIR:-/ptmp/rothj/conwell_replication/eval_min_nn_otc}
LINK_BASE=${LINK_BASE:-/ptmp/rothj/conwell_replication/eval_min_nn_otc_full_links}
AGG_BASE=${AGG_BASE:-/ptmp/rothj/conwell_replication/eval_min_nn_otc_full_links_agg}
OUT_BASE=${OUT_BASE:-${REPO}/figures/min_nn_results}
NOISE_CEILING=${NOISE_CEILING:-/ptmp/rothj/conwell_replication/noise_ceiling/noise_ceiling_ncsnr.csv}
REGION=${REGION:-otc}
NC_METRIC=${NC_METRIC:-rsa}
SUBJECTS=${SUBJECTS:-"sub-01 sub-03 sub-05 sub-06 sub-07"}
SPLITS=${SPLITS:-"random_0 random_1 random_2 random_3 random_4 cluster_k5_0 cluster_k5_1 cluster_k5_2 cluster_k5_3 cluster_k5_4 tau ood"}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "${LINK_BASE}" "${AGG_BASE}" "${OUT_BASE}" /ptmp/rothj/conwell_replication/.copy_logs

echo "=== started: $(date --iso-8601=seconds)"
echo "=== node: $(hostname)"
echo "=== results=${RESULTS_DIR}"
echo "=== out=${OUT_BASE}"
echo "=== region=${REGION}"

build_combo_links() {
    local pool_label=$1
    local split=$2
    local out_dir="${LINK_BASE}/${pool_label}_${split}"
    mkdir -p "${out_dir}"
    find "${out_dir}" -maxdepth 1 -type l -delete

    local sub pattern f
    for sub in ${SUBJECTS}; do
        if [[ "${pool_label}" == "shared" ]]; then
            pattern="${RESULTS_DIR}/results_${sub}_pool-shared_${split}_v0_batch*.parquet"
        else
            pattern="${RESULTS_DIR}/results_${sub}_pool-${sub}_${split}_v0_batch*.parquet"
        fi
        for f in ${pattern}; do
            [[ -e "${f}" ]] || continue
            ln -sfn "${f}" "${out_dir}/$(basename "${f}")"
        done
    done
    local n
    n=$(find "${out_dir}" -maxdepth 1 -name '*.parquet' | wc -l)
    echo "=== links ${pool_label}/${split}: ${n}"
    if [[ "${n}" -ne 590 ]]; then
        echo "Expected 590 shards for ${pool_label}/${split}, found ${n}" >&2
        exit 2
    fi
}

plot_combo() {
    local pool_label=$1
    local split=$2
    local in_dir="${LINK_BASE}/${pool_label}_${split}"
    local out_dir="${OUT_BASE}/${pool_label}_${split}"
    local ood_filter="none"
    if [[ "${split}" == "ood" ]]; then
        ood_filter="none,all"
    fi
    mkdir -p "${out_dir}"
    echo "=== plotting ${pool_label}/${split} -> ${out_dir}"
    "${ENV}/bin/python" "${REPO}/scripts/plot_splithalf_results.py" \
        --results-dir "${in_dir}" \
        --noise-ceiling "${NOISE_CEILING}" \
        --region "${REGION}" \
        --nc-metric "${NC_METRIC}" \
        --filter-ood-type "${ood_filter}" \
        --out-dir "${out_dir}" \
        | tail -35
}

build_agg_links() {
    local pool_label=$1
    local family=$2
    local out_dir="${AGG_BASE}/${pool_label}_${family}_agg"
    mkdir -p "${out_dir}"
    find "${out_dir}" -maxdepth 1 -type l -delete

    local i src f
    for i in 0 1 2 3 4; do
        src="${LINK_BASE}/${pool_label}_${family}_${i}"
        for f in "${src}"/*.parquet; do
            [[ -e "${f}" ]] || continue
            ln -sfn "${f}" "${out_dir}/$(basename "${f}")"
        done
    done
    local n
    n=$(find "${out_dir}" -maxdepth 1 -name '*.parquet' | wc -l)
    echo "=== links ${pool_label}/${family}_agg: ${n}"
    if [[ "${n}" -ne 2950 ]]; then
        echo "Expected 2950 shards for ${pool_label}/${family}_agg, found ${n}" >&2
        exit 2
    fi
}

plot_agg() {
    local combo=$1
    local in_dir="${AGG_BASE}/${combo}"
    local out_dir="${OUT_BASE}/${combo}"
    mkdir -p "${out_dir}"
    echo "=== plotting ${combo} -> ${out_dir}"
    "${ENV}/bin/python" "${REPO}/scripts/plot_splithalf_results.py" \
        --results-dir "${in_dir}" \
        --noise-ceiling "${NOISE_CEILING}" \
        --region "${REGION}" \
        --nc-metric "${NC_METRIC}" \
        --out-dir "${out_dir}" \
        | tail -35
}

for pool_label in shared subject; do
    for split in ${SPLITS}; do
        build_combo_links "${pool_label}" "${split}"
    done
done

for pool_label in shared subject; do
    for split in ${SPLITS}; do
        plot_combo "${pool_label}" "${split}"
    done
done

for pool_label in shared subject; do
    build_agg_links "${pool_label}" random
    build_agg_links "${pool_label}" cluster_k5
    plot_agg "${pool_label}_random_agg"
    plot_agg "${pool_label}_cluster_k5_agg"
done

"${ENV}/bin/python" - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("/u/rothj/conwell_replication/figures/min_nn_results")
rows = []
for d in sorted(p for p in base.iterdir() if p.is_dir()):
    csv = d / "model_metric_subject_scores.csv"
    if not csv.exists():
        continue
    df = pd.read_csv(csv)
    pool, split = d.name.split("_", 1)
    for metric, g in df.groupby("eval_type"):
        rows.append({
            "combo": d.name,
            "pool_plot": pool,
            "split": split,
            "eval_type": metric,
            "n": len(g),
            "n_subjects": g["subject"].nunique(),
            "n_models": g["model"].nunique(),
            "mean": g["test_score"].mean(),
            "median": g["test_score"].median(),
            "min": g["test_score"].min(),
            "max": g["test_score"].max(),
        })
summary = pd.DataFrame(rows)
summary.to_csv(base / "combo_metric_summary.csv", index=False)
print(f"=== wrote {base / 'combo_metric_summary.csv'} ({len(summary)} rows)")
if not summary.empty:
    print(summary[summary["combo"].str.endswith("_agg")].to_string(index=False))
PY

echo "=== finished: $(date --iso-8601=seconds)"
