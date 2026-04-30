#!/usr/bin/env python3
"""Run all statistical tests from Conwell et al. (2024).

Implements the same fixed-effects linear models (with subject as fixed effect)
and correlation analyses described in the paper, applied to our DeepVision
413-model benchmark.

Outputs: statistical_results.json with all test results.
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Set by main() — drives all RESULTS_DIR-relative I/O below.
RESULTS_DIR = Path(__file__).parent
_PKG_RESOURCES = Path(__file__).resolve().parents[3] / "resources"


def load_data():
    scores = pd.read_csv(RESULTS_DIR / "best_layer_scores.csv")
    meta_path = RESULTS_DIR / "model_metadata.csv"
    if not meta_path.exists():
        meta_path = _PKG_RESOURCES / "model_metadata.csv"
    meta = pd.read_csv(meta_path)
    nc = pd.read_csv(RESULTS_DIR / "noise_ceilings.csv")
    return scores, meta, nc


def run_fixed_effects(df, formula, reference_info=""):
    """Run OLS fixed-effects model and extract key statistics."""
    model = smf.ols(formula, data=df).fit()
    results = {
        "formula": formula,
        "reference": reference_info,
        "n_obs": int(model.nobs),
        "r_squared": round(model.rsquared, 4),
        "coefficients": {},
    }
    for name in model.params.index:
        if name == "Intercept":
            continue
        results["coefficients"][name] = {
            "beta": round(model.params[name], 4),
            "ci_lo": round(model.conf_int().loc[name, 0], 4),
            "ci_hi": round(model.conf_int().loc[name, 1], 4),
            "t": round(model.tvalues[name], 3),
            "p": float(f"{model.pvalues[name]:.6g}"),
            "se": round(model.bse[name], 4),
        }
    return results


def run_mixed_effects(df, formula, groups):
    """Run linear mixed-effects model."""
    try:
        model = smf.mixedlm(formula, data=df, groups=groups).fit(reml=True)
        results = {
            "formula": formula,
            "groups": groups,
            "n_obs": int(model.nobs),
            "coefficients": {},
        }
        for name in model.fe_params.index:
            if name == "Intercept":
                continue
            results["coefficients"][name] = {
                "beta": round(model.fe_params[name], 4),
                "ci_lo": round(model.conf_int().loc[name, 0], 4),
                "ci_hi": round(model.conf_int().loc[name, 1], 4),
                "z": round(model.tvalues[name], 3),
                "p": float(f"{model.pvalues[name]:.6g}"),
            }
        return results
    except Exception as e:
        return {"error": str(e)}


def descriptive_stats(values, label=""):
    """Compute descriptive statistics with bootstrap CI."""
    mean = values.mean()
    n_boot = 1000
    boot = np.array([
        np.random.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    return {
        "label": label,
        "n": len(values),
        "mean": round(float(mean), 4),
        "std": round(float(values.std()), 4),
        "ci_lo": round(float(np.percentile(boot, 2.5)), 4),
        "ci_hi": round(float(np.percentile(boot, 97.5)), 4),
    }


def main():
    np.random.seed(42)
    scores, meta, nc = load_data()
    all_results = {}

    print("=" * 70)
    print("STATISTICAL TESTS: Replication of Conwell et al. (2024)")
    print("=" * 70)

    # ================================================================
    # 1. ARCHITECTURE: CNN vs Transformer
    # ================================================================
    print("\n--- 1. Architecture: CNN vs Transformer ---")

    cnn_models = set(meta[meta["comparison_group"] == "arch_cnn_in1k"]["model"])
    trans_models = set(meta[meta["comparison_group"] == "arch_transformer_in1k"]["model"])
    arch_all = cnn_models | trans_models
    arch_scores = scores[scores["model"].isin(arch_all)].copy()
    arch_scores["is_transformer"] = arch_scores["model"].isin(trans_models).astype(int)

    arch_results = {}
    for et in ["crsa", "wrsa"]:
        df = arch_scores[arch_scores["eval_type"] == et].copy()
        result = run_fixed_effects(
            df,
            "test_score ~ is_transformer + C(subject)",
            reference_info="CNN (is_transformer=0)"
        )
        arch_results[et] = result

        # Descriptive stats
        cnn_vals = df[df["is_transformer"] == 0].groupby("model")["test_score"].mean()
        trans_vals = df[df["is_transformer"] == 1].groupby("model")["test_score"].mean()
        arch_results[f"{et}_cnn_desc"] = descriptive_stats(cnn_vals.values, "CNN")
        arch_results[f"{et}_trans_desc"] = descriptive_stats(trans_vals.values, "Transformer")

        beta = result["coefficients"]["is_transformer"]
        print(f"  {et.upper()}: beta={beta['beta']:.4f} [{beta['ci_lo']:.4f}, {beta['ci_hi']:.4f}], "
              f"p={beta['p']:.4g}")
        print(f"    CNN mean={cnn_vals.mean():.4f}, Transformer mean={trans_vals.mean():.4f}")

    all_results["architecture"] = arch_results

    # ================================================================
    # 2. TASK: Taskonomy
    # ================================================================
    print("\n--- 2. Task: Taskonomy ---")

    task_models = meta[meta["comparison_group"] == "task_taskonomy"]
    task_scores = scores.merge(task_models[["model"]], on="model")

    # Extract task name from model name
    task_scores["task"] = task_scores["model"].str.replace("_taskonomy", "")

    task_results = {}
    for et in ["crsa", "wrsa"]:
        df = task_scores[task_scores["eval_type"] == et].copy()

        # Descriptive per task
        task_means = df.groupby("task")["test_score"].mean().sort_values(ascending=False)
        task_results[f"{et}_task_ranking"] = {
            task: round(float(score), 4) for task, score in task_means.items()
        }
        print(f"  {et.upper()} best: {task_means.index[0]} ({task_means.iloc[0]:.4f})")
        print(f"  {et.upper()} worst: {task_means.index[-1]} ({task_means.iloc[-1]:.4f})")

        # Fixed-effects model with task as factor (reference = denoising)
        df["task_cat"] = pd.Categorical(df["task"])
        result = run_fixed_effects(
            df,
            "test_score ~ C(task, Treatment(reference='denoising')) + C(subject)",
            reference_info="denoising"
        )
        task_results[et] = result

    all_results["taskonomy"] = task_results

    # ================================================================
    # 3. TASK: Self-Supervised Learning (Contrastive vs Non-Contrastive)
    # ================================================================
    print("\n--- 3. Task: Self-Supervised Learning ---")

    vissl_models = meta[meta["comparison_group"] == "task_vissl"]
    vissl_scores = scores.merge(vissl_models[["model"]], on="model")

    # Classify contrastive vs non-contrastive
    contrastive_methods = [
        "vissl_resnet50_barlowtwins", "vissl_resnet50_deepclusterv2",
        "vissl_resnet50_mocov2", "vissl_resnet50_swav",
        "vissl_resnet50_simclr", "vissl_resnet50_pirl",
        "vissl_resnet50_npid",
    ]
    # Also include non-VISSL self-supervised ResNet50 models in the comparison
    vissl_scores["is_contrastive"] = vissl_scores["model"].isin(contrastive_methods).astype(int)

    ssl_results = {}
    for et in ["crsa", "wrsa"]:
        df = vissl_scores[vissl_scores["eval_type"] == et].copy()
        result = run_fixed_effects(
            df,
            "test_score ~ is_contrastive + C(subject)",
            reference_info="Non-Contrastive (is_contrastive=0)"
        )
        ssl_results[et] = result

        c_vals = df[df["is_contrastive"] == 1].groupby("model")["test_score"].mean()
        nc_vals = df[df["is_contrastive"] == 0].groupby("model")["test_score"].mean()
        ssl_results[f"{et}_contrastive_desc"] = descriptive_stats(c_vals.values, "Contrastive")
        ssl_results[f"{et}_noncontrastive_desc"] = descriptive_stats(nc_vals.values, "Non-Contrastive")

        beta = result["coefficients"]["is_contrastive"]
        print(f"  {et.upper()}: beta={beta['beta']:.4f} [{beta['ci_lo']:.4f}, {beta['ci_hi']:.4f}], "
              f"p={beta['p']:.4g}")

    all_results["ssl_contrastive"] = ssl_results

    # ================================================================
    # 4. TASK: SLIP Language Alignment
    # ================================================================
    print("\n--- 4. Task: SLIP Language Alignment ---")

    slip_models = meta[meta["comparison_group"] == "task_slip"]
    slip_scores = scores.merge(slip_models[["model"]], on="model")

    # Extract objective and size
    def parse_slip(model_name):
        name = model_name.lower()
        if "simclr" in name:
            obj = "simclr"
        elif "slip" in name and "clip" not in name:
            obj = "slip"
        elif "clip" in name:
            obj = "clip"
        else:
            obj = "unknown"

        if "vit_s" in name or "vit/s" in name.replace(" ", ""):
            size = "small"
        elif "vit_l" in name or "vit/l" in name.replace(" ", ""):
            size = "large"
        else:
            size = "base"
        return obj, size

    slip_scores[["objective", "model_size"]] = slip_scores["model"].apply(
        lambda x: pd.Series(parse_slip(x))
    )

    slip_results = {}
    for et in ["crsa", "wrsa"]:
        df = slip_scores[slip_scores["eval_type"] == et].copy()
        df["objective"] = pd.Categorical(df["objective"], categories=["simclr", "clip", "slip"])

        result = run_fixed_effects(
            df,
            "test_score ~ C(objective, Treatment(reference='simclr')) + C(subject)",
            reference_info="SimCLR"
        )
        slip_results[et] = result

        for coef_name, coef_data in result["coefficients"].items():
            if "subject" not in coef_name:
                print(f"  {et.upper()} {coef_name}: beta={coef_data['beta']:.4f}, p={coef_data['p']:.4g}")

    all_results["slip"] = slip_results

    # ================================================================
    # 5. INPUT: ImageNet1K vs ImageNet21K
    # ================================================================
    print("\n--- 5. Input: ImageNet1K vs ImageNet21K ---")

    in_models = meta[meta["comparison_group"] == "input_in1k_vs_in21k"]
    in_scores = scores.merge(in_models[["model"]], on="model")
    in_scores["is_21k"] = in_scores["model"].str.contains(
        "in21k|in22k|in22", case=False
    ).astype(int)

    # Extract base architecture for random effect
    def get_base_arch(model_name):
        name = model_name.lower()
        for suffix in ["_in21k_classification", "_in22k_classification",
                        "_in22ft1k_classification", "_classification"]:
            name = name.replace(suffix, "")
        return name

    in_scores["base_arch"] = in_scores["model"].apply(get_base_arch)

    input_size_results = {}
    for et in ["crsa", "wrsa"]:
        df = in_scores[in_scores["eval_type"] == et].copy()

        # Mixed-effects model with random intercept for architecture
        result = run_mixed_effects(
            df,
            "test_score ~ is_21k + C(subject)",
            groups="base_arch"
        )
        input_size_results[et] = result

        if "coefficients" in result and "is_21k" in result["coefficients"]:
            beta = result["coefficients"]["is_21k"]
            print(f"  {et.upper()}: beta={beta['beta']:.4f} [{beta['ci_lo']:.4f}, {beta['ci_hi']:.4f}], "
                  f"p={beta['p']:.4g}")
        elif "error" in result:
            # Fall back to fixed effects
            result_fe = run_fixed_effects(
                df,
                "test_score ~ is_21k + C(subject)",
                reference_info="ImageNet1K"
            )
            input_size_results[et] = result_fe
            beta = result_fe["coefficients"]["is_21k"]
            print(f"  {et.upper()} (FE): beta={beta['beta']:.4f} [{beta['ci_lo']:.4f}, {beta['ci_hi']:.4f}], "
                  f"p={beta['p']:.4g}")

    all_results["input_size"] = input_size_results

    # ================================================================
    # 6. INPUT: Domain-Specific (IPCL)
    # ================================================================
    print("\n--- 6. Input: Domain-Specific (IPCL) ---")

    ipcl_models = meta[meta["comparison_group"] == "input_ipcl"]
    ipcl_scores = scores.merge(ipcl_models[["model"]], on="model")
    ipcl_scores["dataset"] = ipcl_scores["model"].str.replace("alexnet_gn_ipcl_", "")

    ipcl_results = {}
    for et in ["crsa", "wrsa"]:
        df = ipcl_scores[ipcl_scores["eval_type"] == et].copy()
        df["dataset"] = pd.Categorical(
            df["dataset"],
            categories=["imagenet", "openimages", "places256", "vggface2"]
        )

        result = run_fixed_effects(
            df,
            "test_score ~ C(dataset, Treatment(reference='imagenet')) + C(subject)",
            reference_info="imagenet"
        )
        ipcl_results[et] = result

        for coef_name, coef_data in result["coefficients"].items():
            if "subject" not in coef_name:
                print(f"  {et.upper()} {coef_name}: beta={coef_data['beta']:.4f}, p={coef_data['p']:.4g}")

    all_results["ipcl_domains"] = ipcl_results

    # ================================================================
    # 7. cRSA vs WRSA Linking Method
    # ================================================================
    print("\n--- 7. cRSA vs WRSA Linking Method ---")

    method_df = scores[scores["eval_type"].isin(["crsa", "wrsa"])].copy()
    method_df["is_wrsa"] = (method_df["eval_type"] == "wrsa").astype(int)

    result = run_fixed_effects(
        method_df,
        "test_score ~ is_wrsa + C(subject)",
        reference_info="cRSA"
    )
    beta = result["coefficients"]["is_wrsa"]
    print(f"  All models: beta={beta['beta']:.4f} [{beta['ci_lo']:.4f}, {beta['ci_hi']:.4f}], p={beta['p']:.4g}")
    all_results["linking_method"] = result

    # ================================================================
    # 8. Segmented Regression (Rank Breakpoint)
    # ================================================================
    print("\n--- 8. Segmented Regression ---")

    wrsa_means = (
        scores[scores["eval_type"] == "wrsa"]
        .groupby("model")["test_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    wrsa_means["rank"] = np.arange(1, len(wrsa_means) + 1)

    # Fit piecewise linear regression
    from scipy.optimize import minimize_scalar

    def piecewise_sse(breakpoint, x, y):
        bp = int(breakpoint)
        if bp < 10 or bp > len(x) - 10:
            return np.inf
        # Two separate linear regressions
        x1, y1 = x[:bp], y[:bp]
        x2, y2 = x[bp:], y[bp:]
        slope1, intercept1, _, _, _ = sp_stats.linregress(x1, y1)
        slope2, intercept2, _, _, _ = sp_stats.linregress(x2, y2)
        pred1 = intercept1 + slope1 * x1
        pred2 = intercept2 + slope2 * x2
        return np.sum((y1 - pred1) ** 2) + np.sum((y2 - pred2) ** 2)

    x = wrsa_means["rank"].values.astype(float)
    y = wrsa_means["test_score"].values

    result = minimize_scalar(
        piecewise_sse, bounds=(20, len(x) - 20),
        method="bounded", args=(x, y)
    )
    bp = int(result.x)
    bp_score = y[bp - 1]

    # Count models within 0.1 of top
    top_score = y[0]
    n_within_01 = int(np.sum(y >= top_score - 0.1))

    seg_results = {
        "breakpoint_rank": bp,
        "breakpoint_score": round(float(bp_score), 4),
        "n_models_above_breakpoint": bp,
        "top_score": round(float(top_score), 4),
        "n_within_0.1_of_top": n_within_01,
        "total_models": len(wrsa_means),
    }
    print(f"  Breakpoint at rank {bp} (score={bp_score:.4f})")
    print(f"  {n_within_01} models within 0.1 of top score ({top_score:.4f})")
    all_results["segmented_regression"] = seg_results

    # ================================================================
    # 9. Model-to-Model Similarity
    # ================================================================
    print("\n--- 9. Model-to-Model Similarity ---")

    raw_df = pd.read_parquet(RESULTS_DIR / "results_all.parquet")
    raw_df = raw_df[raw_df["region"] == "hlvis"]

    m2m_results = {}
    for et in ["crsa", "wrsa"]:
        # Build model profiles: for each model, get best-layer test scores across subjects
        best_layers = scores[scores["eval_type"] == et][["model", "best_layer"]].drop_duplicates()

        # For each model, get the score profile across subjects
        profiles = []
        model_names = []
        for _, row in best_layers.iterrows():
            model_data = scores[
                (scores["model"] == row["model"]) &
                (scores["eval_type"] == et)
            ].sort_values("subject")
            if len(model_data) >= 3:
                profiles.append(model_data["test_score"].values)
                model_names.append(row["model"])

        profile_matrix = np.array(profiles)
        corr_matrix = np.corrcoef(profile_matrix)
        n = len(model_names)
        upper_tri = corr_matrix[np.triu_indices(n, k=1)]

        m2m_results[et] = {
            "n_models": n,
            "n_pairs": len(upper_tri),
            "mean_similarity": round(float(upper_tri.mean()), 4),
            "std_similarity": round(float(upper_tri.std()), 4),
            "min_similarity": round(float(upper_tri.min()), 4),
            "max_similarity": round(float(upper_tri.max()), 4),
        }
        print(f"  {et.upper()}: mean={upper_tri.mean():.4f}, std={upper_tri.std():.4f}, "
              f"range=[{upper_tri.min():.4f}, {upper_tri.max():.4f}]")

    all_results["model_to_model"] = m2m_results

    # ================================================================
    # 10. Correlation: cRSA vs WRSA
    # ================================================================
    print("\n--- 10. cRSA vs WRSA Correlation ---")

    crsa_means = scores[scores["eval_type"] == "crsa"].groupby("model")["test_score"].mean()
    wrsa_means_s = scores[scores["eval_type"] == "wrsa"].groupby("model")["test_score"].mean()
    common = crsa_means.index.intersection(wrsa_means_s.index)

    rho, p = sp_stats.spearmanr(crsa_means[common], wrsa_means_s[common])
    r_p, p_p = sp_stats.pearsonr(crsa_means[common], wrsa_means_s[common])

    corr_results = {
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(f"{p:.6g}"),
        "pearson_r": round(float(r_p), 4),
        "pearson_p": float(f"{p_p:.6g}"),
        "n_models": len(common),
    }
    print(f"  Spearman rho={rho:.4f}, p={p:.4g}")
    print(f"  Pearson r={r_p:.4f}, p={p_p:.4g}")
    all_results["crsa_vs_wrsa_correlation"] = corr_results

    # ================================================================
    # 11. Effect Size Hierarchy
    # ================================================================
    print("\n--- 11. Effect Size Hierarchy ---")

    hierarchy = []

    # Architecture
    for et in ["crsa", "wrsa"]:
        r = all_results["architecture"][et]
        b = r["coefficients"]["is_transformer"]
        hierarchy.append({
            "comparison": "Architecture (Transformer vs CNN)",
            "eval_type": et,
            "beta": b["beta"],
            "p": b["p"],
        })

    # SSL
    for et in ["crsa", "wrsa"]:
        r = all_results["ssl_contrastive"][et]
        b = r["coefficients"]["is_contrastive"]
        hierarchy.append({
            "comparison": "SSL (Contrastive vs Non-Contrastive)",
            "eval_type": et,
            "beta": b["beta"],
            "p": b["p"],
        })

    # IPCL VGGFace2 (largest diet effect)
    for et in ["crsa", "wrsa"]:
        r = all_results["ipcl_domains"][et]
        face_key = [k for k in r["coefficients"] if "vggface2" in k]
        if face_key:
            b = r["coefficients"][face_key[0]]
            hierarchy.append({
                "comparison": "Diet (Faces vs Objects)",
                "eval_type": et,
                "beta": b["beta"],
                "p": b["p"],
            })

    # Linking method
    b = all_results["linking_method"]["coefficients"]["is_wrsa"]
    hierarchy.append({
        "comparison": "Linking Method (WRSA vs cRSA)",
        "eval_type": "both",
        "beta": b["beta"],
        "p": b["p"],
    })

    hierarchy_df = pd.DataFrame(hierarchy)
    hierarchy_df["abs_beta"] = hierarchy_df["beta"].abs()
    hierarchy_df = hierarchy_df.sort_values("abs_beta", ascending=False)
    print(hierarchy_df[["comparison", "eval_type", "beta", "p"]].to_string(index=False))

    all_results["effect_hierarchy"] = hierarchy

    # ================================================================
    # 12. Noise Ceiling Summary
    # ================================================================
    nc_summary = {
        "subjects": nc.to_dict("records"),
        "mean_nc_pearson": round(float(nc["nc_pearson"].mean()), 4),
        "mean_nc_spearman": round(float(nc["nc_spearman"].mean()), 4),
    }
    all_results["noise_ceiling"] = nc_summary

    # ================================================================
    # SAVE
    # ================================================================
    with open(RESULTS_DIR / "statistical_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {RESULTS_DIR / 'statistical_results.json'}")


def cli(argv=None):
    """CLI wrapper: sets RESULTS_DIR from --results-dir then calls main()."""
    global RESULTS_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Directory containing best_layer_scores.csv + "
                         "noise_ceilings.csv. statistical_results.json is "
                         "written here too.")
    args = ap.parse_args(argv)
    RESULTS_DIR = args.results_dir
    main()
    return 0


if __name__ == "__main__":
    sys.exit(cli())
