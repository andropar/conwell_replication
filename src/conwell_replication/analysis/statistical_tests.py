#!/usr/bin/env python3
"""Conwell-style statistical tests for evaluated model score tables.

This module consumes the score tables written by ``scripts/plot_splithalf_results.py``
(``model_metric_subject_scores.csv``) and runs the targeted comparisons described
in Conwell et al. using Python equivalents of the reported formulas:

  * OLS fixed effects: ``Score ~ manipulation + SubjectID``
  * Mixed effects for ImageNet1K vs ImageNet21K:
    ``Score ~ DatasetSize + SubjectID + (1 | ModelID)``

Model-to-model RSA is intentionally not implemented here; it requires saving
best-layer model RSM vectors during evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_CONTRASTS = REPO_ROOT / "resources" / "model_contrasts.csv"
DEFAULT_SCORE_FILES = (
    "model_metric_subject_scores.csv",
    "best_layer_scores.csv",
)
DEFAULT_METRICS = ("crsa", "wrsa")
DROP_MODELS = {"efficientnet_b1_classification"}


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_id: str
    label: str
    column: str
    reference: str
    levels: tuple[str, ...] | None = None
    model: str = "fixed"
    random_group_col: str | None = None


COMPARISONS = (
    ComparisonSpec(
        "architecture",
        "CNNs versus Transformers",
        "compare_architecture",
        "Convolutional",
        levels=("Convolutional", "Transformer"),
    ),
    ComparisonSpec(
        "taskonomy_tasks",
        "Taskonomy Encoders",
        "compare_goal_taskonomy_tasks",
        "denoising",
    ),
    ComparisonSpec(
        "contrastive_self_supervised",
        "Contrastive Self-Supervised Learning",
        "compare_goal_contrastive",
        "Non-Contrastive",
        levels=("Non-Contrastive", "Contrastive"),
    ),
    ComparisonSpec(
        "language_alignment",
        "Language Alignment",
        "compare_goal_slip",
        "SimCLR",
        levels=("SimCLR", "CLIP", "SLIP"),
    ),
    ComparisonSpec(
        "imagenet_size",
        "ImageNet1K versus ImageNet21K",
        "compare_diet_imagenetsize",
        "imagenet",
        levels=("imagenet", "imagenet21k"),
        model="mixed",
        random_group_col="architecture",
    ),
    ComparisonSpec(
        "objects_faces_places",
        "Objects, Faces, Places",
        "compare_diet_ipcl",
        "imagenet",
        levels=("imagenet", "openimages", "places256", "vggface2"),
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing model_metric_subject_scores.csv. If --scores is "
            "not given, this is required."
        ),
    )
    ap.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Explicit path to a score CSV. Defaults to model_metric_subject_scores.csv under --results-dir.",
    )
    ap.add_argument(
        "--raw-results-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing raw eval parquet shards with train/test layer "
            "scores. Use with --region or --all-regions to compute best-layer "
            "score tables before running stats."
        ),
    )
    ap.add_argument(
        "--all-regions",
        action="store_true",
        help=(
            "Run stats separately for every region in the score/raw table and "
            "write combined *_by_roi outputs."
        ),
    )
    ap.add_argument(
        "--filter-pool",
        default=None,
        help="Optional raw-results filter for min-nn outputs.",
    )
    ap.add_argument(
        "--filter-split",
        default=None,
        help="Optional raw-results filter for min-nn outputs.",
    )
    ap.add_argument(
        "--filter-ood-type",
        default=None,
        help=(
            "Optional raw-results filter for min-nn OOD outputs. Comma-separated; "
            "token 'none' keeps null ood_type rows."
        ),
    )
    ap.add_argument(
        "--model-contrasts",
        type=Path,
        default=DEFAULT_MODEL_CONTRASTS,
        help=f"Conwell contrast metadata CSV. Default: {DEFAULT_MODEL_CONTRASTS}",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write stats outputs. Default: --results-dir/statistics, or the score file directory.",
    )
    ap.add_argument(
        "--region",
        default=None,
        help="Optional region filter if the score table contains multiple regions.",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Eval metrics to test. Default: crsa wrsa.",
    )
    ap.add_argument(
        "--score-column",
        default="test_score",
        help="Score column to use as outcome. Default: test_score.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Confidence interval alpha. Default: 0.05.",
    )
    ap.add_argument(
        "--breakpoint-metric",
        default="wrsa",
        help="Metric used for rank breakpoint analysis. Default: wrsa.",
    )
    ap.add_argument(
        "--breakpoint-min-segment",
        type=int,
        default=10,
        help="Minimum models in each piecewise segment. Default: 10.",
    )
    return ap.parse_args(argv)


def unique_ordered(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        value = str(value)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def model_string_candidates(model_string: str) -> list[str]:
    raw = str(model_string)
    candidates = [raw]
    slash = raw.replace("-", "/")
    if slash != raw:
        candidates.append(slash)
    for candidate in list(candidates):
        if candidate.endswith("_ipcl"):
            candidates.append(candidate[: -len("_ipcl")])
    return unique_ordered(candidates)


def resolve_model_string(model_string: str, available_models: set[str]) -> str:
    candidates = model_string_candidates(model_string)
    for candidate in candidates:
        if candidate in available_models:
            return candidate
    transformed = [candidate for candidate in candidates if candidate != str(model_string)]
    return transformed[0] if transformed else candidates[0]


def find_score_file(args: argparse.Namespace) -> Path:
    if args.raw_results_dir is not None:
        raise ValueError("find_score_file should not be called for --raw-results-dir")
    if args.scores is not None:
        return args.scores
    if args.results_dir is None:
        raise SystemExit("Either --results-dir or --scores is required.")
    for name in DEFAULT_SCORE_FILES:
        path = args.results_dir / name
        if path.exists():
            return path
    names = ", ".join(DEFAULT_SCORE_FILES)
    raise FileNotFoundError(f"No score file found under {args.results_dir}; looked for {names}")


def load_scores(path: Path, *, region: str | None, metrics: list[str], score_column: str) -> pd.DataFrame:
    scores = pd.read_csv(path)
    required = {"subject", "model", "eval_type", score_column}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if region is not None:
        if "region" not in scores.columns:
            raise ValueError("--region was supplied, but score table has no region column")
        scores = scores[scores["region"].astype(str) == str(region)].copy()
    scores = scores[scores["eval_type"].astype(str).isin(metrics)].copy()
    scores = scores.rename(columns={score_column: "score"})
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores = scores.dropna(subset=["score"]).reset_index(drop=True)
    scores["subject"] = scores["subject"].astype(str)
    scores["model"] = scores["model"].astype(str)
    scores["eval_type"] = scores["eval_type"].astype(str)
    if DROP_MODELS:
        scores = scores[~scores["model"].isin(DROP_MODELS)].reset_index(drop=True)
    return scores


def filter_optional_columns(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    for col, value in (("pool", args.filter_pool), ("split", args.filter_split)):
        if value is None:
            continue
        if col not in df.columns:
            raise ValueError(f"--filter-{col} was supplied, but raw table has no {col!r} column")
        df = df[df[col].astype(str) == str(value)].copy()
    if args.filter_ood_type is not None:
        if "ood_type" not in df.columns:
            raise ValueError("--filter-ood-type was supplied, but raw table has no 'ood_type' column")
        tokens = [token.strip() for token in str(args.filter_ood_type).split(",") if token.strip()]
        keep_null = any(token.lower() == "none" for token in tokens)
        explicit = [token for token in tokens if token.lower() != "none"]
        mask = pd.Series(False, index=df.index)
        if keep_null:
            mask |= df["ood_type"].isna()
        if explicit:
            mask |= df["ood_type"].astype(str).isin(explicit)
        df = df[mask].copy()
    return df.reset_index(drop=True)


def load_raw_results(args: argparse.Namespace) -> pd.DataFrame:
    if args.raw_results_dir is None:
        raise ValueError("load_raw_results requires --raw-results-dir")
    files = sorted(args.raw_results_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {args.raw_results_dir}")
    frames = []
    for path in files:
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    required = {
        "score",
        "score_set",
        "eval_type",
        "region",
        "model",
        "model_layer",
        "model_layer_index",
        "subject",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Raw result shards are missing required columns: {sorted(missing)}")
    if args.region is not None:
        df = df[df["region"].astype(str) == str(args.region)].copy()
    df = filter_optional_columns(df, args)
    df = df[df["eval_type"].astype(str).isin(args.metrics)].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["model_layer_index"] = pd.to_numeric(df["model_layer_index"], errors="coerce").astype("Int64")
    for col in ["score_set", "eval_type", "region", "model", "model_layer", "subject"]:
        df[col] = df[col].astype(str)
    df = df.dropna(subset=["score", "model_layer_index"]).reset_index(drop=True)
    if DROP_MODELS:
        df = df[~df["model"].isin(DROP_MODELS)].reset_index(drop=True)
    return df


def best_layer_extra_keys(raw: pd.DataFrame) -> list[str]:
    keys = []
    for col in ["pool", "split", "variant", "ood_type"]:
        if col in raw.columns and raw[col].nunique(dropna=False) > 1:
            keys.append(col)
    return keys


def compute_best_layers_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    extra = best_layer_extra_keys(raw)
    keys = ["subject", "model", "eval_type", "region", *extra]
    layer_keys = [*keys, "model_layer", "model_layer_index"]
    score_keys = [*layer_keys, "score_set"]
    compact = raw[score_keys + ["score"]].copy()
    if compact.duplicated(score_keys).any():
        compact = (
            compact.groupby(score_keys, observed=True, dropna=False)["score"]
            .mean()
            .reset_index()
        )
    wide = compact.set_index(score_keys)["score"].unstack("score_set").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"train": "train_score", "test": "test_score"})
    for col in ("train_score", "test_score"):
        if col not in wide.columns:
            wide[col] = np.nan
    finite_train = wide.dropna(subset=["train_score"])
    if finite_train.empty:
        raise ValueError("No train scores available after filtering; cannot select best layers")
    idx = finite_train.groupby(keys, observed=True, dropna=False)["train_score"].idxmax()
    best = finite_train.loc[idx].copy().reset_index(drop=True)
    max_layer = (
        wide.groupby([k for k in keys if k in wide.columns], observed=True, dropna=False)["model_layer_index"]
        .max()
        .rename("max_model_layer_index")
        .reset_index()
    )
    best = best.merge(max_layer, on=keys, how="left")
    best["selected_layer_fraction"] = (
        best["model_layer_index"].astype(float)
        / best["max_model_layer_index"].replace({0: np.nan}).astype(float)
    )
    best["score"] = pd.to_numeric(best["test_score"], errors="coerce")
    return best.sort_values(keys).reset_index(drop=True)


def load_model_contrasts(path: Path, available_models: set[str]) -> pd.DataFrame:
    contrasts = pd.read_csv(path).rename(columns={"model": "deepnsd_model"})
    required = {"model_string", "model_display_name"}
    missing = required.difference(contrasts.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    contrasts["result_model"] = contrasts["model_string"].map(
        lambda model: resolve_model_string(model, available_models)
    )
    contrasts["result_model_in_scores"] = contrasts["result_model"].isin(available_models)
    contrasts["model_display_name"] = contrasts["model_display_name"].fillna(contrasts["result_model"])
    return contrasts


def merge_contrasts(scores: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "result_model",
        "deepnsd_model",
        "model_string",
        "model_display_name",
        "train_type",
        "train_data",
        "architecture",
        "model_class",
        "task_cluster",
        "compare_architecture",
        "compare_goal_slip",
        "compare_goal_taskonomy_tasks",
        "compare_goal_taskonomy_cluster",
        "compare_goal_expertise",
        "compare_diet_ipcl",
        "compare_goal_selfsupervised",
        "compare_goal_contrastive",
        "compare_diet_imagenetsize",
    ]
    keep = [col for col in keep if col in contrasts.columns]
    contrast_meta = (
        contrasts[contrasts["result_model_in_scores"]][keep]
        .drop_duplicates("result_model")
        .reset_index(drop=True)
    )
    conflicts = [col for col in keep if col in scores.columns and col != "model"]
    base = scores.drop(columns=conflicts, errors="ignore")
    return base.merge(contrast_meta, left_on="model", right_on="result_model", how="left")


def clean_group_values(series: pd.Series) -> pd.Series:
    values = series.astype("string")
    values = values.mask(values.str.len().fillna(0) == 0)
    return values


def comparison_subset(table: pd.DataFrame, spec: ComparisonSpec, metric: str) -> pd.DataFrame:
    if spec.column not in table.columns:
        return pd.DataFrame()
    sub = table[table["eval_type"] == metric].copy()
    sub[spec.column] = clean_group_values(sub[spec.column])
    sub = sub[sub[spec.column].notna()].copy()
    if spec.levels is not None:
        sub = sub[sub[spec.column].isin(spec.levels)].copy()
    if sub.empty or spec.reference not in set(sub[spec.column]):
        return pd.DataFrame()
    levels = list(spec.levels) if spec.levels is not None else unique_ordered([spec.reference, *sub[spec.column]])
    if spec.reference in levels:
        levels = [spec.reference] + [level for level in levels if level != spec.reference]
    sub[spec.column] = pd.Categorical(sub[spec.column], categories=levels)
    return sub


def factor_formula(factor: str, reference: str) -> str:
    return f"C({factor}, Treatment(reference={reference!r}))"


def parse_factor_level(term: str) -> str:
    match = re.search(r"\[T\.(.*)\]$", term)
    if match:
        return match.group(1)
    return term


def result_rows_from_fit(
    *,
    fit: Any,
    params: pd.Series,
    bse: pd.Series,
    pvalues: pd.Series,
    conf_int: pd.DataFrame,
    spec: ComparisonSpec,
    metric: str,
    df: pd.DataFrame,
    formula: str,
    model_type: str,
    statistic_name: str,
    random_group_col: str | None = None,
) -> list[dict[str, Any]]:
    factor_prefix = f"C({spec.column},"
    rows = []
    for term in params.index:
        if term == "Intercept" or term.startswith("C(subject)"):
            continue
        if not term.startswith(factor_prefix):
            continue
        ci_lo, ci_hi = conf_int.loc[term]
        rows.append(
            {
                "comparison_id": spec.comparison_id,
                "comparison": spec.label,
                "eval_type": metric,
                "model_type": model_type,
                "formula": formula,
                "reference": spec.reference,
                "term": term,
                "level": parse_factor_level(term),
                "beta": float(params[term]),
                "se": float(bse[term]),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "statistic_name": statistic_name,
                "statistic": float(getattr(fit, "tvalues", pd.Series(dtype=float)).loc[term]),
                "p_value": float(pvalues[term]),
                "n_obs": int(getattr(fit, "nobs", len(df))),
                "n_subjects": int(df["subject"].nunique()),
                "n_models": int(df["model"].nunique()),
                "random_group_col": random_group_col,
                "n_random_groups": (
                    int(df[random_group_col].nunique())
                    if random_group_col is not None and random_group_col in df.columns
                    else np.nan
                ),
                "r_squared": float(getattr(fit, "rsquared", np.nan)),
                "converged": bool(getattr(fit, "converged", True)),
            }
        )
    return rows


def run_fixed_effects(df: pd.DataFrame, spec: ComparisonSpec, metric: str, alpha: float) -> list[dict[str, Any]]:
    fterm = factor_formula(spec.column, spec.reference)
    formula = f"score ~ {fterm} + C(subject)"
    fit = smf.ols(formula, data=df).fit()
    conf = fit.conf_int(alpha=alpha)
    return result_rows_from_fit(
        fit=fit,
        params=fit.params,
        bse=fit.bse,
        pvalues=fit.pvalues,
        conf_int=conf,
        spec=spec,
        metric=metric,
        df=df,
        formula=formula,
        model_type="ols_fixed_effects",
        statistic_name="t",
    )


def run_mixed_effects(df: pd.DataFrame, spec: ComparisonSpec, metric: str, alpha: float) -> list[dict[str, Any]]:
    group_col = spec.random_group_col
    if group_col is None or group_col not in df.columns:
        return run_fixed_effects(df, spec, metric, alpha)
    df = df.dropna(subset=[group_col]).copy()
    if df.empty:
        return []
    fterm = factor_formula(spec.column, spec.reference)
    formula = f"score ~ {fterm} + C(subject)"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(formula, data=df, groups=df[group_col]).fit(
                reml=True,
                method="lbfgs",
                maxiter=1000,
                disp=False,
            )
        conf = fit.conf_int(alpha=alpha)
        return result_rows_from_fit(
            fit=fit,
            params=fit.fe_params,
            bse=fit.bse_fe,
            pvalues=fit.pvalues,
            conf_int=conf,
            spec=spec,
            metric=metric,
            df=df,
            formula=formula + f" + (1 | {group_col})",
            model_type="mixedlm_random_intercept",
            statistic_name="z",
            random_group_col=group_col,
        )
    except Exception as exc:
        fallback_formula = f"score ~ {fterm} + C(subject) + C({group_col})"
        fit = smf.ols(fallback_formula, data=df).fit()
        conf = fit.conf_int(alpha=alpha)
        rows = result_rows_from_fit(
            fit=fit,
            params=fit.params,
            bse=fit.bse,
            pvalues=fit.pvalues,
            conf_int=conf,
            spec=spec,
            metric=metric,
            df=df,
            formula=fallback_formula,
            model_type="ols_fixed_effects_fallback",
            statistic_name="t",
            random_group_col=group_col,
        )
        for row in rows:
            row["fallback_reason"] = str(exc)
        return rows


def describe_groups(table: pd.DataFrame, specs: tuple[ComparisonSpec, ...], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for metric in metrics:
            sub = comparison_subset(table, spec, metric)
            if sub.empty:
                continue
            for level, group in sub.groupby(spec.column, observed=True):
                vals = group["score"].dropna()
                rows.append(
                    {
                        "comparison_id": spec.comparison_id,
                        "comparison": spec.label,
                        "eval_type": metric,
                        "level": str(level),
                        "n_obs": int(len(vals)),
                        "n_subjects": int(group["subject"].nunique()),
                        "n_models": int(group["model"].nunique()),
                        "mean": float(vals.mean()),
                        "median": float(vals.median()),
                        "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                        "sem": float(vals.sem(ddof=1)) if len(vals) > 1 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_coverage(contrasts: pd.DataFrame, specs: tuple[ComparisonSpec, ...]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        if spec.column not in contrasts.columns:
            continue
        sub = contrasts.copy()
        sub[spec.column] = clean_group_values(sub[spec.column])
        sub = sub[sub[spec.column].notna()]
        if spec.levels is not None:
            sub = sub[sub[spec.column].isin(spec.levels)]
        for level, group in sub.groupby(spec.column, observed=True):
            present = group[group["result_model_in_scores"]]
            missing = group[~group["result_model_in_scores"]]["model_string"].astype(str).tolist()
            rows.append(
                {
                    "comparison_id": spec.comparison_id,
                    "comparison": spec.label,
                    "level": str(level),
                    "models_in_contrast": int(group["model_string"].nunique()),
                    "models_in_scores": int(present["result_model"].nunique()),
                    "missing_models": "; ".join(missing[:12]),
                    "n_missing": int(len(missing)),
                }
            )
    return pd.DataFrame(rows)


def fit_line_sse(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.inf
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    return float(np.sum((y - pred) ** 2))


def two_breakpoint_fit(x: np.ndarray, y: np.ndarray, min_segment: int) -> dict[str, Any]:
    n = len(x)
    if n < min_segment * 3:
        return {"error": f"Need at least {min_segment * 3} models for 2-breakpoint fit; got {n}"}
    best: tuple[float, int, int] | None = None
    for bp1 in range(min_segment, n - 2 * min_segment + 1):
        for bp2 in range(bp1 + min_segment, n - min_segment + 1):
            sse = (
                fit_line_sse(x[:bp1], y[:bp1])
                + fit_line_sse(x[bp1:bp2], y[bp1:bp2])
                + fit_line_sse(x[bp2:], y[bp2:])
            )
            if best is None or sse < best[0]:
                best = (sse, bp1, bp2)
    assert best is not None
    sse, bp1, bp2 = best
    return {
        "breakpoint_1_rank": int(bp1),
        "breakpoint_1_score": float(y[bp1 - 1]),
        "breakpoint_2_rank": int(bp2),
        "breakpoint_2_score": float(y[bp2 - 1]),
        "n_models_top_segment": int(bp1),
        "n_models_middle_segment": int(bp2 - bp1),
        "n_models_bottom_segment": int(n - bp2),
        "sse": float(sse),
    }


def run_breakpoint(table: pd.DataFrame, metric: str, min_segment: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = table[table["eval_type"] == metric].copy()
    means = (
        sub.groupby("model", observed=True)["score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    means["rank"] = np.arange(1, len(means) + 1)
    if means.empty:
        return pd.DataFrame(), pd.DataFrame([{"metric": metric, "error": "No rows"}])
    fit = two_breakpoint_fit(means["rank"].to_numpy(float), means["score"].to_numpy(float), min_segment)
    fit.update(
        {
            "metric": metric,
            "n_models": int(len(means)),
            "top_model": means["model"].iloc[0],
            "top_score": float(means["score"].iloc[0]),
            "method": "pure_python_grid_search_two_breakpoints",
            "ci_method": "not_computed",
        }
    )
    return means, pd.DataFrame([fit])


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if pd.isna(value) if not isinstance(value, (str, bytes, list, dict, tuple)) else False:
        return None
    return value


def write_markdown(path: Path, tests: pd.DataFrame, breakpoint: pd.DataFrame) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        text = df.copy().astype(str)
        headers = list(text.columns)
        rows = text.values.tolist()
        out = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            out.append("| " + " | ".join(row) + " |")
        return "\n".join(out)

    lines = [
        "# Conwell-Style Statistical Tests",
        "",
        "Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.",
        "",
    ]
    if not tests.empty:
        display = tests[
            [
                "comparison_id",
                "eval_type",
                "level",
                "beta",
                "ci_lo",
                "ci_hi",
                "statistic_name",
                "statistic",
                "p_value",
                "model_type",
                "n_obs",
            ]
        ].copy()
        for col in ["beta", "ci_lo", "ci_hi", "statistic", "p_value"]:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        lines.extend(["## Targeted Comparisons", "", markdown_table(display), ""])
    if not breakpoint.empty:
        lines.extend(["## Breakpoint", "", markdown_table(breakpoint), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stats_for_scores(
    scores: pd.DataFrame,
    *,
    contrasts: pd.DataFrame,
    metrics: list[str],
    alpha: float,
    breakpoint_metric: str,
    breakpoint_min_segment: int,
) -> dict[str, pd.DataFrame]:
    table = merge_contrasts(scores, contrasts)
    coverage = build_coverage(contrasts, COMPARISONS)
    descriptives = describe_groups(table, COMPARISONS, metrics)

    test_rows: list[dict[str, Any]] = []
    for spec in COMPARISONS:
        for metric in metrics:
            sub = comparison_subset(table, spec, metric)
            if sub.empty:
                continue
            if spec.model == "mixed":
                rows = run_mixed_effects(sub, spec, metric, alpha)
            else:
                rows = run_fixed_effects(sub, spec, metric, alpha)
            test_rows.extend(rows)
    tests = pd.DataFrame(test_rows)
    rankings, breakpoint = run_breakpoint(table, breakpoint_metric, breakpoint_min_segment)
    return {
        "stats_input_table": table,
        "model_coverage": coverage,
        "comparison_descriptives": descriptives,
        "statistical_tests": tests,
        "model_rankings_for_breakpoint": rankings,
        "breakpoint_analysis": breakpoint,
    }


def write_single_outputs(
    *,
    out_dir: Path,
    score_source: str,
    args: argparse.Namespace,
    outputs: dict[str, pd.DataFrame],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = outputs["stats_input_table"]
    coverage = outputs["model_coverage"]
    descriptives = outputs["comparison_descriptives"]
    tests = outputs["statistical_tests"]
    rankings = outputs["model_rankings_for_breakpoint"]
    breakpoint = outputs["breakpoint_analysis"]

    table.to_csv(out_dir / "stats_input_table.csv", index=False)
    coverage.to_csv(out_dir / "model_coverage.csv", index=False)
    descriptives.to_csv(out_dir / "comparison_descriptives.csv", index=False)
    tests.to_csv(out_dir / "statistical_tests.csv", index=False)
    rankings.to_csv(out_dir / "model_rankings_for_breakpoint.csv", index=False)
    breakpoint.to_csv(out_dir / "breakpoint_analysis.csv", index=False)

    payload = {
        "score_source": score_source,
        "model_contrasts": str(args.model_contrasts),
        "region": args.region,
        "metrics": args.metrics,
        "score_column": args.score_column,
        "notes": [
            "Subject is modeled as a fixed effect.",
            "ImageNet1K vs ImageNet21K uses a random intercept for contrast-file architecture/model ID.",
            "Breakpoint analysis is a pure-Python two-breakpoint grid-search approximation to R segmented.",
            "Model-to-model RSA is postponed until best-layer RSM vectors are saved by the eval pipeline.",
        ],
        "tests": tests.to_dict("records"),
        "descriptives": descriptives.to_dict("records"),
        "breakpoint": breakpoint.to_dict("records"),
        "coverage": coverage.to_dict("records"),
    }
    with (out_dir / "statistical_results.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)
    write_markdown(out_dir / "statistical_tests.md", tests, breakpoint)

    print(f"Score rows used: {len(table):,}")
    print(f"Models used: {table['model'].nunique():,}")
    print(f"Subjects used: {table['subject'].nunique():,}")
    print(f"Tests written: {len(tests):,}")
    print(f"Outputs written to {out_dir}")
    if not tests.empty:
        print()
        print(
            tests[
                [
                    "comparison_id",
                    "eval_type",
                    "level",
                    "beta",
                    "ci_lo",
                    "ci_hi",
                    "statistic_name",
                    "statistic",
                    "p_value",
                    "model_type",
                ]
            ].to_string(index=False, max_colwidth=28)
        )


def write_roi_outputs(
    *,
    out_dir: Path,
    score_source: str,
    args: argparse.Namespace,
    scores_by_roi: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_by_roi.to_csv(out_dir / "best_layer_scores_by_roi.csv", index=False)
    coverage = build_coverage(contrasts, COMPARISONS)
    coverage.to_csv(out_dir / "model_coverage.csv", index=False)

    combined: dict[str, list[pd.DataFrame]] = {
        "stats_input_table": [],
        "comparison_descriptives": [],
        "statistical_tests": [],
        "model_rankings_for_breakpoint": [],
        "breakpoint_analysis": [],
    }
    for region, region_scores in scores_by_roi.groupby("region", observed=True):
        outputs = run_stats_for_scores(
            region_scores,
            contrasts=contrasts,
            metrics=args.metrics,
            alpha=args.alpha,
            breakpoint_metric=args.breakpoint_metric,
            breakpoint_min_segment=args.breakpoint_min_segment,
        )
        for name, frame in outputs.items():
            if name == "model_coverage" or frame.empty:
                continue
            out = frame.copy()
            if "region" in out.columns:
                out["region"] = region
                cols = ["region", *[col for col in out.columns if col != "region"]]
                out = out[cols]
            else:
                out.insert(0, "region", region)
            combined[name].append(out)

    written: dict[str, pd.DataFrame] = {}
    for name, frames in combined.items():
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        written[name] = frame
        frame.to_csv(out_dir / f"{name}_by_roi.csv", index=False)

    payload = {
        "score_source": score_source,
        "model_contrasts": str(args.model_contrasts),
        "regions": sorted(scores_by_roi["region"].astype(str).unique().tolist()),
        "metrics": args.metrics,
        "score_column": args.score_column,
        "notes": [
            "Each ROI is analyzed independently, including best-layer selection.",
            "Subject is modeled as a fixed effect.",
            "Model-to-model RSA is postponed until best-layer RSM vectors are saved by the eval pipeline.",
        ],
        "tests": written["statistical_tests"].to_dict("records"),
        "descriptives": written["comparison_descriptives"].to_dict("records"),
        "breakpoint": written["breakpoint_analysis"].to_dict("records"),
        "coverage": coverage.to_dict("records"),
    }
    with (out_dir / "statistical_results_by_roi.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)
    print(f"ROI score rows used: {len(scores_by_roi):,}")
    print(f"Regions used: {scores_by_roi['region'].nunique():,}")
    print(f"Models used: {scores_by_roi['model'].nunique():,}")
    print(f"Subjects used: {scores_by_roi['subject'].nunique():,}")
    print(f"Tests written: {len(written['statistical_tests']):,}")
    print(f"Outputs written to {out_dir}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.raw_results_dir is not None:
        if args.scores is not None:
            raise SystemExit("--scores and --raw-results-dir are mutually exclusive")
        if not args.all_regions and args.region is None:
            raise SystemExit("--raw-results-dir requires either --region or --all-regions")
        out_dir = args.out_dir or (args.raw_results_dir / "statistics")
        raw = load_raw_results(args)
        scores = compute_best_layers_from_raw(raw)
        contrasts = load_model_contrasts(args.model_contrasts, set(scores["model"]))
        source = str(args.raw_results_dir)
        if args.all_regions:
            write_roi_outputs(
                out_dir=out_dir,
                score_source=source,
                args=args,
                scores_by_roi=scores,
                contrasts=contrasts,
            )
        else:
            outputs = run_stats_for_scores(
                scores,
                contrasts=contrasts,
                metrics=args.metrics,
                alpha=args.alpha,
                breakpoint_metric=args.breakpoint_metric,
                breakpoint_min_segment=args.breakpoint_min_segment,
            )
            write_single_outputs(out_dir=out_dir, score_source=source, args=args, outputs=outputs)
        return 0

    score_path = find_score_file(args)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (args.results_dir / "statistics") if args.results_dir is not None else (score_path.parent / "statistics")
    scores = load_scores(score_path, region=args.region, metrics=args.metrics, score_column=args.score_column)
    contrasts = load_model_contrasts(args.model_contrasts, set(scores["model"]))
    if args.all_regions:
        if "region" not in scores.columns:
            raise SystemExit("--all-regions requires a score table with a region column")
        write_roi_outputs(
            out_dir=out_dir,
            score_source=str(score_path),
            args=args,
            scores_by_roi=scores,
            contrasts=contrasts,
        )
    else:
        outputs = run_stats_for_scores(
            scores,
            contrasts=contrasts,
            metrics=args.metrics,
            alpha=args.alpha,
            breakpoint_metric=args.breakpoint_metric,
            breakpoint_min_segment=args.breakpoint_min_segment,
        )
        write_single_outputs(out_dir=out_dir, score_source=str(score_path), args=args, outputs=outputs)
    return 0


def cli(argv: list[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    sys.exit(cli())
