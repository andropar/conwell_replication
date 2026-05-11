#!/usr/bin/env python3
"""Conwell-style controlled-comparison plots for split-half results.

Reads completed parquet shards, selects each subject/model/metric best layer by
train score, and plots the test score from that same layer. The output figures
mirror the geometry of Conwell et al. Figs. 2-4: model boxes, open cRSA boxes,
filled veRSA/wRSA boxes, group-mean ribbons, and a noise-ceiling reference when
available.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Iterable

import colorsys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
import numpy as np
import pandas as pd


def _saturate(hex_color: str, factor: float = 1.0) -> str:
    """Multiply HLS saturation by ``factor`` (clamped to 1.0)."""
    r, g, b = matplotlib.colors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = min(1.0, s * factor)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return matplotlib.colors.to_hex((r, g, b))


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = Path(os.environ.get("CONWELL_SPLITHALF_RESULTS_DIR", REPO_ROOT / "results" / "splithalf"))
DEFAULT_OUT_DIR = REPO_ROOT / "figures" / "splithalf_results"
DEFAULT_METADATA = REPO_ROOT / "resources" / "model_metadata.csv"
DEFAULT_MODEL_CONTRASTS = REPO_ROOT / "resources" / "model_contrasts.csv"
DEFAULT_NOISE_CEILING = Path(
    os.environ.get(
        "CONWELL_NOISE_CEILING_CSV",
        REPO_ROOT / "results" / "noise_ceiling" / "noise_ceiling.csv",
    )
)

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
METRICS = ["crsa", "wrsa", "srpr"]
PLOT_METRICS = ["crsa", "wrsa"]
DROP_MODELS: set[str] = set()

# Palette extracted pixel-by-pixel from conwell_figures/fig{2,3,4}.png. Each
# value is the median hue sampled from the colored italic group labels or
# filled veRSA boxes in the published figures. Saturation is boosted on top of
# the raw hues so the colors read more vividly on screen than the muted
# publication PNGs.
_RAW_CONWELL = {
    "cnn_salmon":      "#d6859a",  # fig2 CNN label + box (rose salmon)
    "transformer":     "#cb7c62",  # fig2 Transformer box (peachy red)
    "taskonomy_olive": "#c79e5b",  # fig3a Taskonomy label
    "ssl_olive":       "#a4982f",  # fig3b SSL label
    "simclr_pink":     "#d6859a",  # fig3c SimCLR — same family as CNN salmon
    "clip_peach":      "#cb7c62",  # fig3c CLIP — same as Transformer peach
    "slip_green":      "#4faa7a",  # fig3c SLIP teal-green
    "in_blue":         "#6ba0e1",  # fig4a ImageNet-1K AND ImageNet-21K (same)
    "ipcl_imagenet":   "#9f96e4",  # fig4b IPCL Imagenet purple
    "ipcl_openimages": "#ca6bc5",  # fig4b IPCL OpenImages magenta
    "ipcl_places256":  "#d46da8",  # fig4b IPCL Places256 rose
    "ipcl_vggface2":   "#d68069",  # fig4b IPCL VGGFace2 peach (extrapolated)
}
_SAT_BOOST = 1.45
CONWELL = {k: _saturate(v, _SAT_BOOST) for k, v in _RAW_CONWELL.items()}
CONWELL.update({
    "gray_mid":   "#8a8a8a",
    "gray_dark":  "#3a3a3a",
    "noise_band": "#d5d5d5",
    "noise_line": "#7a7a7a",
})

COLORS = {
    "cnn": CONWELL["cnn_salmon"],
    "transformer": CONWELL["transformer"],
    "taskonomy": CONWELL["taskonomy_olive"],
    # Kept for legacy callers; Conwell collapses Taskonomy onto a single hue.
    "task_2d": CONWELL["taskonomy_olive"],
    "task_3d": CONWELL["taskonomy_olive"],
    "task_geometric": CONWELL["taskonomy_olive"],
    "task_semantic": CONWELL["taskonomy_olive"],
    "task_other": CONWELL["taskonomy_olive"],
    "task_random": CONWELL["taskonomy_olive"],
    "contrastive": CONWELL["ssl_olive"],
    "noncontrastive": CONWELL["ssl_olive"],
    "supervised": CONWELL["gray_mid"],
    "simclr": CONWELL["simclr_pink"],
    "clip": CONWELL["clip_peach"],
    "slip": CONWELL["slip_green"],
    "in1k": CONWELL["in_blue"],
    "in21k": CONWELL["in_blue"],
    "ipcl_imagenet": CONWELL["ipcl_imagenet"],
    "ipcl_openimages": CONWELL["ipcl_openimages"],
    "ipcl_places256": CONWELL["ipcl_places256"],
    "ipcl_vggface2": CONWELL["ipcl_vggface2"],
    "noise": CONWELL["noise_band"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing split-half parquet shards. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory inside the repo. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Model metadata CSV. Default: {DEFAULT_METADATA}",
    )
    parser.add_argument(
        "--model-contrasts",
        type=Path,
        default=DEFAULT_MODEL_CONTRASTS,
        help=(
            "DeepNSD model_contrasts.csv defining the controlled figures. "
            f"Default: {DEFAULT_MODEL_CONTRASTS}"
        ),
    )
    parser.add_argument(
        "--noise-ceiling",
        type=Path,
        default=DEFAULT_NOISE_CEILING,
        help=(
            "CSV with per-(subject, region, metric, score_set) noise ceilings, "
            "as produced by scripts/compute_noise_ceiling.py. "
            f"Default: {DEFAULT_NOISE_CEILING}"
        ),
    )
    parser.add_argument(
        "--region",
        default="otc",
        help=(
            "ROI to plot. Best-layer selection and NC are both filtered to this "
            "region. Default: otc"
        ),
    )
    parser.add_argument(
        "--nc-metric",
        default="rsa",
        choices=("rsa", "srpr"),
        help=(
            "Which NC family to draw on the panel (rsa for cRSA/WRSA panels, "
            "srpr if SRPR is plotted). Default: rsa"
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Figure formats to write. Default: png pdf",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Bootstrap resamples for grand-mean-centered CIs. Default: 10000",
    )
    parser.add_argument(
        "--from-stats",
        action="store_true",
        help=(
            "Skip parquet loading; redraw figures using the previously written "
            "model_metric_subject_scores.csv and noise_ceiling_summary.csv in "
            "--out-dir. Useful for restyling without rerunning the pipeline."
        ),
    )
    parser.add_argument(
        "--filter-split",
        default=None,
        help="If results contain a 'split' column (min-nn output), keep only this split.",
    )
    parser.add_argument(
        "--filter-pool",
        default=None,
        help="If results contain a 'pool' column (min-nn output), keep only this pool.",
    )
    parser.add_argument(
        "--filter-ood-type",
        default=None,
        help="If results contain an 'ood_type' column, keep only this ood_type "
             "(use 'none' to match rows where ood_type is null).",
    )
    return parser.parse_args()


def set_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: matplotlib.figure.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}")
    plt.close(fig)


def clear_old_figures(out_dir: Path) -> None:
    """Keep the output directory focused on the current controlled figures."""
    for pattern in ("*.png", "*.pdf"):
        for path in out_dir.glob(pattern):
            path.unlink()


def short_label(model: str, max_len: int = 32) -> str:
    label = str(model)
    replacements = {
        "BiT/Expert/ResNet/V2/": "BiT/",
        "_classification": "",
        "_selfsupervised": "",
        "_big_transfer": "",
        "torchvision_": "tv_",
        "vissl_resnet50_": "",
        "ResNet50/": "",
        "slip_": "",
        "_taskonomy": "",
        "alexnet_gn_ipcl_": "",
        "patch4_window7_224": "",
        "patch16_224": "p16",
        "patch32_224": "p32",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    label = label.replace("__", "_").strip("_")
    if len(label) > max_len:
        return label[: max_len - 3] + "..."
    return label


def load_results(results_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {results_dir}")
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frame["source_file"] = path.name
        frames.append(frame)
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
        raise ValueError(f"Result files are missing required columns: {sorted(missing)}")

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["model_layer_index"] = pd.to_numeric(df["model_layer_index"], errors="coerce").astype("Int64")
    for col in ["score_set", "eval_type", "region", "model", "model_layer", "subject"]:
        df[col] = df[col].astype(str)
    if DROP_MODELS:
        before = len(df)
        df = df[~df["model"].isin(DROP_MODELS)].reset_index(drop=True)
        if before != len(df):
            print(
                f"Dropped {before - len(df)} rows for excluded models "
                f"{sorted(DROP_MODELS)}",
                flush=True,
            )
    return df, files


def load_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    if "model" not in metadata.columns:
        raise ValueError(f"Metadata file lacks a model column: {path}")
    return metadata.drop_duplicates("model")


def unique_ordered(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if pd.isna(value):
            continue
        value = str(value)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def model_string_candidates(model_string: str) -> list[str]:
    """Feature stems replace dashes with slashes; IPCL stems also drop a suffix."""
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


def load_model_contrasts(path: Path, available_models: set[str]) -> pd.DataFrame:
    contrasts = pd.read_csv(path).rename(columns={"model": "deepnsd_model"})
    required = {"model_string", "model_display_name"}
    missing = required.difference(contrasts.columns)
    if missing:
        raise ValueError(f"Model contrast file lacks required columns: {sorted(missing)}")
    contrasts["result_model"] = contrasts["model_string"].map(
        lambda model: resolve_model_string(model, available_models)
    )
    contrasts["result_model_in_results"] = contrasts["result_model"].isin(available_models)
    contrasts["model_display_name"] = contrasts["model_display_name"].fillna(contrasts["result_model"])
    contrasts.attrs["source_path"] = str(path)
    return contrasts


def contrast_models(
    contrasts: pd.DataFrame,
    column: str,
    value: str | None = None,
) -> list[str]:
    if column not in contrasts.columns:
        return []
    sub = contrasts[contrasts[column].notna()]
    if value is not None:
        sub = sub[sub[column] == value]
    return unique_ordered(sub["result_model"])


def contrast_labeler(contrasts: pd.DataFrame, max_len: int = 28):
    labels = (
        contrasts[["result_model", "model_display_name"]]
        .dropna()
        .drop_duplicates("result_model")
        .set_index("result_model")["model_display_name"]
        .to_dict()
    )

    def label_func(model: str) -> str:
        return short_label(labels.get(model, model), max_len)

    return label_func


def comparison_model_sets(contrasts: pd.DataFrame) -> dict[str, list[str]]:
    supervised = [resolve_model_string("resnet50_classification", set(contrasts["result_model"]))]
    return {
        "Fig2 architecture CNN": contrast_models(contrasts, "compare_architecture", "Convolutional"),
        "Fig2 architecture Transformer": contrast_models(contrasts, "compare_architecture", "Transformer"),
        "Fig3A Taskonomy": contrast_models(contrasts, "compare_goal_taskonomy_tasks"),
        "Fig3B SSL plus supervised reference": (
            contrast_models(contrasts, "compare_goal_selfsupervised") + supervised
        ),
        "Fig3C SLIP": contrast_models(contrasts, "compare_goal_slip"),
        "Fig4A ImageNet input": contrast_models(contrasts, "compare_diet_imagenetsize"),
        "Fig4B IPCL": contrast_models(contrasts, "compare_diet_ipcl"),
    }


def add_contrast_metadata(source: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    contrast_cols = [
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
        "compare_diet_ipcl",
        "compare_goal_selfsupervised",
        "compare_goal_contrastive",
        "compare_diet_imagenetsize",
    ]
    keep = [col for col in contrast_cols if col in contrasts.columns]
    contrast_meta = contrasts[keep].drop_duplicates("result_model")
    return source.merge(
        contrast_meta.rename(columns={"result_model": "model"}),
        on="model",
        how="left",
    )


def write_count_tables(results: pd.DataFrame, files: list[Path], out_dir: Path) -> None:
    subjects = []
    for path in files:
        match = re.search(r"(sub-\d+)", path.name)
        subjects.append(match.group(1) if match else "unknown")
    (
        pd.Series(subjects, name="subject")
        .value_counts()
        .rename_axis("subject")
        .reset_index(name="n_files")
        .sort_values("subject")
        .to_csv(out_dir / "result_file_counts_by_subject.csv", index=False)
    )
    (
        results.groupby(["subject", "eval_type", "score_set"], observed=True)
        .agg(
            n_rows=("score", "size"),
            n_nan=("score", lambda x: int(x.isna().sum())),
            n_models=("model", "nunique"),
        )
        .reset_index()
        .sort_values(["subject", "eval_type", "score_set"])
        .to_csv(out_dir / "result_counts_by_subject_metric_score_set.csv", index=False)
    )
    (
        results.groupby(["subject", "model", "eval_type", "score_set"], observed=True)
        .agg(
            n_rows=("score", "size"),
            n_nan=("score", lambda x: int(x.isna().sum())),
            n_layers=("model_layer_index", "nunique"),
        )
        .reset_index()
        .sort_values(["subject", "model", "eval_type", "score_set"])
        .to_csv(out_dir / "result_counts_by_subject_model_metric_score_set.csv", index=False)
    )


def compute_best_layers(
    results: pd.DataFrame,
    region: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if region is not None:
        results = results[results["region"] == region]
        if results.empty:
            raise ValueError(f"No rows for region={region!r}")
    keys = ["subject", "model", "eval_type", "region", "model_layer", "model_layer_index"]
    score_keys = keys + ["score_set"]
    duplicate_rows = int(results.duplicated(score_keys).sum())
    compact = results[score_keys + ["score"]].copy()
    if duplicate_rows:
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
    idx = finite_train.groupby(["subject", "model", "eval_type"], observed=True)["train_score"].idxmax()
    best = finite_train.loc[idx].copy().reset_index(drop=True)
    max_layer = (
        wide.groupby(["subject", "model"], observed=True)["model_layer_index"]
        .max()
        .rename("max_model_layer_index")
        .reset_index()
    )
    best = best.merge(max_layer, on=["subject", "model"], how="left")
    best["selected_layer_fraction"] = (
        best["model_layer_index"].astype(float)
        / best["max_model_layer_index"].replace({0: np.nan}).astype(float)
    )
    return best.sort_values(["subject", "model", "eval_type"]), wide, duplicate_rows


def add_metadata(best: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    keep = [
        col
        for col in [
            "model",
            "model_class",
            "train_type",
            "train_data",
            "model_source",
            "task_cluster",
            "comparison_group",
        ]
        if col in metadata.columns
    ]
    return best.merge(metadata[keep], on="model", how="left")


def sem(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return float("nan")
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def compute_group_summary(best: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        col
        for col in [
            "model_class",
            "train_type",
            "train_data",
            "model_source",
            "task_cluster",
            "comparison_group",
        ]
        if col in best.columns
    ]
    meta = best[["model"] + meta_cols].drop_duplicates("model")
    summary = (
        best.groupby(["model", "eval_type"], observed=True)
        .agg(
            n_subjects_total=("subject", "nunique"),
            n_subjects_test=("test_score", lambda x: int(x.notna().sum())),
            n_nan_test=("test_score", lambda x: int(x.isna().sum())),
            mean_test_score=("test_score", "mean"),
            median_test_score=("test_score", "median"),
            sem_test_score=("test_score", sem),
            std_test_score=("test_score", "std"),
            mean_train_score=("train_score", "mean"),
            median_selected_layer_index=("model_layer_index", "median"),
            median_selected_layer_fraction=("selected_layer_fraction", "median"),
        )
        .reset_index()
    )
    return summary.merge(meta, on="model", how="left")


def centered_bootstrap_ci(values: Iterable[float], rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    mean = float(vals.mean())
    if len(vals) == 1:
        return mean, mean
    centered = vals - mean
    boot = rng.choice(centered, size=(n_boot, len(centered)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(mean + lo), float(mean + hi)


def model_stats(
    best: pd.DataFrame,
    models: list[str],
    eval_type: str,
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    rows = []
    sub = best[(best["eval_type"] == eval_type) & (best["model"].isin(models))]
    for model in models:
        values = sub.loc[sub["model"] == model, "test_score"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        ci_lo, ci_hi = centered_bootstrap_ci(values, rng, n_boot)
        rows.append(
            {
                "model": model,
                "eval_type": eval_type,
                "mean": float(values.mean()),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_subjects": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def order_models_by_wrsa(best: pd.DataFrame, models: list[str]) -> list[str]:
    wrsa = (
        best[(best["eval_type"] == "wrsa") & (best["model"].isin(models))]
        .groupby("model", observed=True)["test_score"]
        .mean()
    )
    crsa = (
        best[(best["eval_type"] == "crsa") & (best["model"].isin(models))]
        .groupby("model", observed=True)["test_score"]
        .mean()
    )

    def key(model: str) -> tuple[float, str]:
        score = wrsa.get(model)
        if pd.isna(score):
            score = crsa.get(model)
        if pd.isna(score):
            score = -np.inf
        return float(score), model

    return sorted([m for m in models if m in set(best["model"])], key=key)


def load_noise_ceiling(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame | None:
    """Load per-subject NC for ``args.region`` + ``args.nc_metric`` (test set)."""
    path = args.noise_ceiling
    if path is None or not Path(path).exists():
        print(
            f"WARNING: noise-ceiling CSV not found at {path}; "
            "omitting noise-ceiling ribbon. Run scripts/compute_noise_ceiling.py first."
        )
        return None
    raw = pd.read_csv(path)
    required = {"subject", "region", "metric", "score_set", "nc_pearson"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(
            f"--noise-ceiling {path} is missing columns {sorted(missing)}"
        )
    sub = raw[
        (raw["region"] == args.region)
        & (raw["metric"] == args.nc_metric)
        & (raw["score_set"] == "test")
    ].copy()
    if sub.empty:
        print(
            f"WARNING: noise-ceiling CSV has no rows for region={args.region}, "
            f"metric={args.nc_metric}, score_set=test; omitting ribbon."
        )
        return None
    sub = (
        sub[["subject", "nc_pearson", "n_voxels", "n_stimuli"]]
        .drop_duplicates("subject")
        .reset_index(drop=True)
    )
    sub["source"] = str(path)
    sub["region"] = args.region
    sub["metric"] = args.nc_metric
    sub.to_csv(out_dir / "noise_ceiling_summary.csv", index=False)
    return sub


def draw_noise_ceiling(ax: matplotlib.axes.Axes, nc: pd.DataFrame | None) -> float | None:
    """Single thick gray bar at the mean noise ceiling (no per-subject band)."""
    if nc is None or nc.empty:
        return None
    vals = nc["nc_pearson"].dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return None
    mean = float(vals.mean())
    ax.axhline(mean, color=CONWELL["noise_line"], linewidth=2.4, alpha=0.95, zorder=1)
    return mean


def add_eve_axis(ax: matplotlib.axes.Axes, nc: pd.DataFrame | None) -> None:
    """Right axis labels EVE = r^2 / nc^2 at Conwell-style non-uniform ticks."""
    if nc is None or nc.empty:
        return
    mean_nc = float(nc["nc_pearson"].mean())
    if not np.isfinite(mean_nc) or mean_nc <= 0:
        return
    twin = ax.twinx()
    twin.set_ylim(ax.get_ylim())
    y_lo, y_hi = ax.get_ylim()

    eve_targets = [0.08, 0.14, 0.39, 0.56, 0.77, 1.00]
    nc2 = mean_nc * mean_nc
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for eve in eve_targets:
        r = math.sqrt(max(eve * nc2, 0.0))
        if r < y_lo - 1e-9 or r > y_hi + 1e-9:
            continue
        tick_positions.append(r)
        tick_labels.append(f"{eve:.2f}".lstrip("0") if eve < 1 else "1.00")
    twin.set_yticks(tick_positions)
    twin.set_yticklabels(tick_labels)
    twin.set_ylabel("Variance Explained", color=CONWELL["gray_dark"])
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color(CONWELL["gray_mid"])
    twin.spines["top"].set_visible(False)
    twin.tick_params(width=0.55, length=2.5, colors=CONWELL["gray_dark"])


def _darken(hex_color: str, factor: float) -> str:
    """Return a darker shade of ``hex_color`` (factor < 1 darkens)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = (int(c * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_box(
    ax: matplotlib.axes.Axes,
    x: float,
    row: pd.Series,
    color: str,
    eval_type: str,
    width: float = 1.0,
) -> None:
    mean = float(row["mean"])
    lo = float(row["ci_lo"])
    hi = float(row["ci_hi"])
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = hi = mean
    if hi - lo < 0.002:
        lo, hi = mean - 0.001, mean + 0.001

    is_crsa = eval_type == "crsa"
    # Box outline + mean line always in the group color at full opacity. The
    # fill is the box color, but only veRSA carries a (semi-transparent) fill;
    # cRSA stays unfilled.
    face_rgba = (*matplotlib.colors.to_rgb(color), 0.0 if is_crsa else 0.30)
    edge = color
    rect = mpatches.Rectangle(
        (x - width / 2, lo),
        width,
        hi - lo,
        facecolor=face_rgba,
        edgecolor=edge,
        linewidth=0.7,
        zorder=3 if is_crsa else 4,
    )
    ax.add_patch(rect)
    ax.hlines(mean, x - width / 2, x + width / 2, color=color, linewidth=1.6, zorder=5)


def draw_group_ribbon(
    ax: matplotlib.axes.Axes,
    stats: pd.DataFrame,
    x_start: float,
    x_end: float,
    color: str,
    eval_type: str,
    rng: np.random.Generator,
    n_boot: int,
) -> dict:
    """Group-level CI ribbon, Conwell-styled.

    veRSA gets a filled colored rectangle spanning the bootstrap CI of the
    group mean (matching Conwell's "median box for the wRSA scores is
    filled"). cRSA gets the same rectangle but outlined only (no fill), so
    the two ribbons are visually parallel to the per-model boxes.
    """
    vals = stats["mean"].dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    mean = float(vals.mean())
    ci_lo, ci_hi = centered_bootstrap_ci(vals, rng, n_boot)
    if not np.isfinite(ci_lo) or not np.isfinite(ci_hi) or (ci_hi - ci_lo) < 0.002:
        ci_lo, ci_hi = mean - 0.0015, mean + 0.0015

    is_crsa = eval_type == "crsa"
    face_rgba = (*matplotlib.colors.to_rgb(color), 0.0 if is_crsa else 0.25)
    rect = mpatches.Rectangle(
        (x_start, ci_lo),
        x_end - x_start,
        ci_hi - ci_lo,
        facecolor=face_rgba,
        edgecolor=color,
        linewidth=0.7,
        linestyle=(0, (1.2, 1.4)),
        zorder=2,
    )
    ax.add_patch(rect)
    return {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi}


def prepare_panel_source(
    best: pd.DataFrame,
    panel_name: str,
    groups: list[tuple[str, list[str], str]],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float, list[str]]]]:
    rows = []
    spans: dict[str, tuple[float, float, list[str]]] = {}
    cursor = 0.0
    gap = 2.0
    for group_label, models, color in groups:
        ordered = order_models_by_wrsa(best, models)
        if not ordered:
            continue
        x_start = cursor
        spans[group_label] = (x_start, x_start + len(ordered) - 1, ordered)
        for eval_type in PLOT_METRICS:
            stats = model_stats(best, ordered, eval_type, rng, n_boot)
            if stats.empty:
                continue
            stats["panel"] = panel_name
            stats["group_label"] = group_label
            stats["group_color"] = color
            stats["x"] = stats["model"].map({m: x_start + i for i, m in enumerate(ordered)})
            rows.append(stats)
        cursor += len(ordered) + gap
    if rows:
        return pd.concat(rows, ignore_index=True), spans
    return pd.DataFrame(), spans


def draw_controlled_panel(
    ax: matplotlib.axes.Axes,
    best: pd.DataFrame,
    panel_name: str,
    groups: list[tuple[str, list[str], str]],
    nc: pd.DataFrame | None,
    rng: np.random.Generator,
    n_boot: int,
    title: str | None = None,
    label_func=short_label,
    xtick_fontsize: float = 5.5,
    group_label_fontsize: float = 9.0,
    ylabel: bool = False,
    show_legend: bool = False,
    group_label_y: float = 0.03,
    group_label_mode: str = "centered",
) -> pd.DataFrame:
    panel_stats, spans = prepare_panel_source(best, panel_name, groups, rng, n_boot)
    draw_noise_ceiling(ax, nc)

    ribbon_rows = []
    for group_label, (x_start, x_end, ordered) in spans.items():
        color = next(color for label, _, color in groups if label == group_label)
        for eval_type in PLOT_METRICS:
            stats = panel_stats[
                (panel_stats["group_label"] == group_label)
                & (panel_stats["eval_type"] == eval_type)
            ].sort_values("x")
            for _, row in stats.iterrows():
                draw_box(ax, float(row["x"]), row, color, eval_type)
            if len(stats) > 1:
                ribbon = draw_group_ribbon(
                    ax,
                    stats,
                    x_start - 0.5,
                    x_end + 0.5,
                    color,
                    eval_type,
                    rng,
                    n_boot,
                )
                ribbon_rows.append(
                    {
                        "panel": panel_name,
                        "group_label": group_label,
                        "eval_type": eval_type,
                        "x_start": x_start - 0.5,
                        "x_end": x_end + 0.5,
                        **ribbon,
                    }
                )

    ticks, labels, tick_colors = [], [], []
    group_centers, group_text, group_colors = [], [], []
    for group_label, (x_start, x_end, ordered) in spans.items():
        color = next(c for label, _, c in groups if label == group_label)
        for i, model in enumerate(ordered):
            ticks.append(x_start + i)
            labels.append(label_func(model))
            tick_colors.append(color)
        group_centers.append((x_start + x_end) / 2)
        group_text.append(group_label)
        group_colors.append(color)

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=xtick_fontsize)
    # Tick marks gray, but each label keeps its group color. tick_params has
    # to run BEFORE the per-label color/style loop because labelcolor here
    # would otherwise overwrite the colored labels.
    ax.tick_params(axis="x", which="both", color=CONWELL["gray_mid"], length=2.0)
    for tick_label, color in zip(ax.get_xticklabels(), tick_colors):
        tick_label.set_color(color)
        tick_label.set_style("italic")
    ax.set_xlabel("")
    if ylabel:
        ax.set_ylabel(r"$r_{\mathrm{Pearson}}$ (Score)")
    if title:
        ax.set_title(title, loc="left", fontsize=8.5, color=CONWELL["gray_dark"], pad=4)
    if ticks:
        ax.set_xlim(min(ticks) - 1.0, max(ticks) + 1.0)

    nc_mean = float(nc["nc_pearson"].mean()) if nc is not None and not nc.empty else np.nan
    data_hi = panel_stats["ci_hi"].max() if not panel_stats.empty else 0.3
    y_top = float(nc_mean) if np.isfinite(nc_mean) else float(data_hi) + 0.06
    ax.set_ylim(0.0, y_top)

    # Group labels inside the plot, sitting just above the x-axis.
    if group_label_mode == "hpacker":
        color_by_group = {label: color for label, color in zip(group_text, group_colors)}
        _draw_bottom_group_labels(
            ax,
            group_text,
            color_by_group,
            y=group_label_y,
            fontsize=group_label_fontsize,
        )
    else:
        blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for center, text, color in zip(group_centers, group_text, group_colors):
            ax.text(
                center,
                group_label_y,
                text,
                transform=blended,
                ha="center",
                va="bottom",
                fontsize=group_label_fontsize,
                color=color,
                style="italic",
                fontweight=900,
            )

    if show_legend:
        handles = [
            mpatches.Patch(facecolor="white", edgecolor=CONWELL["crsa_edge"], linewidth=0.6, label="cRSA"),
            mpatches.Patch(facecolor=CONWELL["gray_mid"], edgecolor=CONWELL["gray_mid"], alpha=0.9, label="veRSA / wRSA"),
        ]
        if nc is not None and not nc.empty:
            handles.append(mpatches.Patch(facecolor=COLORS["noise"], edgecolor="none", alpha=0.55, label="noise ceiling"))
        ax.legend(
            handles=handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.55),
            ncol=len(handles),
            columnspacing=1.4,
        )

    ribbons = pd.DataFrame(ribbon_rows)
    if not ribbons.empty:
        panel_stats = panel_stats.merge(
            ribbons,
            on=["panel", "group_label", "eval_type"],
            how="left",
            suffixes=("", "_group"),
        )
    return panel_stats


def draw_ranked_panel(
    ax: matplotlib.axes.Axes,
    best: pd.DataFrame,
    panel_name: str,
    groups: list[tuple[str, list[str], str]],
    nc: pd.DataFrame | None,
    rng: np.random.Generator,
    n_boot: int,
    title: str | None = None,
    label_func=short_label,
    xtick_fontsize: float = 5.5,
    ylabel: bool = False,
    show_category_legend: bool = True,
    group_label_y: float = 0.03,
    group_label_fontsize: float = 7.5,
) -> pd.DataFrame:
    """Draw one globally ranked model axis, coloring models by source group."""
    model_group: dict[str, tuple[str, str]] = {}
    all_models: list[str] = []
    for group_label, models, color in groups:
        for model in models:
            if model in model_group:
                continue
            model_group[model] = (group_label, color)
            all_models.append(model)

    ordered = order_models_by_wrsa(best, all_models)
    draw_noise_ceiling(ax, nc)
    rows = []
    for eval_type in PLOT_METRICS:
        stats = model_stats(best, ordered, eval_type, rng, n_boot)
        if stats.empty:
            continue
        stats["panel"] = panel_name
        stats["x"] = stats["model"].map({m: i for i, m in enumerate(ordered)})
        stats["group_label"] = stats["model"].map(lambda m: model_group[m][0])
        stats["group_color"] = stats["model"].map(lambda m: model_group[m][1])
        rows.append(stats)
    panel_stats = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    for _, row in panel_stats.sort_values(["x", "eval_type"]).iterrows():
        color = str(row["group_color"])
        draw_box(ax, float(row["x"]), row, color, str(row["eval_type"]))

    # Single dashed reference line per eval_type at the panel-wide mean of means.
    for eval_type in PLOT_METRICS:
        es = panel_stats[panel_stats["eval_type"] == eval_type]
        if es.empty:
            continue
        mean_of_means = float(es["mean"].mean())
        is_crsa = eval_type == "crsa"
        line_color = CONWELL["gray_dark"] if is_crsa else CONWELL["gray_mid"]
        ax.hlines(
            mean_of_means,
            -1.0 + 0.5,
            len(ordered) + 0.5 - 1.0,
            colors=line_color,
            linestyles=(0, (4.5, 2.5)),
            linewidth=0.7,
            alpha=0.6,
            zorder=2,
        )

    ticks = list(range(len(ordered)))
    tick_colors = [model_group[m][1] for m in ordered]
    ax.set_xticks(ticks)
    ax.set_xticklabels([label_func(model) for model in ordered], rotation=60, ha="right", fontsize=xtick_fontsize)
    ax.tick_params(axis="x", which="both", color=CONWELL["gray_mid"], length=2.0)
    for tick_label, color in zip(ax.get_xticklabels(), tick_colors):
        tick_label.set_color(color)
        tick_label.set_style("italic")
    ax.set_xlabel("")
    if ylabel:
        ax.set_ylabel(r"$r_{\mathrm{Pearson}}$ (Score)")
    if title:
        ax.set_title(title, loc="left", fontsize=8.5, color=CONWELL["gray_dark"], pad=4)
    if ticks:
        ax.set_xlim(min(ticks) - 1.0, max(ticks) + 1.0)

    nc_mean = float(nc["nc_pearson"].mean()) if nc is not None and not nc.empty else np.nan
    data_hi = panel_stats["ci_hi"].max() if not panel_stats.empty else 0.3
    y_top = float(nc_mean) if np.isfinite(nc_mean) else float(data_hi) + 0.06
    ax.set_ylim(0.0, y_top)

    if show_category_legend:
        present = unique_ordered(panel_stats["group_label"]) if not panel_stats.empty else []
        color_by_group = {
            group_label: color
            for group_label, _, color in groups
            if group_label in set(present)
        }
        _draw_bottom_group_labels(
            ax,
            present,
            color_by_group,
            y=group_label_y,
            fontsize=group_label_fontsize,
        )
    return panel_stats


def _draw_bottom_group_labels(
    ax: matplotlib.axes.Axes,
    labels: list[str],
    color_by_label: dict[str, str],
    y: float,
    fontsize: float,
) -> None:
    """Place colored italic group labels in one centered row below the axis.

    Uses matplotlib's HPacker so widths are measured per-glyph instead of
    estimated from string length — that avoids the overlap we hit when
    labels are short but very different in pixel width.
    """
    if not labels:
        return
    children = []
    for i, label in enumerate(labels):
        children.append(
            TextArea(
                label,
                textprops=dict(
                    color=color_by_label.get(label, CONWELL["gray_dark"]),
                    fontsize=fontsize,
                    style="italic",
                    fontweight=900,
                ),
            )
        )
        if i < len(labels) - 1:
            children.append(
                TextArea(
                    "|",
                    textprops=dict(color=CONWELL["gray_mid"], fontsize=fontsize, fontweight=900),
                )
            )
    hbox = HPacker(children=children, align="center", pad=0, sep=6)
    box = AnchoredOffsetbox(
        loc="lower center",
        child=hbox,
        frameon=False,
        bbox_to_anchor=(0.5, y),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    ax.add_artist(box)


def add_panel_label(ax: matplotlib.axes.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10,
        color=CONWELL["gray_dark"],
        va="top",
    )


def plot_fig2_architecture(
    best: pd.DataFrame,
    contrasts: pd.DataFrame,
    nc: pd.DataFrame | None,
    out_dir: Path,
    formats: list[str],
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    groups = [
        ("Convolutional Neural Networks", contrast_models(contrasts, "compare_architecture", "Convolutional"), COLORS["cnn"]),
        ("Transformers", contrast_models(contrasts, "compare_architecture", "Transformer"), COLORS["transformer"]),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    stats = draw_controlled_panel(
        ax,
        best,
        "Fig2 architecture",
        groups,
        nc,
        rng,
        n_boot,
        title=None,
        label_func=contrast_labeler(contrasts, 24),
        xtick_fontsize=4.0,
        group_label_fontsize=8.0,
        ylabel=True,
        show_legend=False,
        group_label_y=0.025,
    )
    add_eve_axis(ax, nc)
    fig.subplots_adjust(bottom=0.20, top=0.96, left=0.08, right=0.93)
    save_figure(fig, out_dir, "fig2_architecture_variation", formats)
    return stats


def plot_fig3_task(
    best: pd.DataFrame,
    contrasts: pd.DataFrame,
    nc: pd.DataFrame | None,
    out_dir: Path,
    formats: list[str],
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    # Conwell fig3a uses a single olive shade across every Taskonomy model and
    # a single "Taskonomy Models" label, rather than per-cluster colors. We
    # collapse the cluster contrasts into one group to match.
    all_taskonomy = unique_ordered(
        m
        for cluster in ["2D", "3D", "Geometric", "Semantic", "Other", "Random"]
        for m in contrast_models(contrasts, "compare_goal_taskonomy_cluster", cluster)
    )
    taskonomy_groups = [("Taskonomy Models", all_taskonomy, COLORS["taskonomy"])]
    vissl_groups = [
        (
            "Contrastive",
            contrast_models(contrasts, "compare_goal_contrastive", "Contrastive"),
            COLORS["contrastive"],
        ),
        (
            "Non-contrastive",
            contrast_models(contrasts, "compare_goal_contrastive", "Non-Contrastive"),
            COLORS["noncontrastive"],
        ),
        ("Supervised", [resolve_model_string("resnet50_classification", set(best["model"]))], COLORS["supervised"]),
    ]
    slip_groups = [
        (
            "SimCLR",
            contrast_models(contrasts, "compare_goal_slip", "SimCLR"),
            COLORS["simclr"],
        ),
        (
            "CLIP",
            contrast_models(contrasts, "compare_goal_slip", "CLIP"),
            COLORS["clip"],
        ),
        (
            "SLIP",
            contrast_models(contrasts, "compare_goal_slip", "SLIP"),
            COLORS["slip"],
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 5.2), gridspec_kw={"width_ratios": [1.35, 0.85, 0.9]})
    all_stats = []
    all_stats.append(
        draw_ranked_panel(
            axes[0],
            best,
            "Fig3A Taskonomy",
            taskonomy_groups,
            nc,
            rng,
            n_boot,
            title="Taskonomy ResNet50",
            label_func=contrast_labeler(contrasts, 18),
            xtick_fontsize=5.0,
            ylabel=True,
            group_label_y=0.025,
            group_label_fontsize=7.0,
        )
    )
    all_stats.append(
        draw_controlled_panel(
            axes[1],
            best,
            "Fig3B VISSL",
            vissl_groups,
            nc,
            rng,
            n_boot,
            title="ResNet50 SSL",
            label_func=contrast_labeler(contrasts, 18),
            xtick_fontsize=5.0,
            group_label_fontsize=6.5,
            group_label_y=0.025,
            group_label_mode="hpacker",
        )
    )
    all_stats.append(
        draw_controlled_panel(
            axes[2],
            best,
            "Fig3C SLIP",
            slip_groups,
            nc,
            rng,
            n_boot,
            title="SLIP ViT",
            label_func=contrast_labeler(contrasts, 18),
            xtick_fontsize=5.0,
            group_label_fontsize=6.5,
            group_label_y=0.025,
            group_label_mode="hpacker",
        )
    )
    for label, ax in zip("abc", axes):
        add_panel_label(ax, label)

    y0 = min(ax.get_ylim()[0] for ax in axes)
    y1 = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(y0, y1)
    add_eve_axis(axes[-1], nc)
    fig.subplots_adjust(wspace=0.34, bottom=0.22, top=0.94, left=0.08, right=0.93)
    save_figure(fig, out_dir, "fig3_task_variation", formats)
    return pd.concat(all_stats, ignore_index=True)


def is_in21k_model(model: str) -> bool:
    name = model.lower()
    return "in21k" in name or "in22k" in name or "in22ft" in name


def plot_fig4_input(
    best: pd.DataFrame,
    contrasts: pd.DataFrame,
    nc: pd.DataFrame | None,
    out_dir: Path,
    formats: list[str],
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    input_groups = [
        ("ImageNet-1K", contrast_models(contrasts, "compare_diet_imagenetsize", "imagenet"), COLORS["in1k"]),
        ("ImageNet-21K", contrast_models(contrasts, "compare_diet_imagenetsize", "imagenet21k"), COLORS["in21k"]),
    ]
    ipcl_groups = [
        ("ImageNet", contrast_models(contrasts, "compare_diet_ipcl", "imagenet"), COLORS["ipcl_imagenet"]),
        ("OpenImages", contrast_models(contrasts, "compare_diet_ipcl", "openimages"), COLORS["ipcl_openimages"]),
        ("Places256", contrast_models(contrasts, "compare_diet_ipcl", "places256"), COLORS["ipcl_places256"]),
        ("VGGFace2", contrast_models(contrasts, "compare_diet_ipcl", "vggface2"), COLORS["ipcl_vggface2"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 5.1), gridspec_kw={"width_ratios": [2.65, 1.0]})
    stats_a = draw_controlled_panel(
        axes[0],
        best,
        "Fig4A ImageNet diet",
        input_groups,
        nc,
        rng,
        n_boot,
        title="ImageNet-1K vs ImageNet-21K",
        label_func=contrast_labeler(contrasts, 24),
        xtick_fontsize=4.5,
        group_label_fontsize=7.5,
        ylabel=True,
        group_label_y=0.025,
    )
    stats_b = draw_ranked_panel(
        axes[1],
        best,
        "Fig4B IPCL domains",
        ipcl_groups,
        nc,
        rng,
        n_boot,
        title="IPCL input domains",
        label_func=contrast_labeler(contrasts, 16),
        xtick_fontsize=5.5,
        group_label_y=0.025,
        group_label_fontsize=6.5,
    )
    for label, ax in zip("ab", axes):
        add_panel_label(ax, label)
    y0 = min(ax.get_ylim()[0] for ax in axes)
    y1 = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(y0, y1)
    add_eve_axis(axes[-1], nc)
    fig.subplots_adjust(wspace=0.34, bottom=0.30, top=0.94, left=0.07, right=0.93)
    save_figure(fig, out_dir, "fig4_input_variation", formats)
    return pd.concat([stats_a, stats_b], ignore_index=True)


def write_controlled_summary(
    best: pd.DataFrame,
    contrasts: pd.DataFrame,
    out_dir: Path,
    results: pd.DataFrame,
    files: list[Path],
    nc: pd.DataFrame | None,
    duplicate_rows: int,
) -> str:
    comparisons = comparison_model_sets(contrasts)
    present = set(best["model"].unique())
    count_rows = []
    missing_lines = []
    for label, models in comparisons.items():
        models = unique_ordered(models)
        n_present = len([m for m in models if m in present])
        count_rows.append(
            {
                "comparison": label,
                "models_in_deepnsd_contrast": len(models),
                "models_in_results": n_present,
            }
        )
        missing = [m for m in models if m not in present]
        if missing:
            missing_lines.append(f"- {label}: missing {len(missing)} models, e.g. {', '.join(missing[:5])}")
    count_df = pd.DataFrame(count_rows)
    count_df.to_csv(out_dir / "controlled_comparison_model_counts.csv", index=False)

    metric_stats = (
        best.groupby("eval_type", observed=True)["test_score"]
        .agg(
            n="count",
            n_nan=lambda x: int(x.isna().sum()),
            mean="mean",
            median="median",
            std="std",
            sem=sem,
        )
        .reindex(METRICS)
        .reset_index()
    )

    nc_text = "not drawn"
    if nc is not None and not nc.empty:
        nc_text = (
            f"mean r={nc['nc_pearson'].mean():.4f}, "
            f"range=[{nc['nc_pearson'].min():.4f}, {nc['nc_pearson'].max():.4f}], "
            f"source={nc['source'].iloc[0]}"
        )

    lines = [
        "Conwell odd/even split-half controlled-comparison summary",
        "",
        f"Result files: {len(files)}",
        f"Rows: {len(results):,}",
        f"Subjects: {', '.join(sorted(results['subject'].unique()))}",
        f"Models: {results['model'].nunique()}",
        f"NaN score rows: {int(results['score'].isna().sum())}",
        f"Duplicate raw score keys averaged before selection: {duplicate_rows}",
        f"Controlled contrast file: {contrasts.attrs.get('source_path', 'unknown')}",
        f"Noise ceiling: {nc_text}",
        "",
        "Controlled model coverage:",
        count_df.to_string(index=False),
        "",
        "Selected-layer test score summary by metric:",
        metric_stats.to_string(
            index=False,
            formatters={
                "mean": "{:.4f}".format,
                "median": "{:.4f}".format,
                "std": "{:.4f}".format,
                "sem": "{:.4f}".format,
            },
        ),
    ]
    if missing_lines:
        lines.append("")
        lines.append("Coverage caveats:")
        lines.extend(missing_lines)
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text + "\n")
    return text


def _replot_from_stats(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    scores_path = out_dir / "model_metric_subject_scores.csv"
    nc_path = out_dir / "noise_ceiling_summary.csv"
    if not scores_path.exists():
        raise SystemExit(f"--from-stats requires {scores_path}")
    best = pd.read_csv(scores_path)
    if args.region and "region" in best.columns:
        best = best[best["region"] == args.region].reset_index(drop=True)
    nc = pd.read_csv(nc_path) if nc_path.exists() else None
    if nc is not None and nc.empty:
        nc = None
    contrasts = load_model_contrasts(args.model_contrasts, set(best["model"]))
    rng = np.random.default_rng(20260504)
    for fmt_path in list(out_dir.glob("fig*_*.png")) + list(out_dir.glob("fig*_*.pdf")):
        fmt_path.unlink()
    plot_fig2_architecture(best, contrasts, nc, out_dir, args.formats, rng, args.bootstrap)
    plot_fig3_task(best, contrasts, nc, out_dir, args.formats, rng, args.bootstrap)
    plot_fig4_input(best, contrasts, nc, out_dir, args.formats, rng, args.bootstrap)
    print(f"Replotted figures in {out_dir}")


def main() -> None:
    args = parse_args()
    set_style()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_stats:
        _replot_from_stats(args)
        return

    clear_old_figures(out_dir)

    print(f"Loading parquet files from {args.results_dir}", flush=True)
    results, files = load_results(args.results_dir)

    for col, value in (
        ("split", args.filter_split),
        ("pool", args.filter_pool),
    ):
        if value is None:
            continue
        if col not in results.columns:
            raise SystemExit(f"--filter-{col} given but results have no '{col}' column")
        before = len(results)
        results = results[results[col].astype(str) == str(value)].reset_index(drop=True)
        print(f"Filtered to {col}={value!r}: {before:,} -> {len(results):,} rows", flush=True)
    if args.filter_ood_type is not None:
        if "ood_type" not in results.columns:
            raise SystemExit("--filter-ood-type given but results have no 'ood_type' column")
        before = len(results)
        # Comma-separated list. The token 'none' matches null ood_type rows
        # (i.e. the train rows in the OOD split, plus all rows in non-OOD
        # splits). Pass e.g. 'none,all' to keep both train rows and the
        # aggregate test rows of the OOD split.
        tokens = [t.strip() for t in str(args.filter_ood_type).split(",") if t.strip()]
        keep_null = any(t.lower() == "none" for t in tokens)
        explicit = [t for t in tokens if t.lower() != "none"]
        mask = pd.Series(False, index=results.index)
        if keep_null:
            mask |= results["ood_type"].isna()
        if explicit:
            mask |= results["ood_type"].astype(str).isin(explicit)
        results = results[mask].reset_index(drop=True)
        print(f"Filtered to ood_type={args.filter_ood_type}: {before:,} -> {len(results):,} rows", flush=True)
    if results.empty:
        raise SystemExit("No rows after filtering")

    metadata = load_metadata(args.metadata)
    write_count_tables(results, files, out_dir)

    print(f"Selecting best layers by highest train score (region={args.region})", flush=True)
    best, wide, duplicate_rows = compute_best_layers(results, region=args.region)
    best = add_metadata(best, metadata)
    contrasts = load_model_contrasts(args.model_contrasts, set(best["model"]))
    group_summary = compute_group_summary(best)

    best.to_parquet(out_dir / "best_layer_by_subject_model_metric.parquet", index=False)
    best.to_csv(out_dir / "model_metric_subject_scores.csv", index=False)
    group_summary.to_csv(out_dir / "model_metric_group_summary.csv", index=False)

    nc = load_noise_ceiling(args, out_dir)
    rng = np.random.default_rng(20260504)

    print(f"Writing controlled figures to {out_dir}", flush=True)
    source_frames = [
        plot_fig2_architecture(best, contrasts, nc, out_dir, args.formats, rng, args.bootstrap),
        plot_fig3_task(best, contrasts, nc, out_dir, args.formats, rng, args.bootstrap),
        plot_fig4_input(best, contrasts, nc, out_dir, args.formats, rng, args.bootstrap),
    ]
    source = add_contrast_metadata(pd.concat(source_frames, ignore_index=True), contrasts)
    source.to_csv(out_dir / "controlled_figure_model_stats.csv", index=False)

    summary = write_controlled_summary(best, contrasts, out_dir, results, files, nc, duplicate_rows)
    print()
    print(textwrap.dedent(summary))
    print()
    print(f"Done. Outputs are in {out_dir}")


if __name__ == "__main__":
    main()
