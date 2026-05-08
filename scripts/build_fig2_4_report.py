#!/usr/bin/env python3
"""Build a Markdown report for LAION-fMRI Conwell Fig. 2-4 results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures"
REPORTS = REPO / "reports"
REPORT_FIGURES = REPORTS / "figures"
REPORT = REPORTS / "laion_fmri_conwell_fig2_4_report.md"

OKABE = {
    "black": "#000000",
    "orange": "#e69f00",
    "sky": "#56b4e9",
    "green": "#009e73",
    "yellow": "#f0e442",
    "blue": "#0072b2",
    "vermillion": "#d55e00",
    "purple": "#cc79a7",
    "gray": "#777777",
}
METRIC_ORDER = ("crsa", "wrsa", "srpr")
METRIC_LABELS = {"crsa": "cRSA", "wrsa": "wRSA/veRSA", "srpr": "SRPR"}
METRIC_COLORS = {"crsa": OKABE["black"], "wrsa": OKABE["blue"], "srpr": OKABE["vermillion"]}


@dataclass(frozen=True)
class Variant:
    label: str
    path: Path


@dataclass(frozen=True)
class SplitFamily:
    key: str
    title: str
    description: str
    variants: tuple[Variant, ...]


SPLITS = (
    SplitFamily(
        "splithalf",
        "Odd/Even Split-Half",
        "Closest direct replication of Conwell's shared-image split-half setup.",
        (Variant("split-half", FIGURES / "splithalf_results"),),
    ),
    SplitFamily(
        "splithalf_ood",
        "Odd/Even Split-Half + OOD",
        "Existing OOD-inclusive split-half analysis.",
        (Variant("split-half + OOD", FIGURES / "splithalf_results_ood"),),
    ),
    SplitFamily(
        "random_agg",
        "Random Splits, Averaged Across 5 Repeats",
        "Five random train/test splits averaged before best-layer selection.",
        (
            Variant("shared-image pool", FIGURES / "min_nn_results" / "shared_random_agg"),
            Variant("subject-image pool", FIGURES / "min_nn_results" / "subject_random_agg"),
        ),
    ),
    SplitFamily(
        "cluster_k5_agg",
        "Cluster Splits, Averaged Across 5 Repeats",
        "Five k-means cluster splits averaged before best-layer selection.",
        (
            Variant("shared-image pool", FIGURES / "min_nn_results" / "shared_cluster_k5_agg"),
            Variant("subject-image pool", FIGURES / "min_nn_results" / "subject_cluster_k5_agg"),
        ),
    ),
    SplitFamily(
        "tau",
        "Min-NN Threshold Split",
        "Nearest-neighbor threshold split.",
        (
            Variant("shared-image pool", FIGURES / "min_nn_results" / "shared_tau"),
            Variant("subject-image pool", FIGURES / "min_nn_results" / "subject_tau"),
        ),
    ),
    SplitFamily(
        "ood",
        "OOD Test Split",
        "Model fit once and evaluated on the aggregate OOD test set.",
        (
            Variant("shared-image pool", FIGURES / "min_nn_results" / "shared_ood"),
            Variant("subject-image pool", FIGURES / "min_nn_results" / "subject_ood"),
        ),
    ),
)


EFFECTS = (
    ("architecture", "Transformer", "Transformer - CNN", "Architecture"),
    ("contrastive_self_supervised", "Contrastive", "Contrastive - Non-Contrastive", "Self-supervised"),
    ("language_alignment", "CLIP", "CLIP - SimCLR", "SLIP objective"),
    ("language_alignment", "SLIP", "SLIP - SimCLR", "SLIP objective"),
    ("imagenet_size", "imagenet21k", "ImageNet21K - ImageNet1K", "Input diet"),
    ("objects_faces_places", "openimages", "OpenImages - ImageNet", "IPCL input"),
    ("objects_faces_places", "places256", "Places365 - ImageNet", "IPCL input"),
    ("objects_faces_places", "vggface2", "VGGFace2 - ImageNet", "IPCL input"),
)


PAPER_EFFECTS = {
    ("architecture", "Transformer", "crsa"): "-0.04 [-0.05, -0.03], p < 0.001",
    ("architecture", "Transformer", "wrsa"): "-0.01 [-0.02, -0.00], p < 0.001",
    ("contrastive_self_supervised", "Contrastive", "crsa"): "+0.06 contrastive advantage, p < 0.001",
    ("contrastive_self_supervised", "Contrastive", "wrsa"): "+0.09 contrastive advantage, p < 0.001",
    ("language_alignment", "CLIP", "crsa"): "-0.05 [-0.06, -0.03], p < 0.001",
    ("language_alignment", "CLIP", "wrsa"): "-0.02 [-0.04, -0.01], p = 0.005",
    ("language_alignment", "SLIP", "crsa"): "No substantial effect reported",
    ("language_alignment", "SLIP", "wrsa"): "No substantial effect reported",
    ("imagenet_size", "imagenet21k", "crsa"): "0.00 [-0.03, 0.03], p = 0.957",
    ("imagenet_size", "imagenet21k", "wrsa"): "+0.01 [0.00, 0.03], p = 0.147",
    ("objects_faces_places", "openimages", "crsa"): "-0.02 [-0.03, -0.01], p = 0.002",
    ("objects_faces_places", "openimages", "wrsa"): "-0.04 [-0.07, -0.02], p < 0.001",
    ("objects_faces_places", "places256", "crsa"): "-0.03 [-0.04, -0.02], p < 0.001",
    ("objects_faces_places", "places256", "wrsa"): "-0.07 [-0.09, -0.04], p < 0.001",
    ("objects_faces_places", "vggface2", "crsa"): "-0.17 [-0.18, -0.16], p < 0.001",
    ("objects_faces_places", "vggface2", "wrsa"): "-0.27 [-0.30, -0.25], p < 0.001",
}

PAPER_EFFECT_NUMS = {
    ("architecture", "Transformer", "crsa"): (-0.04, -0.05, -0.03),
    ("architecture", "Transformer", "wrsa"): (-0.01, -0.02, 0.00),
    ("contrastive_self_supervised", "Contrastive", "crsa"): (0.06, np.nan, np.nan),
    ("contrastive_self_supervised", "Contrastive", "wrsa"): (0.09, np.nan, np.nan),
    ("language_alignment", "CLIP", "crsa"): (-0.05, -0.06, -0.03),
    ("language_alignment", "CLIP", "wrsa"): (-0.02, -0.04, -0.01),
    ("imagenet_size", "imagenet21k", "crsa"): (0.00, -0.03, 0.03),
    ("imagenet_size", "imagenet21k", "wrsa"): (0.01, 0.00, 0.03),
    ("objects_faces_places", "openimages", "crsa"): (-0.02, -0.03, -0.01),
    ("objects_faces_places", "openimages", "wrsa"): (-0.04, -0.07, -0.02),
    ("objects_faces_places", "places256", "crsa"): (-0.03, -0.04, -0.02),
    ("objects_faces_places", "places256", "wrsa"): (-0.07, -0.09, -0.04),
    ("objects_faces_places", "vggface2", "crsa"): (-0.17, -0.18, -0.16),
    ("objects_faces_places", "vggface2", "wrsa"): (-0.27, -0.30, -0.25),
}

TASKONOMY_REFERENCES = {
    ("autoencoding", "crsa"): "0.077 [0.066, 0.085]",
    ("autoencoding", "wrsa"): "0.103 [0.096, 0.110]",
    ("class_object", "crsa"): "0.189 [0.178, 0.201]",
    ("class_object", "wrsa"): "0.436 [0.419, 0.454]",
}
TASKONOMY_REFERENCE_NUMS = {
    ("autoencoding", "crsa"): (0.077, 0.066, 0.085),
    ("autoencoding", "wrsa"): (0.103, 0.096, 0.110),
    ("class_object", "crsa"): (0.189, 0.178, 0.201),
    ("class_object", "wrsa"): (0.436, 0.419, 0.454),
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def rel_from_report(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPORT.parent.resolve()).as_posix()
    except ValueError:
        return "../" + path.relative_to(REPO.resolve()).as_posix()


def fmt_num(value: object, ndigits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{ndigits}f}"


def fmt_p(value: object) -> str:
    if pd.isna(value):
        return ""
    p = float(value)
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_effect(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return (
        f"{float(row['beta']):+.3f} "
        f"[{float(row['ci_lo']):+.3f}, {float(row['ci_hi']):+.3f}], "
        f"p {fmt_p(row['p_value'])}"
    )


def md_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    headers = [header for header, _ in columns]
    keys = [key for _, key in columns]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return out


def read_scores(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "model_metric_subject_scores.csv")


def read_stats(path: Path, name: str) -> pd.DataFrame:
    p = path / "statistics" / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 4000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def save_figure(fig: plt.Figure, stem: str) -> dict[str, Path]:
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": REPORT_FIGURES / f"{stem}.png",
        "pdf": REPORT_FIGURES / f"{stem}.pdf",
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return paths


def score_summary(variants: tuple[Variant, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in variants:
        scores = read_scores(variant.path)
        for metric in METRIC_ORDER:
            group = scores[scores["eval_type"] == metric]
            if group.empty:
                continue
            rows.append(
                {
                    "variant": variant.label,
                    "metric": METRIC_LABELS.get(metric, metric),
                    "n": len(group),
                    "subjects": group["subject"].nunique(),
                    "models": group["model"].nunique(),
                    "mean": fmt_num(group["test_score"].mean(), 3),
                    "median": fmt_num(group["test_score"].median(), 3),
                    "min": fmt_num(group["test_score"].min(), 3),
                    "max": fmt_num(group["test_score"].max(), 3),
                }
            )
    return rows


def effect_records(variants: tuple[Variant, ...]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    stats = {variant.label: read_stats(variant.path, "statistical_tests.csv") for variant in variants}
    for comparison_id, level, effect_label, family in EFFECTS:
        for metric in ("crsa", "wrsa"):
            beta, lo, hi = PAPER_EFFECT_NUMS.get((comparison_id, level, metric), (np.nan, np.nan, np.nan))
            row: dict[str, object] = {
                "family": family,
                "comparison_id": comparison_id,
                "level": level,
                "effect": effect_label,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "conwell": PAPER_EFFECTS.get((comparison_id, level, metric), ""),
                "conwell_beta": beta,
                "conwell_ci_lo": lo,
                "conwell_ci_hi": hi,
            }
            for label, table in stats.items():
                match = table[
                    (table["comparison_id"] == comparison_id)
                    & (table["level"].astype(str) == level)
                    & (table["eval_type"] == metric)
                ]
                if len(match):
                    fit = match.iloc[0]
                    row[label] = fmt_effect(fit)
                    row[f"{label}_beta"] = float(fit["beta"])
                    row[f"{label}_ci_lo"] = float(fit["ci_lo"])
                    row[f"{label}_ci_hi"] = float(fit["ci_hi"])
                    row[f"{label}_p"] = float(fit["p_value"])
                else:
                    row[label] = ""
                    row[f"{label}_beta"] = np.nan
                    row[f"{label}_ci_lo"] = np.nan
                    row[f"{label}_ci_hi"] = np.nan
                    row[f"{label}_p"] = np.nan
            records.append(row)
    return records


def effect_table(variants: tuple[Variant, ...]) -> list[dict[str, object]]:
    keep = []
    for row in effect_records(variants):
        out = {
            "family": row["family"],
            "effect": row["effect"],
            "metric": row["metric_label"],
            "conwell": row["conwell"],
        }
        for variant in variants:
            out[variant.label] = row[variant.label]
        keep.append(out)
    return keep


def taskonomy_records(variants: tuple[Variant, ...]) -> list[dict[str, object]]:
    desc = {variant.label: read_stats(variant.path, "comparison_descriptives.csv") for variant in variants}
    rows: list[dict[str, object]] = []
    for level in ("autoencoding", "class_object"):
        for metric in ("crsa", "wrsa"):
            mean, lo, hi = TASKONOMY_REFERENCE_NUMS[(level, metric)]
            row: dict[str, object] = {
                "task": level,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "conwell": TASKONOMY_REFERENCES[(level, metric)],
                "conwell_mean": mean,
                "conwell_ci_lo": lo,
                "conwell_ci_hi": hi,
            }
            for label, table in desc.items():
                match = table[
                    (table["comparison_id"] == "taskonomy_tasks")
                    & (table["level"].astype(str) == level)
                    & (table["eval_type"] == metric)
                ]
                if len(match):
                    row[label] = fmt_num(match.iloc[0]["mean"], 3)
                    row[f"{label}_mean"] = float(match.iloc[0]["mean"])
                else:
                    row[label] = ""
                    row[f"{label}_mean"] = np.nan
            rows.append(row)
    return rows


def taskonomy_table(variants: tuple[Variant, ...]) -> list[dict[str, object]]:
    keep = []
    for row in taskonomy_records(variants):
        out = {"task": row["task"], "metric": row["metric_label"], "conwell": row["conwell"]}
        for variant in variants:
            out[variant.label] = row[variant.label]
        keep.append(out)
    return keep


def overview_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        for variant in split.variants:
            scores = read_scores(variant.path)
            for metric in METRIC_ORDER:
                group = scores[scores["eval_type"] == metric]
                if group.empty:
                    continue
                rows.append(
                    {
                        "split": split.key,
                        "variant": variant.label,
                        "metric": METRIC_LABELS.get(metric, metric),
                        "metric_id": metric,
                        "mean": float(group["test_score"].mean()),
                        "median": float(group["test_score"].median()),
                        "min": float(group["test_score"].min()),
                        "max": float(group["test_score"].max()),
                        "n": len(group),
                        "subjects": group["subject"].nunique(),
                        "models": group["model"].nunique(),
                    }
                )
    return rows


def write_support_tables() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(overview_rows()).to_csv(REPORTS / "laion_fmri_conwell_fig2_4_score_summary.csv", index=False)
    effects = []
    taskonomy = []
    for split in SPLITS:
        for row in effect_table(split.variants):
            effects.append({"split": split.key, **row})
        for row in taskonomy_table(split.variants):
            taskonomy.append({"split": split.key, **row})
    pd.DataFrame(effects).to_csv(REPORTS / "laion_fmri_conwell_fig2_4_effects.csv", index=False)
    pd.DataFrame(taskonomy).to_csv(REPORTS / "laion_fmri_conwell_fig2_4_taskonomy_endpoints.csv", index=False)


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)


def plot_score_summary(split: SplitFamily) -> dict[str, Path]:
    rows = []
    for variant in split.variants:
        scores = read_scores(variant.path)
        for metric in METRIC_ORDER:
            group = scores[scores["eval_type"] == metric]
            if group.empty:
                continue
            rows.append(
                {
                    "label": f"{variant.label} | {METRIC_LABELS[metric]}",
                    "metric": metric,
                    "mean": group["test_score"].mean(),
                    "min": group["test_score"].min(),
                    "max": group["test_score"].max(),
                }
            )
    df = pd.DataFrame(rows)
    fig_h = max(1.8, 0.33 * len(df) + 0.75)
    fig, ax = plt.subplots(figsize=(4.8, fig_h))
    y = np.arange(len(df))[::-1]
    for yi, (_, row) in zip(y, df.iterrows()):
        color = METRIC_COLORS[str(row["metric"])]
        ax.plot([row["min"], row["max"]], [yi, yi], color="#b8b8b8", linewidth=1.2, solid_capstyle="round")
        ax.scatter(row["mean"], yi, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color="#555555", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("selected-layer test score (r)")
    ax.set_title("Score ranges and means", loc="left", fontweight="bold")
    ax.set_xlim(-0.36, 0.70)
    clean_axis(ax)
    return save_figure(fig, f"{split.key}_score_summary")


def plot_effects(split: SplitFamily) -> dict[str, Path]:
    records = effect_records(split.variants)
    labels = [f"{row['effect']} | {row['metric_label']}" for row in records]
    y = np.arange(len(records))[::-1]
    fig_h = max(5.0, 0.28 * len(records) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    ax.axvline(0, color="#555555", linewidth=0.8)
    offsets = np.linspace(-0.17, 0.17, len(split.variants)) if len(split.variants) > 1 else np.array([0.0])
    variant_colors = [OKABE["blue"], OKABE["orange"], OKABE["green"]]

    for yi, row in zip(y, records):
        beta = row["conwell_beta"]
        if np.isfinite(beta):
            lo = row["conwell_ci_lo"]
            hi = row["conwell_ci_hi"]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [yi, yi], color="black", linewidth=1.1)
            ax.scatter(beta, yi, marker="D", s=24, color="black", zorder=4, label="Conwell" if yi == y[0] else None)
        for off, variant, color in zip(offsets, split.variants, variant_colors):
            b = row[f"{variant.label}_beta"]
            if not np.isfinite(b):
                continue
            lo = row[f"{variant.label}_ci_lo"]
            hi = row[f"{variant.label}_ci_hi"]
            ax.plot([lo, hi], [yi + off, yi + off], color=color, linewidth=1.0, alpha=0.95)
            ax.scatter(b, yi + off, s=24, color=color, edgecolor="white", linewidth=0.4, zorder=5)

    handles = [
        plt.Line2D([0], [0], marker="D", linestyle="", color="black", markersize=4.5, label="Conwell"),
        *[
            plt.Line2D([0], [0], marker="o", linestyle="", color=color, markersize=5, label=variant.label)
            for variant, color in zip(split.variants, variant_colors)
        ],
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", ncol=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("effect estimate (beta)")
    ax.set_title("Controlled-effect estimates", loc="left", fontweight="bold")
    ax.set_xlim(-0.34, 0.24)
    clean_axis(ax)
    return save_figure(fig, f"{split.key}_controlled_effects")


def plot_taskonomy_endpoints(split: SplitFamily) -> dict[str, Path]:
    records = taskonomy_records(split.variants)
    labels = [f"{row['task']} | {row['metric_label']}" for row in records]
    y = np.arange(len(records))[::-1]
    fig, ax = plt.subplots(figsize=(5.0, 2.15))
    offsets = np.linspace(-0.14, 0.14, len(split.variants)) if len(split.variants) > 1 else np.array([0.0])
    variant_colors = [OKABE["blue"], OKABE["orange"], OKABE["green"]]

    for yi, row in zip(y, records):
        ax.plot([row["conwell_ci_lo"], row["conwell_ci_hi"]], [yi, yi], color="black", linewidth=1.1)
        ax.scatter(row["conwell_mean"], yi, marker="D", s=24, color="black", zorder=4)
        for off, variant, color in zip(offsets, split.variants, variant_colors):
            mean = row[f"{variant.label}_mean"]
            if np.isfinite(mean):
                ax.scatter(mean, yi + off, s=28, color=color, edgecolor="white", linewidth=0.4, zorder=5)

    handles = [
        plt.Line2D([0], [0], marker="D", linestyle="", color="black", markersize=4.5, label="Conwell"),
        *[
            plt.Line2D([0], [0], marker="o", linestyle="", color=color, markersize=5, label=variant.label)
            for variant, color in zip(split.variants, variant_colors)
        ],
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=len(handles),
        columnspacing=1.2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean selected-layer test score (r)")
    ax.set_title("Taskonomy endpoints", loc="left", fontweight="bold")
    ax.set_xlim(-0.12, 0.52)
    clean_axis(ax)
    fig.subplots_adjust(bottom=0.34)
    return save_figure(fig, f"{split.key}_taskonomy_endpoints")


def plot_overview_scores() -> dict[str, Path]:
    df = pd.DataFrame(overview_rows())
    variant_order = []
    for split in SPLITS:
        for variant in split.variants:
            variant_order.append(f"{split.key} | {variant.label}")
    df["row_label"] = df["split"] + " | " + df["variant"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.9), sharey=True)
    y = np.arange(len(variant_order))[::-1]
    y_lookup = dict(zip(variant_order, y))
    for ax, metric in zip(axes, METRIC_ORDER):
        sub = df[df["metric_id"] == metric]
        for _, row in sub.iterrows():
            yi = y_lookup[row["row_label"]]
            color = METRIC_COLORS[metric]
            ax.plot([row["min"], row["max"]], [yi, yi], color="#bdbdbd", linewidth=1.0, solid_capstyle="round")
            ax.scatter(row["mean"], yi, s=22, color=color, edgecolor="white", linewidth=0.4, zorder=3)
        ax.axvline(0, color="#555555", linewidth=0.65)
        ax.set_title(METRIC_LABELS[metric], loc="left", fontweight="bold")
        ax.set_xlabel("score (r)")
        ax.set_xlim(-0.36, 0.70)
        clean_axis(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(variant_order)
    axes[0].set_ylabel("split and image pool")
    fig.suptitle("Overall score ranges and means across split families", x=0.02, ha="left", fontweight="bold")
    fig.subplots_adjust(wspace=0.18)
    return save_figure(fig, "overview_score_summary")


def plot_shared_subject_comparison() -> dict[str, Path]:
    pairs = [
        ("random_agg", FIGURES / "min_nn_results" / "shared_random_agg", FIGURES / "min_nn_results" / "subject_random_agg"),
        (
            "cluster_k5_agg",
            FIGURES / "min_nn_results" / "shared_cluster_k5_agg",
            FIGURES / "min_nn_results" / "subject_cluster_k5_agg",
        ),
        ("tau", FIGURES / "min_nn_results" / "shared_tau", FIGURES / "min_nn_results" / "subject_tau"),
        ("ood", FIGURES / "min_nn_results" / "shared_ood", FIGURES / "min_nn_results" / "subject_ood"),
    ]
    rows = []
    for split, shared_path, subject_path in pairs:
        shared = read_scores(shared_path)[["subject", "model", "eval_type", "test_score"]].rename(
            columns={"test_score": "shared_score"}
        )
        subject = read_scores(subject_path)[["subject", "model", "eval_type", "test_score"]].rename(
            columns={"test_score": "subject_score"}
        )
        merged = shared.merge(subject, on=["subject", "model", "eval_type"], how="inner")
        merged["delta"] = merged["subject_score"] - merged["shared_score"]
        model_delta = (
            merged.groupby(["model", "eval_type"], observed=True)["delta"]
            .mean()
            .reset_index()
        )
        model_delta["split"] = split
        rows.append(model_delta)
    df = pd.concat(rows, ignore_index=True)
    rng = np.random.default_rng(20260508)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
    split_order = [p[0] for p in pairs]
    y = np.arange(len(split_order))[::-1]
    y_lookup = dict(zip(split_order, y))
    max_abs = max(0.03, float(np.nanmax(np.abs(df["delta"]))) * 1.08)
    for ax, metric in zip(axes, METRIC_ORDER):
        sub = df[df["eval_type"] == metric]
        color = METRIC_COLORS[metric]
        for split in split_order:
            vals = sub[sub["split"] == split]["delta"].dropna().to_numpy(float)
            yi = y_lookup[split]
            jitter = rng.normal(0, 0.045, size=len(vals))
            ax.scatter(vals, yi + jitter, s=8, color="#8f8f8f", alpha=0.30, linewidth=0, rasterized=True)
            lo, hi = bootstrap_ci(vals, rng)
            mean = float(np.mean(vals))
            ax.plot([lo, hi], [yi, yi], color=color, linewidth=1.5)
            ax.scatter(mean, yi, s=30, color=color, edgecolor="white", linewidth=0.5, zorder=4)
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_title(METRIC_LABELS[metric], loc="left", fontweight="bold")
        ax.set_xlabel("subject pool - shared pool")
        ax.set_xlim(-max_abs, max_abs)
        clean_axis(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(split_order)
    axes[0].set_ylabel("split family")
    fig.suptitle("Paired model-level effect of adding subject-unique images", x=0.02, ha="left", fontweight="bold")
    fig.subplots_adjust(wspace=0.18)
    return save_figure(fig, "shared_vs_subject_pool")


def write_report_figures() -> dict[str, dict[str, Path]]:
    set_style()
    paths: dict[str, dict[str, Path]] = {
        "overview": plot_overview_scores(),
        "shared_vs_subject": plot_shared_subject_comparison(),
    }
    for split in SPLITS:
        paths[f"{split.key}_score"] = plot_score_summary(split)
        paths[f"{split.key}_effects"] = plot_effects(split)
        paths[f"{split.key}_taskonomy"] = plot_taskonomy_endpoints(split)
    return paths


def add_image(lines: list[str], title: str, paths: dict[str, Path], alt: str, width: int = 900) -> None:
    lines.extend(
        [
            f"**{title}** ([PDF]({rel_from_report(paths['pdf'])}))",
            "",
            f'<img src="{rel_from_report(paths["png"])}" alt="{alt}" width="{width}">',
            "",
        ]
    )


def add_conwell_figures(lines: list[str], variant: Variant) -> None:
    figs = (
        ("Fig. 2: architecture variation", "fig2_architecture_variation.png"),
        ("Fig. 3: task variation", "fig3_task_variation.png"),
        ("Fig. 4: input variation", "fig4_input_variation.png"),
    )
    for caption, filename in figs:
        png = variant.path / filename
        pdf = variant.path / filename.replace(".png", ".pdf")
        lines.extend(
            [
                f"**{caption}** ([PDF]({rel_from_report(pdf)}))",
                "",
                f'<img src="{rel_from_report(png)}" alt="{variant.label} {caption}" width="900">',
                "",
            ]
        )


def overview_display_rows() -> list[dict[str, object]]:
    rows = []
    for row in overview_rows():
        rows.append(
            {
                "split": row["split"],
                "variant": row["variant"],
                "metric": row["metric"],
                "mean": fmt_num(row["mean"], 3),
                "median": fmt_num(row["median"], 3),
                "min": fmt_num(row["min"], 3),
                "max": fmt_num(row["max"], 3),
                "subjects": row["subjects"],
                "models": row["models"],
            }
        )
    return rows


def add_appendix_tables(lines: list[str]) -> None:
    lines.extend(["---", "", "## Appendix: Detailed Tables", "", "### Overall Score Summary", ""])
    lines.extend(
        md_table(
            overview_display_rows(),
            [
                ("split", "split"),
                ("variant", "variant"),
                ("metric", "metric"),
                ("mean", "mean"),
                ("median", "median"),
                ("min", "min"),
                ("max", "max"),
                ("subjects", "subjects"),
                ("models", "models"),
            ],
        )
    )
    for split in SPLITS:
        lines.extend(["", f"### {split.title}", "", "#### Score Summary", ""])
        lines.extend(
            md_table(
                score_summary(split.variants),
                [
                    ("variant", "variant"),
                    ("metric", "metric"),
                    ("n", "n"),
                    ("subjects", "subjects"),
                    ("models", "models"),
                    ("mean", "mean"),
                    ("median", "median"),
                    ("min", "min"),
                    ("max", "max"),
                ],
            )
        )
        lines.extend(["", "#### Controlled-Effect Comparison", ""])
        lines.extend(
            md_table(
                effect_table(split.variants),
                [
                    ("family", "family"),
                    ("effect", "effect"),
                    ("metric", "metric"),
                    ("Conwell paper", "conwell"),
                    *[(variant.label, variant.label) for variant in split.variants],
                ],
            )
        )
        lines.extend(["", "#### Taskonomy Endpoints", ""])
        lines.extend(
            md_table(
                taskonomy_table(split.variants),
                [
                    ("task", "task"),
                    ("metric", "metric"),
                    ("Conwell paper", "conwell"),
                    *[(variant.label, variant.label) for variant in split.variants],
                ],
            )
        )


def build_report() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_support_tables()
    report_figs = write_report_figures()
    lines: list[str] = [
        "# LAION-fMRI Conwell Fig. 2-4 Replication Report",
        "",
        "This report summarizes the completed OTC analyses for the Conwell-style controlled comparisons: architecture variation (Fig. 2), task variation (Fig. 3), and input-diet variation (Fig. 4). The internal metric named `wrsa` is the voxel-encoding RSA metric corresponding most closely to Conwell's veRSA, so it is labeled here as wRSA/veRSA.",
        "",
        "The LAION-fMRI analyses use 5 subjects and 117 plotted models after dropping the known `efficientnet_b1_classification` coverage issue. Conwell's paper used the NSD shared-image setting with 4 subjects and a larger model survey, so the most important comparison is the direction and relative size of controlled effects rather than exact equality of absolute scores.",
        "",
        "## Supporting Tables",
        "",
        f"- [Score summary CSV]({rel_from_report(REPORTS / 'laion_fmri_conwell_fig2_4_score_summary.csv')})",
        f"- [Effect comparison CSV]({rel_from_report(REPORTS / 'laion_fmri_conwell_fig2_4_effects.csv')})",
        f"- [Taskonomy endpoint CSV]({rel_from_report(REPORTS / 'laion_fmri_conwell_fig2_4_taskonomy_endpoints.csv')})",
        "",
        "## Table-Derived Summary Figures",
        "",
    ]
    add_image(lines, "Overall score summary", report_figs["overview"], "Overall score summary")
    add_image(
        lines,
        "Shared-image versus subject-image pool comparison",
        report_figs["shared_vs_subject"],
        "Shared versus subject image pool paired differences",
    )
    lines.extend(
        [
            "## Initial Reading",
            "",
            "- Absolute LAION-fMRI wRSA/veRSA scores are generally below the headline Conwell NSD values, especially for split-half, cluster, and OOD tests. The strongest LAION-fMRI aggregates are random and tau splits, while cluster and OOD splits are visibly harder.",
            "- The cRSA architecture effect consistently preserves the Conwell direction: transformers are below CNNs. In wRSA/veRSA, however, the LAION-fMRI estimates often flip slightly positive for transformers, so the architecture conclusion is not a literal replication in the voxel-encoding metric.",
            "- The contrastive self-supervised advantage is robust in the random, cluster, tau, and plain split-half analyses, broadly matching Conwell's qualitative result.",
            "- The SLIP comparison mostly matches Conwell's qualitative pattern in the split variants where CLIP is below SimCLR and SLIP is close to SimCLR. The OOD-inclusive split-half result should be read more cautiously because its score scale differs from the other split families.",
            "- The ImageNet21K comparison remains small, consistent with Conwell's conclusion that the larger ImageNet21K diet does not clearly improve OTC predictivity. Some LAION-fMRI wRSA/veRSA estimates are slightly negative.",
            "- The IPCL input-diet result is only partial: Places and VGGFace2 are often below ImageNet in wRSA/veRSA, but the cRSA VGGFace2 effect is not consistently Conwell-like and OOD results can invert. This is the main controlled input-diet comparison to inspect manually in Fig. 4.",
            "",
            "## Caveats",
            "",
            "- These analyses use the current OTC ROI and the plotted 117-model subset. Conwell's paper reports NSD OTC results over a larger model survey and 4 subjects.",
            "- OOD sections use the aggregate `ood_type=all` rows from the current evaluator. They do not show the nine OOD types separately.",
            "- The language-alignment table compares CLIP and SLIP to SimCLR using the implemented Conwell-style fixed-effect test. It does not include the model-size interaction discussed in the paper narrative.",
            "- SRPR is included descriptively because it is useful for this project, but it is not one of the Fig. 2-4 Conwell paper metrics.",
            "",
            "## Conwell Reference Statistics Used Here",
            "",
            "- Architecture: transformers were slightly below CNNs in Conwell (cRSA beta -0.04; veRSA beta -0.01).",
            "- Taskonomy: Conwell reported low autoencoding scores and higher object-classification scores, with Taskonomy still below ImageNet-trained ResNet50.",
            "- Self-supervised learning: Conwell reported an instance-level contrastive advantage of about 0.06 in cRSA and 0.09 in veRSA.",
            "- SLIP: Conwell reported CLIP below SimCLR by about 0.05 in cRSA and 0.02 in veRSA, with no substantial SLIP effect.",
            "- Input diet: Conwell reported little/no ImageNet21K advantage over ImageNet1K, and IPCL diets ordered roughly ImageNet > OpenImages > Places > VGGFace2.",
            "",
            "The per-section summary figures below visualize the tables that are printed in full in the appendix.",
            "",
        ]
    )

    for split in SPLITS:
        lines.extend(["---", "", f"## {split.title}", "", split.description, ""])
        lines.extend(["### Table-Derived Figures", ""])
        add_image(lines, "Score summary", report_figs[f"{split.key}_score"], f"{split.key} score summary")
        add_image(lines, "Controlled-effect comparison", report_figs[f"{split.key}_effects"], f"{split.key} effects")
        add_image(
            lines,
            "Taskonomy endpoint comparison",
            report_figs[f"{split.key}_taskonomy"],
            f"{split.key} Taskonomy endpoints",
        )
        lines.extend(["### Conwell-Style Model Figures", ""])
        for variant in split.variants:
            lines.extend([f"#### {variant.label}", ""])
            add_conwell_figures(lines, variant)

    add_appendix_tables(lines)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    build_report()
