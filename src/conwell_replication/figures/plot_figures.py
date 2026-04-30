#!/usr/bin/env python3
"""Reproduce main figures from Conwell et al. (2024) replication.

Generates:
  fig1_controlled_comparisons.pdf/png  – Combined architecture / task / input (was figs 2-4)
  fig2_overall.pdf/png                 – All models ranked + scatter (was fig 5)
  fig3_model_comparison.pdf/png        – Model-to-model similarity (was fig 6)

Style: Nature Neuroscience (7 pt, Helvetica, thin spines, panel labels).
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent
_PKG_RESOURCES = Path(__file__).resolve().parents[3] / "resources"

# ---------------------------------------------------------------------------
# Nature Neuroscience styling
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_CRSA = "#c0c0c0"          # gray for cRSA boxes
C_WRSA = "#4a90d9"          # blue for veRSA/WRSA boxes
C_NC = "#d0d0d0"            # noise ceiling band
C_CNN = "#f28e2b"           # orange for CNNs
C_TRANSFORMER = "#4e79a7"   # blue for transformers

PANEL_LABELS = list("abcdefghijklmnop")


# ===========================================================================
# Data loading
# ===========================================================================

def load_data():
    """Load all prepared data."""
    scores = pd.read_csv(RESULTS_DIR / "best_layer_scores.csv")
    meta_path = RESULTS_DIR / "model_metadata.csv"
    if not meta_path.exists():
        meta_path = _PKG_RESOURCES / "model_metadata.csv"
    meta = pd.read_csv(meta_path)
    nc = pd.read_csv(RESULTS_DIR / "noise_ceilings.csv")
    return scores, meta, nc


def get_model_stats(scores, eval_type, models=None):
    """Get per-model mean + CI across subjects."""
    sub = scores[scores["eval_type"] == eval_type]
    if models is not None:
        sub = sub[sub["model"].isin(models)]

    stats_list = []
    for model, grp in sub.groupby("model"):
        vals = grp["test_score"].values
        mean = vals.mean()
        n_boot = 1000
        boot_means = np.array([
            np.random.choice(vals, size=len(vals), replace=True).mean()
            for _ in range(n_boot)
        ])
        ci_lo = np.percentile(boot_means, 2.5)
        ci_hi = np.percentile(boot_means, 97.5)
        stats_list.append({
            "model": model,
            "mean": mean,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })
    return pd.DataFrame(stats_list)


# ===========================================================================
# Drawing helpers
# ===========================================================================

def draw_noise_ceiling(ax, nc_df, alpha=0.12):
    """Draw noise ceiling band (mean +/- range across subjects)."""
    nc_vals = nc_df["nc_pearson"].values
    nc_mean = nc_vals.mean()
    nc_lo = nc_vals.min()
    nc_hi = nc_vals.max()
    ax.axhspan(nc_lo, nc_hi, color=C_NC, alpha=alpha, zorder=0, linewidth=0)
    ax.axhline(nc_mean, color="gray", linewidth=0.4, linestyle="--", alpha=0.4, zorder=0)
    return nc_mean


def draw_model_boxes(ax, model_stats, x_positions, eval_type, width=0.7,
                     sort_by_mean=True, color=None):
    """Draw individual model boxes (mean line inside CI box)."""
    if sort_by_mean:
        model_stats = model_stats.sort_values("mean", ascending=False).reset_index(drop=True)

    is_crsa = eval_type == "crsa"
    face_color = "white" if is_crsa else (color or C_WRSA)
    edge_color = "#999999" if is_crsa else (color or C_WRSA)
    alpha_val = 0.75 if not is_crsa else 0.6
    lw = 0.4

    for i, (_, row) in enumerate(model_stats.iterrows()):
        x = x_positions[i] if isinstance(x_positions, (list, np.ndarray)) else x_positions + i
        ci_height = row["ci_hi"] - row["ci_lo"]
        if ci_height < 1e-4:
            ci_height = 0.003
        rect = mpatches.FancyBboxPatch(
            (x - width / 2, row["ci_lo"]),
            width, ci_height,
            boxstyle="square,pad=0",
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=lw,
            alpha=alpha_val,
            zorder=2 if is_crsa else 3,
        )
        ax.add_patch(rect)
        ml_color = "black" if is_crsa else "white"
        ax.hlines(row["mean"], x - width / 2, x + width / 2,
                  colors=ml_color, linewidth=0.4, zorder=4)

    return model_stats


def draw_group_ribbon(ax, model_stats, x_center, width, eval_type, color=None):
    """Draw group mean ribbon (hatched)."""
    vals = model_stats["mean"].values
    n_boot = 1000
    boot_means = np.array([
        np.random.choice(vals, size=len(vals), replace=True).mean()
        for _ in range(n_boot)
    ])
    group_mean = vals.mean()
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)

    is_crsa = eval_type == "crsa"
    fc = "white" if is_crsa else (color or C_WRSA)
    ec = "#999999" if is_crsa else (color or C_WRSA)

    rect = mpatches.FancyBboxPatch(
        (x_center - width / 2, ci_lo),
        width, ci_hi - ci_lo,
        boxstyle="square,pad=0",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.8,
        alpha=0.35,
        hatch="///",
        zorder=5,
    )
    ax.add_patch(rect)
    ax.hlines(group_mean, x_center - width / 2, x_center + width / 2,
              colors=ec, linewidth=1.0, zorder=6)

    return group_mean, ci_lo, ci_hi


def add_panel_label(ax, label, x=-0.06, y=1.10):
    """Add bold panel label (a, b, c, ...) to axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")


# ===========================================================================
# FIGURE 1: Combined controlled comparisons (was figs 2-4)
#   Row 1: Architecture (wide)
#   Row 2: Taskonomy | VISSL | SLIP
#   Row 3: IN1K vs IN21K | IPCL
# ===========================================================================

def _panel_architecture(ax, scores, meta, nc):
    """Panel a: CNN vs Transformer."""
    cnn_models = meta[meta["comparison_group"] == "arch_cnn_in1k"]["model"].tolist()
    trans_models = meta[meta["comparison_group"] == "arch_transformer_in1k"]["model"].tolist()

    nc_mean = draw_noise_ceiling(ax, nc)

    gap = 3
    cnn_offset = 0
    trans_offset = len(cnn_models) + gap

    for eval_type in ["crsa", "wrsa"]:
        cnn_stats = get_model_stats(scores, eval_type, cnn_models)
        trans_stats = get_model_stats(scores, eval_type, trans_models)

        if eval_type == "crsa":
            wrsa_cnn = get_model_stats(scores, "wrsa", cnn_models)
            wrsa_trans = get_model_stats(scores, "wrsa", trans_models)
            cnn_order = wrsa_cnn.sort_values("mean")["model"].tolist()
            trans_order = wrsa_trans.sort_values("mean")["model"].tolist()

        cnn_stats = cnn_stats.set_index("model").loc[
            [m for m in cnn_order if m in cnn_stats["model"].values]
        ].reset_index()
        trans_stats = trans_stats.set_index("model").loc[
            [m for m in trans_order if m in trans_stats["model"].values]
        ].reset_index()

        cnn_x = np.arange(len(cnn_stats)) + cnn_offset
        trans_x = np.arange(len(trans_stats)) + trans_offset

        draw_model_boxes(ax, cnn_stats, cnn_x, eval_type, width=0.7,
                         sort_by_mean=False, color=C_CNN)
        draw_model_boxes(ax, trans_stats, trans_x, eval_type, width=0.7,
                         sort_by_mean=False, color=C_TRANSFORMER)

        draw_group_ribbon(ax, cnn_stats, cnn_x.mean(),
                          len(cnn_stats) * 0.9, eval_type, color=C_CNN)
        draw_group_ribbon(ax, trans_stats, trans_x.mean(),
                          len(trans_stats) * 0.9, eval_type, color=C_TRANSFORMER)

    # Group labels — above noise ceiling
    cnn_center = (len(cnn_models) - 1) / 2 + cnn_offset
    trans_center = (len(trans_models) - 1) / 2 + trans_offset
    label_y = nc_mean + 0.12  # above NC band
    ax.text(cnn_center, label_y, f"Convolutional (n={len(cnn_models)})",
            ha="center", fontsize=6, fontweight="bold", color="#555555")
    ax.text(trans_center, label_y, f"Transformer (n={len(trans_models)})",
            ha="center", fontsize=6, fontweight="bold", color="#555555")

    ax.set_ylabel("Brain Predictivity ($r$)")
    ax.set_title("Architecture Variation — ImageNet-1K Classification",
                 fontweight="bold", pad=4)
    ax.set_xlim(-1, trans_offset + len(trans_models))
    ax.set_ylim(0, nc_mean + 0.18)

    # X-tick labels (model names)
    all_x = list(np.arange(len(cnn_order)) + cnn_offset) + \
            list(np.arange(len(trans_order)) + trans_offset)
    all_labels = [m.replace("_classification", "").replace("_in1k", "")
                  for m in cnn_order + trans_order]
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=55, ha="right", fontsize=2.5)

    # Stats annotation
    crsa_cnn = get_model_stats(scores, "crsa", cnn_models)["mean"]
    crsa_trans = get_model_stats(scores, "crsa", trans_models)["mean"]
    wrsa_cnn_v = get_model_stats(scores, "wrsa", cnn_models)["mean"]
    wrsa_trans_v = get_model_stats(scores, "wrsa", trans_models)["mean"]
    stat_text = (
        f"cRSA: CNN={crsa_cnn.mean():.3f}, Trans={crsa_trans.mean():.3f}, "
        f"\u0394={crsa_cnn.mean()-crsa_trans.mean():.3f}\n"
        f"WRSA: CNN={wrsa_cnn_v.mean():.3f}, Trans={wrsa_trans_v.mean():.3f}, "
        f"\u0394={wrsa_cnn_v.mean()-wrsa_trans_v.mean():.3f}"
    )
    ax.text(0.02, 0.04, stat_text, transform=ax.transAxes,
            fontsize=5, verticalalignment="bottom",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.4, alpha=0.9))


def _panel_task_ungrouped(ax, scores, nc, models, title, label_func=None):
    """Simple task panel: all models sorted by wrsa, no sub-groups."""
    nc_mean = draw_noise_ceiling(ax, nc)

    for eval_type in ["crsa", "wrsa"]:
        stats = get_model_stats(scores, eval_type, models)
        if eval_type == "crsa":
            wrsa_order = (get_model_stats(scores, "wrsa", models)
                          .sort_values("mean")["model"].tolist())
        available = set(stats["model"].values)
        stats = stats.set_index("model").loc[
            [m for m in wrsa_order if m in available]
        ].reset_index()

        x = np.arange(len(stats))
        draw_model_boxes(ax, stats, x, eval_type, width=0.7,
                         sort_by_mean=False)

        # Overall group ribbon
        if len(stats) > 1:
            x_center = (len(stats) - 1) / 2
            ribbon_w = len(stats) * 0.9
            draw_group_ribbon(ax, stats, x_center, ribbon_w, eval_type)

    ax.set_xticks(np.arange(len(wrsa_order)))
    labels = [label_func(m) if label_func else m for m in wrsa_order]
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=4.5)
    ax.set_title(title, fontweight="bold", pad=4)

    # Data-driven ylim
    all_y = []
    for et in ["crsa", "wrsa"]:
        s = get_model_stats(scores, et, models)
        if len(s) > 0:
            all_y.append(s["ci_hi"].max())
    ax.set_ylim(-0.02, max(all_y) * 1.15 if all_y else 0.5)


def _panel_task_grouped(ax, scores, meta, nc, comparison_group, title,
                        groups, label_func=None, group_colors=None):
    """Task panel with model groupings and group-mean ribbons.

    Parameters
    ----------
    groups : dict[str, list[str]]
        {group_label: [model_names, ...]}  — plotted left-to-right with gaps.
    group_colors : dict[str, str] or None
        Optional per-group color for veRSA boxes. Falls back to C_WRSA.
    """
    all_models = meta[meta["comparison_group"] == comparison_group]["model"].tolist()
    nc_mean = draw_noise_ceiling(ax, nc)

    gap = 1.0  # gap between groups
    cursor = 0.0
    group_spans = {}  # {group_label: (x_start, x_end, [model_names_ordered])}

    # Pre-compute wrsa order for all models at once
    wrsa_all = get_model_stats(scores, "wrsa", all_models)
    wrsa_rank = wrsa_all.set_index("model")["mean"]

    # Sort groups by their mean WRSA score
    def _group_mean(item):
        label, model_list = item
        present = [m for m in model_list if m in all_models]
        if not present:
            return -999
        return np.mean([wrsa_rank.get(m, 0) for m in present])

    groups_sorted = sorted(groups.items(), key=_group_mean)

    for gi, (group_label, group_models) in enumerate(groups_sorted):
        present = [m for m in group_models if m in all_models]
        if not present:
            continue
        # Sort within group by wrsa
        present_sorted = sorted(present,
                                key=lambda m: wrsa_rank.get(m, -999))
        n = len(present_sorted)
        x_start = cursor
        x_end = cursor + n - 1
        group_spans[group_label] = (x_start, x_end, present_sorted)
        cursor += n + gap

    # Draw boxes and group ribbons
    for eval_type in ["crsa", "wrsa"]:
        for group_label, (x_start, x_end, models_ordered) in group_spans.items():
            stats = get_model_stats(scores, eval_type, models_ordered)
            available = set(stats["model"].values)
            stats = stats.set_index("model").loc[
                [m for m in models_ordered if m in available]
            ].reset_index()

            x_positions = np.arange(len(stats)) + x_start
            gc = (group_colors or {}).get(group_label, None)
            draw_model_boxes(ax, stats, x_positions, eval_type, width=0.7,
                             sort_by_mean=False, color=gc)

            # Group ribbon
            if len(stats) > 1:
                x_center = (x_start + x_end) / 2
                ribbon_w = (x_end - x_start + 1) * 0.9
                draw_group_ribbon(ax, stats, x_center, ribbon_w,
                                  eval_type, color=gc)

    # X-tick labels (individual models)
    all_ticks = []
    all_labels = []
    for group_label, (x_start, x_end, models_ordered) in group_spans.items():
        for i, m in enumerate(models_ordered):
            all_ticks.append(x_start + i)
            lbl = label_func(m) if label_func else m
            all_labels.append(lbl)

    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, rotation=55, ha="right", fontsize=4.5)

    ax.set_title(title, fontweight="bold", pad=4)
    total_x = cursor - gap  # remove trailing gap
    ax.set_xlim(-0.7, total_x + 0.7)

    # Y-limit: based on actual data range (NC band may extend above)
    all_y_vals = []
    for eval_type in ["crsa", "wrsa"]:
        stats = get_model_stats(scores, eval_type, all_models)
        if len(stats) > 0:
            all_y_vals.append(stats["ci_hi"].max())
    y_data_max = max(all_y_vals) if all_y_vals else 0.5
    y_top = y_data_max * 1.25  # extra room for group labels
    ax.set_ylim(-0.02, y_top)

    # Group labels — above the data
    label_y = y_data_max * 1.10
    for group_label, (x_start, x_end, models_ordered) in group_spans.items():
        center = (x_start + x_end) / 2
        ax.text(center, label_y, group_label,
                ha="center", fontsize=5, fontweight="bold", color="#555555")


def _panel_input_size(ax, scores, meta, nc):
    """Panel e: IN1K vs IN21K."""
    in_models = meta[meta["comparison_group"] == "input_in1k_vs_in21k"]["model"].tolist()
    nc_mean = draw_noise_ceiling(ax, nc)

    in1k_models = [m for m in in_models
                   if "in21k" not in m and "in22k" not in m and "in22" not in m]
    in21k_models = [m for m in in_models
                    if "in21k" in m or "in22k" in m or "in22" in m]

    for eval_type in ["crsa", "wrsa"]:
        in1k_stats = get_model_stats(scores, eval_type, in1k_models)
        in21k_stats = get_model_stats(scores, eval_type, in21k_models)

        if eval_type == "crsa":
            in1k_order = (get_model_stats(scores, "wrsa", in1k_models)
                          .sort_values("mean")["model"].tolist())
            in21k_order = (get_model_stats(scores, "wrsa", in21k_models)
                           .sort_values("mean")["model"].tolist())

        in1k_stats = in1k_stats.set_index("model").loc[
            [m for m in in1k_order if m in in1k_stats["model"].values]
        ].reset_index()
        in21k_stats = in21k_stats.set_index("model").loc[
            [m for m in in21k_order if m in in21k_stats["model"].values]
        ].reset_index()

        gap = 2
        x_1k = np.arange(len(in1k_stats))
        x_21k = np.arange(len(in21k_stats)) + len(in1k_stats) + gap

        c1 = "#e45756" if eval_type == "wrsa" else None
        c2 = "#54a24b" if eval_type == "wrsa" else None
        draw_model_boxes(ax, in1k_stats, x_1k, eval_type, width=0.7,
                         sort_by_mean=False, color=c1)
        draw_model_boxes(ax, in21k_stats, x_21k, eval_type, width=0.7,
                         sort_by_mean=False, color=c2)

        draw_group_ribbon(ax, in1k_stats, x_1k.mean(),
                          len(in1k_stats) * 0.9, eval_type, color=c1)
        draw_group_ribbon(ax, in21k_stats, x_21k.mean(),
                          len(in21k_stats) * 0.9, eval_type, color=c2)

    center_1k = (len(in1k_models) - 1) / 2
    center_21k = (len(in21k_models) - 1) / 2 + len(in1k_models) + 2
    label_y = nc_mean + 0.12
    ax.text(center_1k, label_y, f"IN-1K (n={len(in1k_models)})",
            ha="center", fontsize=5.5, fontweight="bold", color="#555555")
    ax.text(center_21k, label_y, f"IN-21K (n={len(in21k_models)})",
            ha="center", fontsize=5.5, fontweight="bold", color="#555555")

    ax.set_title("Training Dataset Size", fontweight="bold", pad=4)
    ax.set_ylim(0, nc_mean + 0.18)

    # X-tick labels
    all_x = list(x_1k) + list(x_21k)
    all_labels = [m.replace("_classification", "").replace("_in1k", "")
                  for m in in1k_order + in21k_order]
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=55, ha="right", fontsize=4)


def _panel_ipcl(ax, scores, meta, nc):
    """Panel f: IPCL domains."""
    ipcl_models = meta[meta["comparison_group"] == "input_ipcl"]["model"].tolist()
    nc_mean = draw_noise_ceiling(ax, nc)

    domain_colors = {
        "alexnet_gn_ipcl_imagenet": "#4e79a7",
        "alexnet_gn_ipcl_openimages": "#f28e2b",
        "alexnet_gn_ipcl_places256": "#76b7b2",
        "alexnet_gn_ipcl_vggface2": "#e15759",
    }
    domain_labels = {
        "alexnet_gn_ipcl_imagenet": "ImageNet",
        "alexnet_gn_ipcl_openimages": "OpenImages",
        "alexnet_gn_ipcl_places256": "Places",
        "alexnet_gn_ipcl_vggface2": "VGGFace2",
    }
    ipcl_order = [m for m in [
        "alexnet_gn_ipcl_imagenet", "alexnet_gn_ipcl_openimages",
        "alexnet_gn_ipcl_places256", "alexnet_gn_ipcl_vggface2",
    ] if m in ipcl_models]

    for eval_type in ["crsa", "wrsa"]:
        ipcl_stats = get_model_stats(scores, eval_type, ipcl_order)
        ipcl_stats = ipcl_stats.set_index("model").loc[ipcl_order].reset_index()

        for i, (_, row) in enumerate(ipcl_stats.iterrows()):
            c = domain_colors.get(row["model"], C_WRSA)
            is_crsa = eval_type == "crsa"
            fc = "white" if is_crsa else c
            ec = "#999999" if is_crsa else c

            ci_height = max(row["ci_hi"] - row["ci_lo"], 0.003)
            rect = mpatches.FancyBboxPatch(
                (i - 0.35, row["ci_lo"]), 0.7, ci_height,
                boxstyle="square,pad=0",
                facecolor=fc, edgecolor=ec,
                linewidth=0.4, alpha=0.75,
                zorder=2 if is_crsa else 3,
            )
            ax.add_patch(rect)
            ml_color = "black" if is_crsa else "white"
            ax.hlines(row["mean"], i - 0.35, i + 0.35,
                      colors=ml_color, linewidth=0.4, zorder=4)

    ax.set_xticks(range(len(ipcl_order)))
    ax.set_xticklabels([domain_labels.get(m, m) for m in ipcl_order],
                       rotation=45, ha="right", fontsize=5.5)
    ax.set_title("Training Domain (IPCL)", fontweight="bold", pad=4)
    ax.set_ylim(-0.05, nc_mean + 0.15)


def figure1(scores, meta, nc):
    """Combined controlled comparisons figure.

    Layout (3 rows, widths proportional to model counts):
      Row 1: Architecture (152 models, full width)               — a
      Row 2: Taskonomy (21) | VISSL (14) | SLIP (14)             — b c d
      Row 3: IN1K vs IN21K (34+gap) | IPCL (4)                   — e f
    """
    fig = plt.figure(figsize=(10, 8.5))

    # ── Margins and gaps ──
    left = 0.06
    right = 0.97
    total_w = right - left
    row_gap = 0.12          # vertical gap between rows (rotated labels + titles below)
    col_gap = 0.05          # horizontal gap between panels

    # All rows same height
    h = 0.20
    # Vertical positions (bottom of each row)
    y3 = 0.06
    y2 = y3 + h + row_gap
    y1 = y2 + h + row_gap

    # ── Row 1: Architecture (full width) ──
    ax_arch = fig.add_axes([left, y1, total_w, h])
    _panel_architecture(ax_arch, scores, meta, nc)
    add_panel_label(ax_arch, "a")

    # ── Row 2: Taskonomy (21) | VISSL (14) | SLIP (14) ──
    # Width proportional to model count, with equal gaps
    n_task, n_vissl, n_slip = 21, 14, 14
    total_models_r2 = n_task + n_vissl + n_slip
    usable_w_r2 = total_w - 2 * col_gap
    w_task = usable_w_r2 * n_task / total_models_r2
    w_vissl = usable_w_r2 * n_vissl / total_models_r2
    w_slip = usable_w_r2 * n_slip / total_models_r2

    x_task = left
    x_vissl = x_task + w_task + col_gap
    x_slip = x_vissl + w_vissl + col_gap

    # --- Taskonomy (no grouping — single set sorted by wrsa) ---
    task_models = meta[meta["comparison_group"] == "task_taskonomy"]["model"].tolist()
    ax_task = fig.add_axes([x_task, y2, w_task, h])
    _panel_task_ungrouped(ax_task, scores, nc, task_models,
                          "Task Variation\n(ResNet50, Indoor Scenes)",
                          label_func=lambda m: m.replace("_taskonomy", ""))
    ax_task.set_ylabel("Brain Predictivity ($r$)")
    add_panel_label(ax_task, "b")

    # --- VISSL groupings (supervised reference removed) ---
    vissl_groups = {
        "Contrastive": [
            "vissl_resnet50_barlowtwins",
            "vissl_resnet50_mocov2",
            "vissl_resnet50_simclr",
            "vissl_resnet50_pirl",
            "vissl_resnet50_npid",
            "ResNet50/SwAV/BS4096/2x224+6x96_selfsupervised",
            "ResNet50/DeepClusterV2/2x224+6x96_selfsupervised",
        ],
        "Non-Contrastive": [
            "vissl_resnet50_deepclusterv2",
            "vissl_resnet50_swav",
            "vissl_resnet50_rotnet",
            "vissl_resnet50_jigsaw_goyal19",
            "ResNet50/JigSaw/P100_selfsupervised",
            "ResNet50/ClusterFit/16K/RotNet_selfsupervised",
        ],
    }
    vissl_colors = {
        "Contrastive": "#e8853a",      # orange
        "Non-Contrastive": "#7b4fad",  # purple
    }
    ax_vissl = fig.add_axes([x_vissl, y2, w_vissl, h])
    _panel_task_grouped(ax_vissl, scores, meta, nc, "task_vissl",
                        "Contrastive vs Non-Contrastive\n(ResNet50, ImageNet-1K)",
                        vissl_groups, group_colors=vissl_colors,
                        label_func=lambda m: (m.replace("vissl_resnet50_", "")
                                               .replace("ResNet50/", "")
                                               .replace("_selfsupervised", "")))
    add_panel_label(ax_vissl, "c")

    # --- SLIP groupings ---
    slip_groups = {
        "SimCLR": [
            "slip_vit_s_simclr", "slip_vit_b_simclr", "slip_vit_l_simclr",
        ],
        "CLIP": [
            "slip_vit_s_clip", "slip_vit_b_clip", "slip_vit_l_clip",
            "ViT/L/CLIP/CC12M_slip",
        ],
        "SLIP": [
            "slip_vit_s_slip", "slip_vit_b_slip", "slip_vit_l_slip",
            "ViT/S/SLIP/Ep100_slip", "ViT/B/SLIP/Ep100_slip",
            "ViT/L/SLIP/Ep100_slip", "ViT/L/SLIP/CC12M_slip",
        ],
    }
    slip_colors = {
        "SimCLR": "#e8853a",   # orange
        "CLIP": "#3b7dd8",     # blue
        "SLIP": "#4daf4a",     # green
    }
    ax_slip = fig.add_axes([x_slip, y2, w_slip, h])
    _panel_task_grouped(ax_slip, scores, meta, nc, "task_slip",
                        "Visual vs Language Alignment\n(ViT, YFCC15M)",
                        slip_groups, group_colors=slip_colors,
                        label_func=lambda m: (m.replace("slip_", "")
                                               .replace("ViT/", "")
                                               .replace("/SLIP/", " SLIP ")
                                               .replace("/CLIP/", " CLIP ")
                                               .replace("_slip", " SLIP")
                                               .replace("_clip", " CLIP")
                                               .replace("_simclr", " SimCLR")))
    add_panel_label(ax_slip, "d")

    # ── Row 3: IN1K vs IN21K | IPCL ──
    # Split ~60/40 — e narrower (boxes are dense), f wider (4 bars need room)
    w_input = total_w * 0.55
    w_ipcl = total_w - col_gap - w_input

    x_input = left
    x_ipcl = x_input + w_input + col_gap

    ax_input = fig.add_axes([x_input, y3, w_input, h])
    _panel_input_size(ax_input, scores, meta, nc)
    ax_input.set_ylabel("Brain Predictivity ($r$)")
    add_panel_label(ax_input, "e")

    ax_ipcl = fig.add_axes([x_ipcl, y3, w_ipcl, h])
    _panel_ipcl(ax_ipcl, scores, meta, nc)
    add_panel_label(ax_ipcl, "f")

    # ── Shared legend (top-right of architecture panel) ──
    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor="#999999",
                       linewidth=0.5, label="cRSA"),
        mpatches.Patch(facecolor=C_WRSA, edgecolor=C_WRSA,
                       linewidth=0.5, label="veRSA (WRSA)"),
        mpatches.Patch(facecolor=C_NC, alpha=0.2,
                       edgecolor="none", label="Noise ceiling"),
    ]
    ax_arch.legend(handles=legend_elements, loc="upper center",
                   frameon=True, framealpha=0.95, edgecolor="none",
                   ncol=3, columnspacing=0.8, handletextpad=0.3,
                   handlelength=1.2)

    fig.savefig(RESULTS_DIR / "fig1_controlled_comparisons.pdf")
    fig.savefig(RESULTS_DIR / "fig1_controlled_comparisons.png", dpi=300)
    plt.close(fig)
    print("Figure 1 saved (controlled comparisons).")


# ===========================================================================
# FIGURE 2: Overall Model Variation (was fig 5)
# ===========================================================================

def figure2(scores, meta, nc):
    """All models ranked + scatter analyses."""
    fig = plt.figure(figsize=(10, 6.0))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.45, wspace=0.4)

    nc_mean = nc["nc_pearson"].mean()

    # --- Panel a: All models ranked by WRSA ---
    ax = fig.add_subplot(gs[0, :])

    wrsa_model_means = (
        scores[scores["eval_type"] == "wrsa"]
        .groupby("model")["test_score"].mean()
    )
    rank_order = wrsa_model_means.sort_values(ascending=False).index.tolist()

    for eval_type, color, alpha, label in [
        ("crsa", "#b0b0b0", 0.4, "cRSA"),
        ("wrsa", C_WRSA, 0.7, "WRSA (veRSA)"),
    ]:
        model_means = (
            scores[scores["eval_type"] == eval_type]
            .groupby("model")["test_score"].mean()
        )
        model_means = model_means.reindex(rank_order)
        x = np.arange(len(model_means))
        ax.scatter(x, model_means.values, s=4, c=color, alpha=alpha,
                   edgecolors="none", zorder=2, label=label)

    # Noise ceiling
    nc_hi = nc["nc_pearson"].max()
    nc_lo = nc["nc_pearson"].min()
    ax.axhspan(nc_lo, nc_hi, color=C_NC, alpha=0.12, zorder=0, linewidth=0)
    ax.axhline(nc_mean, color="gray", linewidth=0.4, linestyle="--", alpha=0.4)

    # Label top 5 and bottom 3
    wrsa_ranked = wrsa_model_means.reindex(rank_order)
    for i in list(range(5)) + list(range(len(rank_order) - 3, len(rank_order))):
        model = rank_order[i]
        short = model.split("_")[0] if len(model) > 30 else model
        short = short[:25]
        y = wrsa_ranked.iloc[i]
        ax.annotate(short, (i, y), fontsize=4, rotation=45,
                    ha="left", va="bottom", alpha=0.6)

    ax.set_xlabel("Model rank (by WRSA)")
    ax.set_ylabel("Brain Predictivity ($r$)")
    ax.set_title("All Models Ranked by Brain Predictivity",
                 fontweight="bold", pad=4)
    ax.set_xlim(-5, len(rank_order) + 5)
    ax.set_ylim(-0.05, nc_hi + 0.1)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="none", markerscale=2)
    add_panel_label(ax, "a")

    # Class colors for scatter panels
    class_colors = {
        "Convolutional": C_CNN,
        "Transformer": C_TRANSFORMER,
        "Hybrid": "#59a14f",
        "MLP-Mixer": "#b07aa1",
    }

    # --- Panel b: cRSA vs WRSA scatter ---
    ax = fig.add_subplot(gs[1, 0])

    crsa_means = (scores[scores["eval_type"] == "crsa"]
                  .groupby("model")["test_score"].mean())
    wrsa_means = (scores[scores["eval_type"] == "wrsa"]
                  .groupby("model")["test_score"].mean())
    common = crsa_means.index.intersection(wrsa_means.index)
    x_vals = crsa_means[common].values
    y_vals = wrsa_means[common].values

    colors = []
    for m in common:
        mc = meta[meta["model"] == m]["model_class"].values
        colors.append(class_colors.get(mc[0], "gray") if len(mc) > 0 else "gray")

    ax.scatter(x_vals, y_vals, c=colors, s=6, alpha=0.5, edgecolors="none")
    ax.plot([0, 0.5], [0, 0.5], "k--", alpha=0.2, linewidth=0.4)

    r, p = stats.spearmanr(x_vals, y_vals)
    ax.text(0.05, 0.95, f"\u03c1 = {r:.3f}", transform=ax.transAxes,
            fontsize=6, verticalalignment="top")

    ax.set_xlabel("cRSA ($r$)")
    ax.set_ylabel("WRSA / veRSA ($r$)")
    ax.set_title("cRSA vs veRSA", fontweight="bold", pad=4)

    legend_elements = [mpatches.Patch(color=c, label=l)
                       for l, c in class_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=5,
              frameon=True, framealpha=0.95, edgecolor="none",
              handlelength=1.0, handletextpad=0.3)
    add_panel_label(ax, "b")

    # --- Panel c: Training objective groups ---
    ax = fig.add_subplot(gs[1, 1])

    train_groups = {
        "Classification": ["classification"],
        "CLIP/Language": ["clip", "siglip", "slip"],
        "Self-Supervised": ["selfsupervised", "bert_pretraining", "seer"],
        "Detection": ["detection", "segmentation", "panoptics"],
        "Taskonomy": ["taskonomy"],
        "Other": ["bit_expert", "big_transfer", "adversarial", "noisy_student",
                   "monoculardepth", "ipcl", "semi-supervised", "semi-weakly-supervised"],
    }

    group_scores = []
    for group_name, train_types in train_groups.items():
        group_models = meta[meta["train_type"].isin(train_types)]["model"].tolist()
        if not group_models:
            continue
        for et in ["crsa", "wrsa"]:
            ms = get_model_stats(scores, et, group_models)
            group_scores.append({
                "group": group_name,
                "eval_type": et,
                "mean": ms["mean"].mean(),
                "ci_lo": ms["mean"].quantile(0.025),
                "ci_hi": ms["mean"].quantile(0.975),
                "n": len(ms),
            })

    gs_df = pd.DataFrame(group_scores)
    groups_ordered = (gs_df[gs_df["eval_type"] == "wrsa"]
                      .sort_values("mean")["group"].tolist())
    x_pos = np.arange(len(groups_ordered))

    for eval_type, offset, color in [("crsa", -0.15, "#b0b0b0"),
                                      ("wrsa", 0.15, C_WRSA)]:
        sub_df = (gs_df[gs_df["eval_type"] == eval_type]
                  .set_index("group").loc[groups_ordered])
        ax.barh(x_pos + offset, sub_df["mean"].values, height=0.25,
                color=color, alpha=0.7, label=eval_type.upper(),
                linewidth=0)
        ax.errorbar(sub_df["mean"].values, x_pos + offset,
                    xerr=[sub_df["mean"].values - sub_df["ci_lo"].values,
                          sub_df["ci_hi"].values - sub_df["mean"].values],
                    fmt="none", ecolor="black", elinewidth=0.3, capsize=1.5)

    ax.set_yticks(x_pos)
    ax.set_yticklabels(
        [f"{g} (n={gs_df[(gs_df['group']==g) & (gs_df['eval_type']=='wrsa')]['n'].values[0]})"
         for g in groups_ordered], fontsize=5.5)
    ax.set_xlabel("Brain Predictivity ($r$)")
    ax.set_title("Training Objective Groups", fontweight="bold", pad=4)
    ax.legend(loc="lower right", fontsize=5, frameon=True,
              framealpha=0.95, edgecolor="none")
    add_panel_label(ax, "c")

    fig.savefig(RESULTS_DIR / "fig2_overall.pdf")
    fig.savefig(RESULTS_DIR / "fig2_overall.png", dpi=300)
    plt.close(fig)
    print("Figure 2 saved (overall).")


# ===========================================================================
# FIGURE 3: Model-to-Model Comparison (was fig 6)
# ===========================================================================

def _clean_model_name(name):
    """Strip common prefixes/suffixes for readable labels."""
    for prefix in ["timm_", "openclip_", "torchvision_", "vissl_resnet50_",
                    "slip_", "clip_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Shorten common fragments
    name = (name.replace("_clip_", " CLIP ")
                .replace("_dinov2", " DINOv2")
                .replace("_reg4_", " reg4 ")
                .replace("patch14_", "p14 ")
                .replace("patch16_", "p16 ")
                .replace("_448_m", "")
                .replace("_378", "")
                .replace("_clipa_dataco", " CLIPA"))
    return name[:35]


def _zoom_mds_panel(ax, embedding, models, model_classes,
                    class_colors_map, class_markers, title,
                    wrsa_means, top_n=3):
    """Draw an MDS panel zoomed to the main cluster, with outliers on the border.

    Strategy: compute the central 90% bounding box, zoom the axes there,
    then re-project outliers onto the border with small arrows pointing
    in their true direction.
    """
    from adjustText import adjust_text

    # Determine inlier range (central 90 % on each axis)
    pct_lo, pct_hi = 5, 95
    x_lo = np.percentile(embedding[:, 0], pct_lo)
    x_hi = np.percentile(embedding[:, 0], pct_hi)
    y_lo = np.percentile(embedding[:, 1], pct_lo)
    y_hi = np.percentile(embedding[:, 1], pct_hi)

    # Add a little padding (15 %)
    pad_x = (x_hi - x_lo) * 0.15
    pad_y = (y_hi - y_lo) * 0.15
    xlim = (x_lo - pad_x, x_hi + pad_x)
    ylim = (y_lo - pad_y, y_hi + pad_y)

    # Classify inliers / outliers
    inlier_mask = ((embedding[:, 0] >= xlim[0]) &
                   (embedding[:, 0] <= xlim[1]) &
                   (embedding[:, 1] >= ylim[0]) &
                   (embedding[:, 1] <= ylim[1]))

    # ── Plot inliers per class ──
    for cls_name, cls_color in class_colors_map.items():
        marker = class_markers[cls_name]
        idxs = [i for i in range(len(models))
                if model_classes[models[i]] == cls_name and inlier_mask[i]]
        if not idxs:
            continue
        ax.scatter(embedding[idxs, 0], embedding[idxs, 1],
                   c=cls_color, marker=marker, s=12, alpha=0.55,
                   edgecolors="white", linewidths=0.15,
                   label=cls_name, zorder=2)

    # ── Project outliers onto the border ──
    border_margin_x = (xlim[1] - xlim[0]) * 0.03
    border_margin_y = (ylim[1] - ylim[0]) * 0.03
    outlier_idxs = np.where(~inlier_mask)[0]

    for oi in outlier_idxs:
        true_x, true_y = embedding[oi]
        # Clamp to just inside the border
        bx = np.clip(true_x, xlim[0] + border_margin_x,
                      xlim[1] - border_margin_x)
        by = np.clip(true_y, ylim[0] + border_margin_y,
                      ylim[1] - border_margin_y)

        cls = model_classes[models[oi]]
        color = class_colors_map.get(cls, "gray")
        marker = class_markers.get(cls, "o")

        # Draw the point at the clamped position
        ax.scatter(bx, by, c=color, marker=marker, s=10, alpha=0.4,
                   edgecolors="black", linewidths=0.3, zorder=3)

        # Small arrow showing true direction
        dx = true_x - bx
        dy = true_y - by
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            arrow_len = min((xlim[1] - xlim[0]) * 0.04, length)
            ax.annotate("", xy=(bx + dx / length * arrow_len,
                                by + dy / length * arrow_len),
                        xytext=(bx, by),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=0.6, alpha=0.6),
                        zorder=3)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # ── Label top models ──
    top_models = wrsa_means.sort_values(ascending=False).head(top_n).index.tolist()
    texts = []
    for m in top_models:
        if m in models:
            idx = models.index(m)
            # Use clamped position if outlier
            if inlier_mask[idx]:
                px, py = embedding[idx]
            else:
                px = np.clip(embedding[idx, 0], xlim[0] + border_margin_x,
                              xlim[1] - border_margin_x)
                py = np.clip(embedding[idx, 1], ylim[0] + border_margin_y,
                              ylim[1] - border_margin_y)
            short = _clean_model_name(m)
            t = ax.annotate(short, (px, py),
                            fontsize=4, alpha=0.85, color="#333333",
                            fontweight="bold")
            texts.append(t)

    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(
            arrowstyle="-", color="#999999", lw=0.3),
            force_text=(0.3, 0.3), expand=(1.2, 1.4))

    ax.set_xlabel("MDS Dimension 1")
    ax.set_ylabel("MDS Dimension 2")
    ax.set_title(title, fontweight="bold", pad=4)

    n_out = len(outlier_idxs)
    if n_out > 0:
        ax.text(0.98, 0.02,
                f"{n_out} outlier{'s' if n_out != 1 else ''}\nprojected to border",
                transform=ax.transAxes, fontsize=4.5, ha="right", va="bottom",
                color="#888888", style="italic")


def figure3(scores, meta, nc):
    """Model-to-model representational similarity.

    Single-row, three square panels:
      a) KDE of pairwise similarities
      b) MDS of model representations (cRSA)  — zoomed to main cluster
      c) MDS of model representations (veRSA) — zoomed to main cluster
    """
    from sklearn.manifold import MDS
    from scipy.stats import gaussian_kde
    from adjustText import adjust_text

    panel_h = 2.2
    fig, axes = plt.subplots(1, 3, figsize=(10, panel_h),
                              gridspec_kw={"width_ratios": [1, 1, 1],
                                           "wspace": 0.45})

    class_colors_map = {
        "Convolutional": "#e8853a",
        "Transformer": "#3b7dd8",
        "Hybrid": "#4daf4a",
        "MLP-Mixer": "#984ea3",
    }
    class_markers = {
        "Convolutional": "o",
        "Transformer": "s",
        "Hybrid": "D",
        "MLP-Mixer": "^",
    }
    title_map = {"crsa": "cRSA", "wrsa": "veRSA"}

    # Pre-compute correlation + MDS for both eval types
    eval_data = {}
    for eval_type in ["crsa", "wrsa"]:
        pivot = (
            scores[scores["eval_type"] == eval_type]
            .pivot_table(index="model", columns="subject", values="test_score")
            .dropna()
        )
        models = pivot.index.tolist()
        corr_matrix = np.corrcoef(pivot.values)
        n = len(models)
        upper_tri = corr_matrix[np.triu_indices(n, k=1)]

        distance_matrix = 1 - corr_matrix
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.maximum(distance_matrix, 0)

        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, normalized_stress="auto")
        embedding = mds.fit_transform(distance_matrix)
        # Centre the embedding so the main cluster sits at (0, 0)
        embedding -= np.median(embedding, axis=0)

        eval_data[eval_type] = {
            "models": models,
            "upper_tri": upper_tri,
            "embedding": embedding,
        }

    # --- Panel a: Overlaid KDE density plot ---
    ax = axes[0]
    kde_colors = {"crsa": "#888888", "wrsa": "#3b7dd8"}
    kde_fills = {"crsa": "#cccccc", "wrsa": "#a8c8ee"}

    for eval_type in ["crsa", "wrsa"]:
        tri = eval_data[eval_type]["upper_tri"]
        kde = gaussian_kde(tri, bw_method=0.04)
        x_grid = np.linspace(-1, 1, 500)
        density = kde(x_grid)

        ax.fill_between(x_grid, density, alpha=0.25,
                        color=kde_fills[eval_type])
        ax.plot(x_grid, density, color=kde_colors[eval_type],
                linewidth=1.0, label=title_map[eval_type])
        ax.axvline(tri.mean(), color=kde_colors[eval_type],
                   linewidth=0.6, linestyle="--", alpha=0.6)
        peak = density.max()
        ax.text(tri.mean() + 0.02,
                peak * (0.92 if eval_type == "wrsa" else 0.72),
                f"$\\mu$ = {tri.mean():.2f}",
                fontsize=5.5, color=kde_colors[eval_type],
                fontweight="bold")

    ax.set_xlabel("Pairwise similarity ($r$)")
    ax.set_ylabel("Density")
    ax.set_title("Similarity Distribution", fontweight="bold", pad=4)
    ax.set_xlim(-1, 1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=5.5)
    add_panel_label(ax, "a")

    # --- Panels b & c: MDS plots ---
    wrsa_means = (scores[scores["eval_type"] == "wrsa"]
                  .groupby("model")["test_score"].mean())
    top_n = 3

    for panel_idx, eval_type in enumerate(["crsa", "wrsa"]):
        ax = axes[panel_idx + 1]
        data = eval_data[eval_type]
        models = data["models"]
        embedding = data["embedding"]

        # Build class lookup
        model_classes = {}
        for m in models:
            mc = meta[meta["model"] == m]["model_class"].values
            model_classes[m] = mc[0] if len(mc) > 0 else ""

        # Plot each class with distinct marker
        for cls_name, cls_color in class_colors_map.items():
            marker = class_markers[cls_name]
            idxs = [i for i, m in enumerate(models)
                    if model_classes[m] == cls_name]
            if not idxs:
                continue
            ax.scatter(embedding[idxs, 0], embedding[idxs, 1],
                       c=cls_color, marker=marker, s=12, alpha=0.55,
                       edgecolors="white", linewidths=0.15,
                       label=cls_name, zorder=2)

        # Label top models
        top_models = wrsa_means.sort_values(ascending=False).head(top_n).index.tolist()
        texts = []
        for m in top_models:
            if m in models:
                idx = models.index(m)
                short = _clean_model_name(m)
                t = ax.annotate(short,
                                (embedding[idx, 0], embedding[idx, 1]),
                                fontsize=4, alpha=0.85, color="#333333",
                                fontweight="bold")
                texts.append(t)

        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(
                arrowstyle="-", color="#999999", lw=0.3),
                force_text=(0.3, 0.3), expand=(1.2, 1.4))

        ax.set_xlabel("MDS Dimension 1")
        ax.set_ylabel("MDS Dimension 2")
        ax.set_title(f"MDS ({title_map[eval_type]})",
                     fontweight="bold", pad=4)

        # Legend only on right panel
        if panel_idx == 1:
            handles = [plt.Line2D([0], [0], marker=class_markers[c],
                                  color="w", markerfacecolor=col,
                                  markeredgecolor="white",
                                  markeredgewidth=0.2, markersize=4,
                                  label=c)
                       for c, col in class_colors_map.items()]
            ax.legend(handles=handles, loc="best", fontsize=4.5,
                      frameon=True, framealpha=0.95, edgecolor="none",
                      handletextpad=0.3)

        add_panel_label(ax, PANEL_LABELS[panel_idx + 1])

    # Match veRSA axes to cRSA axes
    axes[2].set_xlim(axes[1].get_xlim())
    axes[2].set_ylim(axes[1].get_ylim())

    fig.savefig(RESULTS_DIR / "fig3_model_comparison.pdf")
    fig.savefig(RESULTS_DIR / "fig3_model_comparison.png", dpi=300)
    plt.close(fig)
    print("Figure 3 saved (model comparison).")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    np.random.seed(42)
    scores, meta, nc = load_data()
    print(f"Loaded {len(scores)} score rows, {len(meta)} models, "
          f"{len(nc)} subjects")

    figure1(scores, meta, nc)
    figure2(scores, meta, nc)
    figure3(scores, meta, nc)

    print(f"\nAll figures saved to: {RESULTS_DIR}")


def cli(argv=None):
    """CLI wrapper: sets RESULTS_DIR from --results-dir then calls main()."""
    import argparse
    import sys
    global RESULTS_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Directory with best_layer_scores.csv + "
                         "noise_ceilings.csv. Figures are written here too.")
    args = ap.parse_args(argv)
    RESULTS_DIR = args.results_dir
    main()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(cli())
