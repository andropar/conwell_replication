#!/usr/bin/env python3
"""Sanity-check figure: OTC mask + NCSNR-pass overlay per subject.

Uses the R^2-mean statmap as a brain-shaped grayscale background. Overlays:

  * pale yellow  — OTC voxels (streams ∪ category)
  * dark orange  — OTC voxels also passing NCSNR > 0.2 (the eval set)
  * magenta      — category-selective ROIs (faces/bodies/words/places)

Per subject we render three rows: axial (Z), coronal (Y), sagittal (X), each
column a slice index that spans the bulk of the OTC volume on that axis.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("LAION_FMRI_ROOT", "laion_fmri_data"))
DEFAULT_CACHE_ROOT = Path(
    os.environ.get("CONWELL_BRAIN_CACHE", "brain_cache_otc")
)
DEFAULT_OUT = Path(os.environ.get("CONWELL_OTC_ROI_FIGURE", REPO_ROOT / "figures" / "otc_roi_sanity.pdf"))


def load_r2(data_root: Path, subject: str) -> np.ndarray:
    import nibabel as nib

    p = (
        data_root
        / "derivatives"
        / "glmsingle-tedana"
        / subject
        / f"{subject}_task-images_space-T1w_stat-rsquare_desc-R2mean_statmap.nii.gz"
    )
    return np.asarray(nib.load(str(p)).dataobj, dtype=np.float32)


def load_subject_masks(
    data_root: Path,
    cache_root: Path,
    subject: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (r2, otc_mask, ncsnr_pass_mask_in_otc, category_mask) for subject."""
    from conwell_replication.data.roi_masks import load_t1w_roi_masks, build_otc_masks

    r2 = load_r2(data_root, subject)
    roi_masks = load_t1w_roi_masks(subject, data_root / "derivatives" / "rois", r2.shape)
    otc_masks = build_otc_masks(roi_masks)
    otc = otc_masks["otc"].astype(bool)

    # Category-selective union (the 11 ROIs that contribute to OTC-category)
    cat_labels = (
        "OFA", "FFA-1", "FFA-2", "EBA", "FBA",
        "VWFA-1", "VWFA-2", "mfs-words", "pSTS-words",
        "PPA", "OPA",
    )
    category = np.zeros_like(otc, dtype=bool)
    for label in cat_labels:
        if label in roi_masks:
            category |= roi_masks[label].astype(bool)

    # NCSNR-pass overlay: needs the brain_cache flat_index → 3D mapping.
    ncsnr_pass_3d = np.zeros_like(otc, dtype=bool)
    meta_path = cache_root / subject / "voxel_metadata.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        if "ncsnr_pass" in meta.columns:
            keep = meta["ncsnr_pass"].astype(bool).to_numpy()
            ii = meta["i"].to_numpy().astype(int)
            jj = meta["j"].to_numpy().astype(int)
            kk = meta["k"].to_numpy().astype(int)
            ncsnr_pass_3d[ii[keep], jj[keep], kk[keep]] = True
    ncsnr_pass_in_otc = ncsnr_pass_3d & otc
    return r2, otc, ncsnr_pass_in_otc, category


def pick_slices_for_axis(otc: np.ndarray, axis: int, n: int = 6) -> List[int]:
    """Slices on `axis` spanning the [5th, 95th] percentile of OTC mass."""
    other = tuple(a for a in range(3) if a != axis)
    counts = otc.sum(axis=other)
    if counts.sum() == 0:
        return []
    cum = np.cumsum(counts) / counts.sum()
    lo = int(np.searchsorted(cum, 0.05))
    hi = int(np.searchsorted(cum, 0.95))
    if hi <= lo:
        hi = lo + 1
    if n == 1:
        return [int((lo + hi) // 2)]
    return list(np.linspace(lo, hi, n).round().astype(int))


def _axis_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Return a 2D slice through `vol` at `idx` along `axis`, oriented so the
    plot reads as a standard radiological view (superior up, anterior up
    for axial; superior up for coronal/sagittal)."""
    if axis == 2:        # axial: drop Z, show (Y, X) with Y flipped (anterior up)
        sl = vol[:, :, idx].T
        return np.flipud(sl)
    if axis == 1:        # coronal: drop Y, show (Z, X) with Z flipped (superior up)
        sl = vol[:, idx, :].T
        return np.flipud(sl)
    # axis == 0, sagittal: drop X, show (Z, Y) with Z flipped (superior up)
    sl = vol[idx, :, :].T
    return np.flipud(sl)


AXIS_LABELS = {0: "x", 1: "y", 2: "z"}


def render_panel(
    ax: matplotlib.axes.Axes,
    r2: np.ndarray,
    otc: np.ndarray,
    ncsnr_pass: np.ndarray,
    category: np.ndarray,
    axis: int,
    idx: int,
    bg_vmax: float = 15.0,
) -> None:
    bg = _axis_slice(r2, axis, idx)
    ax.imshow(bg, cmap="gray", vmin=0.0, vmax=bg_vmax, origin="upper")

    otc_slice = _axis_slice(otc, axis, idx)
    ncsnr_slice = _axis_slice(ncsnr_pass, axis, idx)
    cat_slice = _axis_slice(category, axis, idx)

    pale = otc_slice & ~ncsnr_slice
    if pale.any():
        ax.imshow(
            np.where(pale, 1, np.nan),
            cmap=ListedColormap(["#f7e08e"]),
            vmin=0,
            vmax=1,
            alpha=0.85,
            origin="upper",
        )
    if ncsnr_slice.any():
        ax.imshow(
            np.where(ncsnr_slice, 1, np.nan),
            cmap=ListedColormap(["#d55e00"]),
            vmin=0,
            vmax=1,
            alpha=0.95,
            origin="upper",
        )
    if cat_slice.any():
        ax.contour(cat_slice.astype(int), levels=[0.5], colors="#cc79a7", linewidths=0.6)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{AXIS_LABELS[axis]}={idx}", fontsize=6, pad=1.5)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--subjects", nargs="+", default=SUBJECTS)
    p.add_argument("--n-slices", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    subj_data = {}
    for subject in args.subjects:
        print(f"loading {subject}", flush=True)
        try:
            subj_data[subject] = load_subject_masks(args.data_root, args.cache_root, subject)
        except Exception as exc:
            print(f"  ERROR for {subject}: {exc}", flush=True)
            continue

    if not subj_data:
        sys.exit("no subjects rendered")

    axis_order = [(2, "axial"), (1, "coronal"), (0, "sagittal")]
    n_axes = len(axis_order)
    n_rows = len(subj_data) * n_axes
    n_cols = args.n_slices
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.55 * n_cols, 1.4 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    for s_idx, (subject, (r2, otc, ncsnr_pass, category)) in enumerate(subj_data.items()):
        n_otc = int(otc.sum())
        n_pass = int(ncsnr_pass.sum())
        for a_idx, (axis, axis_name) in enumerate(axis_order):
            row = s_idx * n_axes + a_idx
            slices = pick_slices_for_axis(otc, axis, args.n_slices)
            for c, ax in enumerate(axes[row]):
                if c < len(slices):
                    render_panel(ax, r2, otc, ncsnr_pass, category, axis, slices[c])
                else:
                    ax.axis("off")
            label = axis_name
            if a_idx == 0:
                label = f"{subject}\nOTC={n_otc:,}\nNCSNR pass={n_pass:,}\n\n{axis_name}"
            axes[row, 0].set_ylabel(
                label,
                fontsize=6,
                rotation=0,
                ha="right",
                va="center",
                labelpad=22,
            )

    fig.suptitle(
        "OTC mask sanity check — axial / coronal / sagittal slices over T1w R² background\n"
        "yellow = OTC, orange = OTC ∩ NCSNR > 0.2 (eval voxels), magenta outline = category-selective ROIs",
        fontsize=7,
    )
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print(f"wrote {args.out} and {args.out.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
