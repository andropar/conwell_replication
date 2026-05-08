#!/usr/bin/env python3
"""Conwell-style noise ceilings per (subject, region, metric, score_set).

Reads ``brain_cache_otc/<subject>/{betas.npy, image_ids.npy, voxel_metadata.parquet}``
and writes a CSV with the per-ROI noise ceilings used by the split-half plot.

For each subject and each ROI column with >= ``--min-roi-voxels`` voxels:

* **RSA**  — brain-RDM split-half reliability. Within each ROI, split each
  stimulus' repetitions into two halves, average → two response matrices →
  two RDMs over the same stim set; Pearson r over the upper triangle,
  Spearman-Brown corrected to full-N trials. Computed separately for the
  train (even index) and test (odd index) stim subsets that ``rsa_splithalf``
  uses, averaged across ``--n-splits`` random half-pair samples.

* **SRPR** — NSD-style per-voxel signal/noise ceiling on z-scored betas
  (matches the formula provided by the user), averaged across ROI voxels,
  converted to a Pearson-r equivalent via ``sqrt(NC% / 100)``. Computed on
  the test stim subset to match SRPR test scoring.

Stimulus pool matches the splithalf eval: ``laion_fmri.splits`` shared pool,
with ``shared_4rep_OOD_*`` images excluded.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = Path(os.environ.get("CONWELL_BRAIN_CACHE", "brain_cache_otc"))
DEFAULT_OUT = Path(
    os.environ.get(
        "CONWELL_NOISE_CEILING_CSV",
        REPO_ROOT / "results" / "noise_ceiling" / "noise_ceiling.csv",
    )
)


# ---------------------------------------------------------------------------
# Helpers (inlined to keep this script standalone)
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _is_ood_image_id(image_id: str) -> bool:
    return image_id.startswith("shared_4rep_OOD_")


def _shared_non_ood_pool() -> set:
    from laion_fmri.splits import list_splits, get_train_test_ids
    out: set = set()
    for name in list_splits():
        try:
            train, test = get_train_test_ids(name, pool="shared")
        except Exception:
            continue
        out.update(train)
        out.update(test)
    return {iid for iid in out if not _is_ood_image_id(iid)}


def _rdm_from_responses(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float32)
    means = Y.mean(axis=1, dtype=np.float64).astype(np.float32)
    Z = Y - means[:, None]
    norms = np.sqrt(np.einsum("ij,ij->i", Z, Z, dtype=np.float64)).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = Z / norms[:, None]
    return 1.0 - (Z @ Z.T)


def _upper_triangle(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


def _spearman_brown(r: float, k: float = 2.0) -> float:
    """Project split-half r to k-times-as-much-data reliability."""
    if not np.isfinite(r):
        return float("nan")
    return float((k * r) / (1.0 + (k - 1.0) * r))


# ---------------------------------------------------------------------------
# RSA NC: brain-RDM split-half reliability per ROI
# ---------------------------------------------------------------------------
def rsa_split_half_nc(
    betas: np.ndarray,
    labels: np.ndarray,
    voxel_idx: np.ndarray,
    n_splits: int = 50,
    seed: int = 0,
    min_reps: int = 2,
) -> Tuple[float, int, int]:
    """Split repetitions per stim into two halves; correlate the two RDMs.

    Returns (sb_corrected_r, n_stim_used, n_splits_actually_run).
    """
    rng = np.random.default_rng(seed)
    Y_roi = np.ascontiguousarray(betas[:, voxel_idx])
    label_to_rows: dict = {}
    for i, lab in enumerate(labels):
        label_to_rows.setdefault(lab, []).append(i)
    label_to_rows = {
        lab: np.asarray(rows, dtype=np.int64)
        for lab, rows in label_to_rows.items()
        if len(rows) >= min_reps
    }
    if not label_to_rows:
        return float("nan"), 0, 0

    stim_order = sorted(label_to_rows.keys())
    n_stim = len(stim_order)
    if n_stim < 3:
        return float("nan"), n_stim, 0

    rs = []
    for split_i in range(n_splits):
        Y_a = np.empty((n_stim, Y_roi.shape[1]), dtype=np.float32)
        Y_b = np.empty((n_stim, Y_roi.shape[1]), dtype=np.float32)
        for j, lab in enumerate(stim_order):
            rows = label_to_rows[lab]
            n = len(rows)
            half = n // 2
            perm = rng.permutation(n)
            sel_a = rows[perm[:half]]
            sel_b = rows[perm[half : 2 * half]]
            Y_a[j] = Y_roi[sel_a].mean(axis=0)
            Y_b[j] = Y_roi[sel_b].mean(axis=0)
        rdm_a = _rdm_from_responses(Y_a)
        rdm_b = _rdm_from_responses(Y_b)
        a = _upper_triangle(rdm_a)
        b = _upper_triangle(rdm_b)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(r):
            rs.append(r)

    if not rs:
        return float("nan"), n_stim, 0
    r_mean = float(np.mean(rs))
    return _spearman_brown(r_mean, k=2.0), n_stim, len(rs)


# ---------------------------------------------------------------------------
# SRPR NC: user's NSD-style per-voxel ceiling
# ---------------------------------------------------------------------------
def _get_repeated_trial_betas(betas: np.ndarray, labels: np.ndarray):
    unique_labels = np.unique(labels)
    repeated_trials = {label: [] for label in unique_labels}
    for i, label in enumerate(labels):
        repeated_trials[label].append(betas[i])
    n_repetitions = [len(repeated_trials[label]) for label in unique_labels]
    repeated_trials = [
        np.nan_to_num(np.array(repeated_trials[label])) for label in unique_labels
    ]
    return repeated_trials, n_repetitions


def noiseceiling(
    betas: np.ndarray,
    labels: np.ndarray,
    n_repetitions=None,
    only_test_set: bool = False,
) -> np.ndarray:
    """NSD-style per-voxel noise ceiling (NC%) — matches the user's formula."""
    repeated_trials, n_repetitions_from_data = _get_repeated_trial_betas(betas, labels)

    if n_repetitions is None:
        n_repetitions = n_repetitions_from_data

    if only_test_set:
        test_set_repetitions = max(n_repetitions_from_data)
        min_repeated_trials = min([len(trials) for trials in repeated_trials]) + 2
        repeated_trials = [
            trials
            for trials in repeated_trials
            if len(trials) >= (test_set_repetitions - min_repeated_trials)
        ]

    repeated_trials = [trials for trials in repeated_trials if len(trials) >= 2]
    if not repeated_trials:
        return np.full(betas.shape[1], np.nan, dtype=np.float64)

    sigma_noise = np.sqrt(
        np.nanmean(
            np.array(
                [
                    np.nanvar(repeated_trials[i], axis=0, ddof=1)
                    for i in range(len(repeated_trials))
                ]
            ),
            axis=0,
        )
    )
    sigma_signal = np.sqrt(np.clip(1.0 - sigma_noise ** 2, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = sigma_signal / sigma_noise

    if isinstance(n_repetitions, list):
        unique_reps, counts = np.unique(n_repetitions, return_counts=True)
        norm = sum(
            (count / reps) for (count, reps) in zip(counts, unique_reps)
        ) / sum(counts)
    else:
        norm = 1.0 / float(n_repetitions)

    nc = 100.0 * ((snr ** 2) / ((snr ** 2) + norm))
    return nc


def srpr_nc(
    betas_z: np.ndarray,
    labels: np.ndarray,
    voxel_idx: np.ndarray,
) -> Tuple[float, int, int]:
    """Mean per-voxel NC% over ROI → r-equivalent. Returns (r, n_stim, n_voxels)."""
    Y_roi = np.ascontiguousarray(betas_z[:, voxel_idx])
    nc_pct = noiseceiling(Y_roi, labels)
    finite = np.isfinite(nc_pct)
    if not finite.any():
        return float("nan"), 0, int(voxel_idx.size)
    mean_pct = float(np.nanmean(np.clip(nc_pct[finite], 0.0, 100.0)))
    r = float(np.sqrt(mean_pct / 100.0))
    n_stim = int(np.unique(labels).size)
    return r, n_stim, int(voxel_idx.size)


# ---------------------------------------------------------------------------
# Per-subject driver
# ---------------------------------------------------------------------------
def _select_trials(
    image_ids: np.ndarray,
    keep_iids: set,
) -> np.ndarray:
    return np.fromiter(
        (i for i, iid in enumerate(image_ids) if iid in keep_iids),
        dtype=np.int64,
    )


def _zscore_per_voxel(betas: np.ndarray) -> np.ndarray:
    """Per-column (per-voxel) z-score; in-place on a copy. ddof=0 to match NSD norms."""
    out = betas.astype(np.float32, copy=True)
    mean = out.mean(axis=0, dtype=np.float64).astype(np.float32)
    out -= mean
    std = np.sqrt(np.einsum("ij,ij->j", out, out, dtype=np.float64) / out.shape[0]).astype(np.float32)
    std[std == 0] = 1.0
    out /= std
    return out


def _even_odd_split(stim_sorted: List[str]) -> Tuple[set, set]:
    even = set(stim_sorted[0::2])
    odd = set(stim_sorted[1::2])
    return even, odd


def compute_subject(
    subject: str,
    cache_root: Path,
    rois: List[str],
    n_splits: int,
    seed: int,
    ncsnr_threshold: Optional[float] = 0.2,
) -> List[dict]:
    sub_dir = cache_root / subject
    betas_path = sub_dir / "betas.npy"
    ids_path = sub_dir / "image_ids.npy"
    meta_path = sub_dir / "voxel_metadata.parquet"
    if not (betas_path.exists() and ids_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"Missing cache files in {sub_dir}")

    _log(f"== {subject}: loading cache")
    t0 = time.time()
    betas = np.load(betas_path)
    image_ids = np.load(ids_path, allow_pickle=False)
    metadata = pd.read_parquet(meta_path)
    _log(
        f"  betas {betas.shape} {betas.dtype} ({betas.nbytes / 1e9:.2f} GB), "
        f"meta cols={list(metadata.columns)[:5]}..., load={time.time() - t0:.1f}s"
    )

    # Apply NCSNR filter so the ceiling matches the eval's voxel set.
    if ncsnr_threshold is not None and "ncsnr" in metadata.columns:
        ncsnr_mask = metadata["ncsnr"].astype(float).to_numpy() > ncsnr_threshold
        n_before = int(ncsnr_mask.size)
        n_after = int(ncsnr_mask.sum())
        betas = betas[:, ncsnr_mask]
        metadata = metadata.loc[ncsnr_mask].reset_index(drop=True)
        _log(
            f"  ncsnr > {ncsnr_threshold}: kept {n_after} of {n_before} voxels "
            f"({100 * n_after / n_before:.1f}%)"
        )
    elif ncsnr_threshold is not None:
        _log(
            "  WARN: ncsnr_threshold set but voxel_metadata has no 'ncsnr' "
            "column; not filtering. Run scripts/compute_ncsnr.py first."
        )

    keep = _shared_non_ood_pool()
    pool_trial_idx = _select_trials(image_ids, keep)
    if pool_trial_idx.size == 0:
        raise RuntimeError(f"{subject}: no shared-non-OOD trials in cache")
    betas_pool = betas[pool_trial_idx]
    labels_pool = image_ids[pool_trial_idx]
    _log(
        f"  {subject}: shared-non-OOD trials = {labels_pool.size:,} "
        f"({np.unique(labels_pool).size} unique stim)"
    )

    # Eval splits stim by alphabetical-sorted-index even/odd; mirror that.
    stim_sorted = sorted(np.unique(labels_pool).tolist())
    train_set, test_set = _even_odd_split(stim_sorted)
    train_mask = np.fromiter((iid in train_set for iid in labels_pool), dtype=bool)
    test_mask = ~train_mask
    betas_train, labels_train = betas_pool[train_mask], labels_pool[train_mask]
    betas_test, labels_test = betas_pool[~train_mask], labels_pool[~train_mask]
    _log(
        f"  {subject}: train stim={len(train_set)} (trials={betas_train.shape[0]:,}); "
        f"test stim={len(test_set)} (trials={betas_test.shape[0]:,})"
    )

    # SRPR NC needs sigma_total≈1 — z-score per voxel on the same data we score on.
    _log(f"  {subject}: z-scoring test betas per voxel")
    betas_test_z = _zscore_per_voxel(betas_test)

    rows: List[dict] = []
    for roi in rois:
        if roi not in metadata.columns:
            continue
        voxel_idx = np.where(metadata[roi].astype(bool).to_numpy())[0]
        if voxel_idx.size < 10:
            continue

        # RSA NC: train + test
        for ss, b, lab in (("train", betas_train, labels_train), ("test", betas_test, labels_test)):
            t = time.time()
            r_sb, n_stim, n_runs = rsa_split_half_nc(
                b, lab, voxel_idx, n_splits=n_splits, seed=seed,
            )
            rows.append({
                "subject":     subject,
                "region":      roi,
                "metric":      "rsa",
                "score_set":   ss,
                "nc_pearson":  r_sb,
                "n_voxels":    int(voxel_idx.size),
                "n_stimuli":   int(n_stim),
                "n_splits":    int(n_runs),
                "method":      f"split-half RDM, Spearman-Brown k=2 over {n_runs} samples",
            })
            _log(
                f"  {subject} {roi:<14s} rsa  {ss}: r={r_sb:.4f}  "
                f"(n_stim={n_stim}, n_voxels={voxel_idx.size}, {time.time() - t:.1f}s)"
            )

        # SRPR NC: test only (matches eval test scoring)
        t = time.time()
        r, n_stim, nv = srpr_nc(betas_test_z, labels_test, voxel_idx)
        rows.append({
            "subject":     subject,
            "region":      roi,
            "metric":      "srpr",
            "score_set":   "test",
            "nc_pearson":  r,
            "n_voxels":    int(nv),
            "n_stimuli":   int(n_stim),
            "n_splits":    0,
            "method":      "NSD per-voxel NC% on z-scored betas, mean over ROI, sqrt(NC%/100)",
        })
        _log(
            f"  {subject} {roi:<14s} srpr test: r={r:.4f}  "
            f"(n_stim={n_stim}, n_voxels={nv}, {time.time() - t:.1f}s)"
        )

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )
    ap.add_argument(
        "--subjects",
        nargs="+",
        default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
    )
    ap.add_argument(
        "--rois",
        nargs="+",
        default=[
            "otc",
            "otc-streams",
            "otc-category",
            "laion-ventral",
            "laion-lateral",
            "v-objects",
            "l-objects",
            "OFA",
            "FFA-1",
            "FFA-2",
            "EBA",
            "FBA",
            "VWFA-1",
            "VWFA-2",
            "mfs-words",
            "pSTS-words",
            "PPA",
            "OPA",
        ],
    )
    ap.add_argument("--n-splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ncsnr-threshold",
        type=float,
        default=0.2,
        help="Drop voxels with ncsnr <= this from each ROI before computing "
             "ceilings. Set negative to disable. Default: 0.2.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    for subject in args.subjects:
        try:
            rows = compute_subject(
                subject=subject,
                cache_root=args.cache_root,
                rois=args.rois,
                n_splits=args.n_splits,
                seed=args.seed,
                ncsnr_threshold=(args.ncsnr_threshold if args.ncsnr_threshold >= 0 else None),
            )
        except Exception as e:
            _log(f"  ERROR on {subject}: {e}")
            raise
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    _log(f"wrote {len(df)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
