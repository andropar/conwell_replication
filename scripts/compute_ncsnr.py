#!/usr/bin/env python3
"""Compute per-voxel NCSNR and merge into the OTC brain cache metadata.

For each subject in ``--cache-root``:

1. Load ``betas.npy`` and ``image_ids.npy`` from ``brain_cache_otc/<subject>/``.
2. Z-score betas per voxel (the user-supplied formula assumes total variance =
   1 — i.e. the betas are normalized).
3. Compute the per-voxel signal-to-noise ratio used in the NSD noise ceiling::

       sigma_noise  = sqrt( mean over stim of var(reps, ddof=1) )
       sigma_signal = sqrt( max(0, 1 - sigma_noise**2) )
       NCSNR        = sigma_signal / sigma_noise

   Only stimuli with >= 2 repetitions are used.

4. Write ``ncsnr.npy`` next to ``betas.npy`` and merge an ``ncsnr`` column
   (plus ``ncsnr_pass`` = ncsnr > threshold) into ``voxel_metadata.parquet``.

Used downstream by:

  * ``LAIONBenchmark`` — drops voxels with ``ncsnr <= --ncsnr-threshold``
    when constructing the per-subject response matrix.
  * ``compute_noise_ceiling.py`` — same filter so noise ceilings and scores
    are computed on identical voxel sets.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_CACHE_ROOT = Path(os.environ.get("CONWELL_BRAIN_CACHE", "brain_cache_otc"))


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _zscore_per_voxel(betas: np.ndarray) -> np.ndarray:
    out = betas.astype(np.float32, copy=True)
    mean = out.mean(axis=0, dtype=np.float64).astype(np.float32)
    out -= mean
    std = np.sqrt(np.einsum("ij,ij->j", out, out, dtype=np.float64) / out.shape[0]).astype(np.float32)
    std[std == 0] = 1.0
    out /= std
    return out


def compute_ncsnr(
    betas_z: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Per-voxel NCSNR from z-scored betas + per-trial labels."""
    label_to_rows: dict = {}
    for i, lab in enumerate(labels):
        label_to_rows.setdefault(lab, []).append(i)
    label_to_rows = {k: np.asarray(v, dtype=np.int64) for k, v in label_to_rows.items() if len(v) >= 2}
    if not label_to_rows:
        return np.full(betas_z.shape[1], np.nan, dtype=np.float64)

    n_voxels = betas_z.shape[1]
    var_per_stim = np.empty((len(label_to_rows), n_voxels), dtype=np.float64)
    for j, rows in enumerate(label_to_rows.values()):
        sub = betas_z[rows]
        var_per_stim[j] = np.nanvar(sub, axis=0, ddof=1)

    sigma_noise = np.sqrt(np.nanmean(var_per_stim, axis=0))
    sigma_signal = np.sqrt(np.clip(1.0 - sigma_noise ** 2, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        ncsnr = sigma_signal / sigma_noise
    return ncsnr


def process_subject(
    subject: str,
    cache_root: Path,
    threshold: float,
    save_npy: bool,
) -> dict:
    sub_dir = cache_root / subject
    betas_path = sub_dir / "betas.npy"
    ids_path = sub_dir / "image_ids.npy"
    meta_path = sub_dir / "voxel_metadata.parquet"
    if not (betas_path.exists() and ids_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"Missing cache files in {sub_dir}")

    _log(f"== {subject}: loading cache")
    betas = np.load(betas_path)
    image_ids = np.load(ids_path, allow_pickle=False)
    metadata = pd.read_parquet(meta_path)
    _log(f"  betas {betas.shape} {betas.dtype}, meta rows={len(metadata)}")
    if betas.shape[1] != len(metadata):
        raise RuntimeError(
            f"{subject}: betas has {betas.shape[1]} voxels but metadata has "
            f"{len(metadata)}"
        )

    _log(f"  z-scoring per voxel")
    betas_z = _zscore_per_voxel(betas)
    del betas

    _log(f"  computing NCSNR over {len(np.unique(image_ids))} unique stimuli")
    ncsnr = compute_ncsnr(betas_z, image_ids).astype(np.float32)
    del betas_z

    n = ncsnr.size
    n_pass = int((ncsnr > threshold).sum())
    summary = {
        "subject": subject,
        "n_voxels": n,
        "ncsnr_mean": float(np.nanmean(ncsnr)),
        "ncsnr_median": float(np.nanmedian(ncsnr)),
        "ncsnr_p90": float(np.nanpercentile(ncsnr, 90)),
        "n_pass": n_pass,
        "frac_pass": n_pass / n if n else float("nan"),
        "threshold": threshold,
    }
    _log(
        f"  NCSNR median={summary['ncsnr_median']:.3f}, mean={summary['ncsnr_mean']:.3f}, "
        f"pass(> {threshold})={n_pass}/{n} ({100 * summary['frac_pass']:.1f}%)"
    )

    if save_npy:
        np.save(sub_dir / "ncsnr.npy", ncsnr)
        _log(f"  wrote {sub_dir / 'ncsnr.npy'}")

    metadata = metadata.copy()
    metadata["ncsnr"] = ncsnr.astype(np.float32)
    metadata["ncsnr_pass"] = (ncsnr > threshold).astype(np.int8)
    metadata.to_parquet(meta_path, index=True)
    _log(f"  merged ncsnr+ncsnr_pass into {meta_path}")

    return summary


def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )
    ap.add_argument(
        "--subjects",
        nargs="+",
        default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
    )
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument(
        "--no-save-npy",
        action="store_true",
        help="Skip writing the ncsnr.npy sidecar (column still merged into parquet).",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    rows = []
    for subject in args.subjects:
        rows.append(
            process_subject(
                subject=subject,
                cache_root=args.cache_root,
                threshold=args.threshold,
                save_npy=not args.no_save_npy,
            )
        )
    summary_df = pd.DataFrame(rows)
    summary_path = args.cache_root / "ncsnr_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    _log(f"wrote summary -> {summary_path}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
