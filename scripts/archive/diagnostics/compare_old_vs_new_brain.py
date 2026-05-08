#!/usr/bin/env python3
"""Compare OLD (deepvision_fmri) vs NEW (laion_fmri) preprocessing.

For each of 5 subjects, computes:

  OLD setup    : per-image betas restricted to the OLD hlvis voxel set
                 (per the cstims voxel_metadata 'hlvis' flag, matching the
                 mask actually used in run_rsa_eval.py).
  NEW setup    : per-image betas (mean over trial repetitions) restricted to
                 the NEW OTC mask intersected with NCSNR > 0.2.

Both RDMs are computed over the 1,492 shared images present in both pools,
then we correlate the two upper triangles. Reports per-subject voxel counts
and RDM correlations as a CSV.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _rdm_from_responses(Y: np.ndarray) -> np.ndarray:
    """1 - Pearson r across rows (n_stim x n_voxels), float32."""
    Y = np.asarray(Y, dtype=np.float32)
    means = Y.mean(axis=1, dtype=np.float64).astype(np.float32)
    Z = Y - means[:, None]
    norms = np.sqrt(np.einsum("ij,ij->i", Z, Z, dtype=np.float64)).astype(np.float32)
    norms[norms == 0] = 1.0
    Z = Z / norms[:, None]
    return 1.0 - (Z @ Z.T)


def _upper_tri(rdm: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(rdm.shape[0], k=1)
    return rdm[iu]


def load_old_brain(old_root: Path, subject: str, common_ids: list) -> tuple:
    """Return (Y_shared, n_total_voxels, n_hlvis_voxels) for OLD subject.

    Y_shared is shape (n_common_images, n_hlvis_voxels), float32.
    """
    cache_dir = old_root / "cache" / "voxel_sets" / "deepvision_shared_hlvis" / "finalinterp" / subject
    betas = np.load(cache_dir / "voxel_betas.npy", mmap_mode="r")
    cols = np.load(cache_dir / "voxel_betas_cols.npy", allow_pickle=True)
    metadata = pd.read_csv(cache_dir / "voxel_metadata.csv")

    n_voxels_total = betas.shape[0]
    hlvis_idx = np.where(metadata["hlvis"].astype(bool).to_numpy())[0]
    n_hlvis = int(hlvis_idx.size)

    cols_to_idx = {iid: i for i, iid in enumerate(cols.tolist())}
    img_idx = np.array([cols_to_idx[iid] for iid in common_ids], dtype=np.int64)

    # Memory plan: load only the (hlvis_voxels x common_images) submatrix.
    # numpy fancy-indexing into mmap copies: shape (n_hlvis, n_common).
    Y = np.asarray(betas[hlvis_idx][:, img_idx], dtype=np.float32).T  # (n_common, n_hlvis)
    return Y, n_voxels_total, n_hlvis


def load_new_brain(new_cache_root: Path, subject: str, common_ids: list) -> tuple:
    """Return (Y_shared, n_total_voxels, n_otc_voxels, n_otc_ncsnr) for NEW subject."""
    sub_dir = new_cache_root / subject
    betas = np.load(sub_dir / "betas.npy")  # (n_trials, n_voxels)
    image_ids = np.load(sub_dir / "image_ids.npy", allow_pickle=False)
    metadata = pd.read_parquet(sub_dir / "voxel_metadata.parquet")
    n_voxels_total = betas.shape[1]

    otc_mask = metadata["otc"].astype(bool).to_numpy()
    n_otc = int(otc_mask.sum())
    if "ncsnr" in metadata.columns:
        ncsnr_mask = metadata["ncsnr"].astype(float).to_numpy() > 0.2
    else:
        ncsnr_mask = np.ones_like(otc_mask)
    final_mask = otc_mask & ncsnr_mask
    n_otc_ncsnr = int(final_mask.sum())

    voxel_idx = np.where(final_mask)[0]
    betas_roi = betas[:, voxel_idx]
    del betas

    common_set = set(common_ids)
    rows_by_image = {iid: [] for iid in common_ids}
    for i, iid in enumerate(image_ids):
        if iid in common_set:
            rows_by_image[iid].append(i)
    Y = np.empty((len(common_ids), voxel_idx.size), dtype=np.float32)
    for k, iid in enumerate(common_ids):
        Y[k] = betas_roi[rows_by_image[iid]].mean(axis=0)
    return Y, n_voxels_total, n_otc, n_otc_ncsnr


def _is_ood(iid: str) -> bool:
    return iid.startswith("shared_4rep_OOD_")


def compute_subject(
    subject: str,
    old_root: Path,
    new_cache_root: Path,
    common_ids: list,
) -> list:
    _log(f"== {subject}")
    t0 = time.time()
    Y_old, n_old_total, n_hlvis = load_old_brain(old_root, subject, common_ids)
    _log(f"  old: total={n_old_total:,} voxels, hlvis={n_hlvis:,} (Y={Y_old.shape})")

    Y_new, n_new_total, n_otc, n_otc_ncsnr = load_new_brain(new_cache_root, subject, common_ids)
    _log(f"  new: total={n_new_total:,}, otc={n_otc:,}, otc∩ncsnr>0.2={n_otc_ncsnr:,} (Y={Y_new.shape})")

    # Image subsets to evaluate.
    is_ood = np.array([_is_ood(iid) for iid in common_ids])
    subsets = {
        "all":      np.arange(len(common_ids)),
        "non_ood":  np.where(~is_ood)[0],
        "ood_only": np.where(is_ood)[0],
    }

    rows = []
    for subset_name, subset_idx in subsets.items():
        if subset_idx.size < 4:
            continue
        rdm_old = _rdm_from_responses(Y_old[subset_idx])
        rdm_new = _rdm_from_responses(Y_new[subset_idx])
        a = _upper_tri(rdm_old)
        b = _upper_tri(rdm_new)
        finite = np.isfinite(a) & np.isfinite(b)
        rdm_corr = float(np.corrcoef(a[finite], b[finite])[0, 1])
        rows.append({
            "subject": subject,
            "image_subset": subset_name,
            "n_images": int(subset_idx.size),
            "n_voxels_old_total": n_old_total,
            "n_voxels_old_hlvis": n_hlvis,
            "n_voxels_new_total": n_new_total,
            "n_voxels_new_otc": n_otc,
            "n_voxels_new_otc_ncsnr": n_otc_ncsnr,
            "rdm_correlation": rdm_corr,
        })
        _log(
            f"  RDM corr ({subset_name}, n={subset_idx.size}): r={rdm_corr:.4f}"
        )
    _log(f"  {subject} done in {time.time() - t0:.1f}s")
    return rows


def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-root", type=Path, default=Path("/ptmp/rothj/deepvision_fmri_old"))
    ap.add_argument("--new-cache-root", type=Path, default=Path("/ptmp/rothj/conwell_replication/brain_cache_otc"))
    ap.add_argument("--subjects", nargs="+", default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"])
    ap.add_argument("--out", type=Path, default=Path("/u/rothj/conwell_replication/figures/old_vs_new_brain_comparison.csv"))
    return ap.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Skip subjects that aren't in the OLD cache.
    available = []
    for s in args.subjects:
        sub_dir = (
            args.old_root
            / "cache" / "voxel_sets" / "deepvision_shared_hlvis" / "finalinterp"
            / s
        )
        if (sub_dir / "voxel_betas.npy").exists():
            available.append(s)
        else:
            _log(f"  {s}: not in OLD cache; skipping")
    if not available:
        raise SystemExit("no subjects available in OLD cache")

    sub_dir = (
        args.old_root
        / "cache" / "voxel_sets" / "deepvision_shared_hlvis" / "finalinterp"
        / available[0]
    )
    cols = np.load(sub_dir / "voxel_betas_cols.npy", allow_pickle=True)
    new_ids = np.load(args.new_cache_root / available[0] / "image_ids.npy", allow_pickle=False)
    common = sorted(set(cols.tolist()) & set(new_ids.tolist()))
    _log(f"common image_ids across pools: {len(common)}")

    rows = []
    for s in available:
        rows.extend(compute_subject(s, args.old_root, args.new_cache_root, common))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    _log(f"wrote {args.out}")
    print()
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
