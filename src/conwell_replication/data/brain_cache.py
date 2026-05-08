#!/usr/bin/env python3
"""Cache the responsive-cortex beta matrix per subject as a memmap-friendly .npy.

The full per-subject beta matrix is the union of ~30 single-trial GLMsingle
sessions, brain-mask-applied. Loading + concatenating costs ~30 s and
~25 GB of working RAM. Doing that once per eval task is wasteful: the eval
sweep is per (subject × split × model) and runs into the thousands.

This module materializes the per-subject beta matrix once, in C-contiguous
float32, alongside an aligned ``image_ids`` array. Each downstream eval
task can then ``np.load(mmap_mode="r")`` and slice rows by image_id with
zero copy — only the pages touched are paged in.

Output (per subject)::

    {cache_root}/{subject}/betas.npy      (n_trials, n_voxels) float32, C-contig
    {cache_root}/{subject}/image_ids.npy  (n_trials,) <U... aligned with betas[i]

Why we don't use ``laion_fmri.subject.Subject.get_betas``: the upstream
package's ``brain_mask_path`` resolves to a ``meanR2gt15mask_mask.nii.gz``
file that the bucket doesn't actually publish. Only the underlying R² map
(``stat-rsquare_desc-R2mean_statmap.nii.gz``) is on the bucket, so we
threshold that ourselves (R² > ``--r2-threshold``, default 0.15) to build
the brain mask in the same way the field standardly does.

Usage::

    conwell-cache-brain --subjects sub-01 --cache-root /ptmp/.../brain_cache
    conwell-cache-brain --cache-root /ptmp/.../brain_cache   # all 5 subjects
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from conwell_replication.data.roi_masks import (
    build_otc_masks,
    build_roi_metadata,
    load_t1w_roi_masks,
)


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _glm_subject_dir(data_root: Path, subject: str) -> Path:
    return data_root / "derivatives" / "glmsingle-tedana" / subject


def _r2mean_path(data_root: Path, subject: str) -> Path:
    return _glm_subject_dir(data_root, subject) / (
        f"{subject}_task-images_space-T1w_stat-rsquare_desc-R2mean_statmap.nii.gz"
    )


def _session_betas_path(data_root: Path, subject: str, session: str) -> Path:
    return _glm_subject_dir(data_root, subject) / session / "func" / (
        f"{subject}_{session}_task-images_space-T1w_"
        f"stat-effect_desc-SingletrialBetas_statmap.nii.gz"
    )


def _session_trials_path(data_root: Path, subject: str, session: str) -> Path:
    return _glm_subject_dir(data_root, subject) / session / "func" / (
        f"{subject}_{session}_task-images_desc-SingletrialBetas_trials.tsv"
    )


def _list_sessions(data_root: Path, subject: str) -> list:
    sub_dir = _glm_subject_dir(data_root, subject)
    if not sub_dir.is_dir():
        raise FileNotFoundError(
            f"Subject directory not found: {sub_dir}. "
            "Did you set --data-root / $LAION_FMRI_ROOT correctly?"
        )
    sessions = sorted(
        d.name for d in sub_dir.iterdir()
        if d.is_dir() and d.name.startswith("ses-")
    )
    if not sessions:
        raise RuntimeError(f"No ses-* directories under {sub_dir}")
    return sessions


def _build_brain_mask(data_root: Path, subject: str, threshold: float) -> Tuple[np.ndarray, tuple]:
    """Return (flat_mask: bool[ravel order='F'], spatial_shape: (X,Y,Z))."""
    import nibabel as nib

    p = _r2mean_path(data_root, subject)
    if not p.exists():
        raise FileNotFoundError(
            f"R²-mean statmap missing: {p}. "
            "Cannot derive brain mask without it."
        )
    img = nib.load(str(p))
    r2 = np.asarray(img.dataobj, dtype=np.float32)  # (X, Y, Z)
    mask = r2 > threshold
    n_voxels = int(mask.sum())
    _log(
        f"  {subject}: brain mask R²>{threshold} → {n_voxels:,} voxels "
        f"out of {mask.size:,} ({100 * n_voxels / mask.size:.1f}%)"
    )
    return mask, r2.shape


def _load_session_betas_masked(
    betas_path: Path, mask_3d: np.ndarray
) -> np.ndarray:
    """Load a 4-D betas NIfTI and apply the 3-D mask.

    Decompresses the whole .nii.gz once into a single (X,Y,Z,T) array
    (~12 GB for our 143³×1044 sessions), then applies the boolean mask
    along the spatial axes. Returns ``(n_trials, n_mask_voxels)`` float32,
    C-contiguous so it concats cleanly across sessions.

    Slicing per-trial out of the lazy ``dataobj`` is way slower because
    each access has to decompress the whole gz blob from scratch — for
    1044 trials × 30 sessions that's 30k full decompressions. One full
    load per session is the right granularity.
    """
    import nibabel as nib

    img = nib.load(str(betas_path))
    if img.shape[:3] != mask_3d.shape:
        raise RuntimeError(
            f"Spatial-shape mismatch: betas {img.shape[:3]} vs mask {mask_3d.shape}"
        )
    arr = np.asarray(img.dataobj, dtype=np.float32)  # (X, Y, Z, T)
    # Fancy-index with the 3D bool mask along the leading axes:
    # arr[mask_3d] -> (n_mask_voxels, T)
    masked = arr[mask_3d]
    out = np.ascontiguousarray(masked.T)  # (T, n_mask_voxels)
    return out


def _process_session(args):
    """Worker entry point: load + mask one session, return image_ids too.

    Returns (session_name, betas_masked: ndarray, image_ids: list[str]).
    """
    subject, data_root, session, mask_3d = args
    b_path = _session_betas_path(data_root, subject, session)
    t_path = _session_trials_path(data_root, subject, session)
    if not b_path.exists() or not t_path.exists():
        return session, None, None
    betas = _load_session_betas_masked(b_path, mask_3d)
    trials = pd.read_csv(t_path, sep="\t")
    if "label" not in trials.columns:
        raise RuntimeError(
            f"{subject}/{session}: trials.tsv has no 'label' column. "
            f"Cols: {list(trials.columns)}"
        )
    if betas.shape[0] != len(trials):
        raise RuntimeError(
            f"{subject}/{session}: betas have {betas.shape[0]} trials "
            f"but trials.tsv has {len(trials)} rows."
        )
    return session, betas, trials["label"].astype(str).tolist()


def cache_subject(
    subject: str,
    data_root: Path,
    cache_root: Path,
    threshold: float = 0.15,
    n_workers: int = 4,
    force: bool = False,
    voxel_set: str = "visual",
    roi_root: Optional[Path] = None,
):
    out_dir = cache_root / subject
    betas_path = out_dir / "betas.npy"
    ids_path = out_dir / "image_ids.npy"
    metadata_path = out_dir / "voxel_metadata.parquet"
    info_path = out_dir / "cache_info.json"

    if betas_path.exists() and ids_path.exists() and not force:
        ids = np.load(ids_path, allow_pickle=False)
        betas = np.load(betas_path, mmap_mode="r")
        if betas.shape[0] == ids.shape[0]:
            _log(
                f"  {subject}: SKIP (already cached) — "
                f"betas {betas.shape} {betas.dtype} "
                f"({betas.nbytes / 1e9:.2f} GB)"
            )
            return
        _log(f"  {subject}: existing cache shape mismatch, re-caching")

    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    mask_3d, _ = _build_brain_mask(data_root, subject, threshold)
    r2_path = _r2mean_path(data_root, subject)
    import nibabel as nib

    r2_img = nib.load(str(r2_path))
    r2 = np.asarray(r2_img.dataobj, dtype=np.float32)
    roi_masks = {}
    otc_masks = {}
    if voxel_set == "otc":
        if roi_root is None:
            roi_root = data_root / "derivatives" / "rois"
        roi_masks = load_t1w_roi_masks(subject, roi_root, r2.shape)
        otc_masks = build_otc_masks(roi_masks)
        mask_3d = mask_3d & otc_masks["otc"]
        _log(
            f"  {subject}: OTC mask after R²>{threshold} → "
            f"{int(mask_3d.sum()):,} voxels"
        )
    elif voxel_set != "visual":
        raise ValueError(f"unsupported voxel_set {voxel_set!r}")
    voxel_metadata = build_roi_metadata(mask_3d, r2, roi_masks) if roi_masks else None

    sessions = _list_sessions(data_root, subject)
    _log(
        f"  {subject}: loading {len(sessions)} sessions in parallel "
        f"(n_workers={n_workers})..."
    )

    # Parallelize session loading — each worker loads + masks one session.
    # Peak memory ≈ n_workers × 12 GB (one decompressed 4D session per worker)
    # plus the growing accumulator on the main process. Keep n_workers
    # conservative: 4 workers ≈ 48 GB peak.
    from multiprocessing import get_context

    work = [(subject, data_root, ses, mask_3d) for ses in sessions]
    ctx = get_context("spawn")  # avoid forking large parent state
    with ctx.Pool(processes=max(1, n_workers)) as pool:
        results = []
        for i, (ses, betas_sess, ids) in enumerate(
            pool.imap_unordered(_process_session, work), 1
        ):
            if betas_sess is None:
                _log(f"    {ses}: SKIP (missing files)")
                continue
            results.append((ses, betas_sess, ids))
            _log(
                f"    [{i}/{len(sessions)}] {ses}: "
                f"{betas_sess.shape[0]} trials × {betas_sess.shape[1]} voxels"
            )

    # imap_unordered yields out-of-order; sort to preserve session order
    # so trial-row index has a defined per-subject ordering.
    results.sort(key=lambda r: r[0])
    all_betas = [r[1] for r in results]
    all_ids: list = []
    for _, _, ids in results:
        all_ids.extend(ids)

    betas = np.concatenate(all_betas, axis=0)
    del all_betas  # free per-session lists before any further allocation
    if not betas.flags["C_CONTIGUOUS"]:
        betas = np.ascontiguousarray(betas)
    image_ids = np.asarray(all_ids)

    if betas.shape[0] != image_ids.shape[0]:
        raise RuntimeError(
            f"{subject}: row mismatch — betas {betas.shape[0]} vs ids {image_ids.shape[0]}"
        )

    # GLMsingle output sometimes has NaN in marginal voxels (poor model fit
    # or boundary). Drop voxels with NaN in any trial here, once at cache
    # build, so downstream eval never sees NaN (sklearn ridge ≥1.5 strictly
    # rejects them). This costs one extra full-array allocation during
    # filter (numpy fancy-indexing copies); peak memory ≈ 2× betas. Worth
    # it to keep the cache clean and the per-eval load cheap.
    nan_voxels = np.isnan(betas).any(axis=0)
    n_nan = int(nan_voxels.sum())
    if n_nan:
        betas = np.ascontiguousarray(betas[:, ~nan_voxels])
        if voxel_metadata is not None:
            voxel_metadata = voxel_metadata.loc[~nan_voxels].copy()
    del nan_voxels

    t_load = time.time() - t0
    t0 = time.time()
    np.save(betas_path, betas)
    np.save(ids_path, image_ids)
    if voxel_metadata is not None:
        voxel_metadata.to_parquet(metadata_path, index=True)
    info = {
        "subject": subject,
        "voxel_set": voxel_set,
        "r2_threshold": threshold,
        "data_root": str(data_root),
        "r2mean_path": str(r2_path),
        "roi_root": str(roi_root) if roi_root is not None else None,
        "spatial_shape": list(r2.shape),
        "affine": r2_img.affine.tolist(),
        "n_voxels": int(betas.shape[1]),
        "n_nan_voxels_dropped": n_nan,
        "metadata_path": str(metadata_path) if voxel_metadata is not None else None,
    }
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True))
    t_save = time.time() - t0

    _log(
        f"  {subject}: cached — betas {betas.shape} {betas.dtype} "
        f"({betas.nbytes / 1e9:.2f} GB), {len(set(image_ids))} unique image_ids, "
        f"dropped {n_nan} NaN voxels "
        f"[load+mask={t_load:.1f}s, save={t_save:.1f}s] -> {out_dir}"
    )


def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--subjects",
        nargs="+",
        default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
        help="BIDS subject IDs to cache. Default: all 5.",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="LAION-fMRI data dir. Defaults to $LAION_FMRI_ROOT.",
    )
    ap.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Where the per-subject .npy caches go.",
    )
    ap.add_argument(
        "--r2-threshold",
        type=float,
        default=0.15,
        help="Voxels with mean R² above this enter the mask. Default 0.15.",
    )
    ap.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Parallel session loaders. Each worker peaks at ~12 GB; with 4 "
             "workers and a few-GB accumulator, plan for ~60–80 GB --mem.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-cache even if existing files appear consistent.",
    )
    ap.add_argument(
        "--voxel-set",
        choices=("visual", "otc"),
        default="visual",
        help="Voxel set to cache. 'otc' builds the Conwell-style OTC mask. "
             "Default: visual.",
    )
    ap.add_argument(
        "--roi-root",
        type=Path,
        default=None,
        help="ROI derivative root. Defaults to DATA_ROOT/derivatives/rois.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    if args.data_root is None:
        env = os.environ.get("LAION_FMRI_ROOT")
        if not env:
            raise SystemExit(
                "--data-root not given and $LAION_FMRI_ROOT is not set."
            )
        args.data_root = Path(env)

    args.cache_root.mkdir(parents=True, exist_ok=True)
    _log(f"data_root  = {args.data_root}")
    _log(f"cache_root = {args.cache_root}")
    _log(f"subjects   = {args.subjects}")
    _log(f"r2 thresh  = {args.r2_threshold}")

    for subject in args.subjects:
        cache_subject(
            subject,
            args.data_root,
            args.cache_root,
            threshold=args.r2_threshold,
            n_workers=args.n_workers,
            force=args.force,
            voxel_set=args.voxel_set,
            roi_root=args.roi_root,
        )

    _log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
