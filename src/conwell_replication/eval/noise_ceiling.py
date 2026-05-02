#!/usr/bin/env python3
"""Compute RDM noise ceilings for the LAION-fMRI data.

Mirrors the ``compute_rdm_noise_ceilings.py`` script from the existing
rsa_large_scale_benchmark, with two modes:

  * ``--mode shared``    Whole-pool noise ceiling on shared images (matches
                          the original script; one row per subject).
  * ``--mode min_nn``    Per-(subject, split, variant, score_set) noise
                          ceilings on the test images of each min_nn split.
                          Use these to compute explainable variance for
                          min_nn evaluations.

Method: split-half across trials within each image_id (odd/even with random
permutation per image), then Spearman-Brown corrected correlation between
the two RDMs (Pearson and Spearman both reported).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from laion_fmri.splits import list_splits, load_split

from conwell_replication.data import LAIONBenchmark

log = logging.getLogger(__name__)


def _upper_tri(rdm: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(rdm.shape[0], k=1)
    return rdm[iu]


def _spearman_brown(r: float) -> float:
    return 2.0 * r / (1.0 + r) if (1.0 + r) != 0 else float("nan")


def _nc_row(
    subject: str,
    rdm_odd: np.ndarray,
    rdm_even: np.ndarray,
    n_stimuli: int,
    extra: Optional[dict] = None,
) -> dict:
    v_odd = _upper_tri(rdm_odd)
    v_even = _upper_tri(rdm_even)
    r_sp_raw = float(spearmanr(v_odd, v_even).correlation)
    r_pe_raw = float(pearsonr(v_odd, v_even).statistic)
    out = {
        "subject":        subject,
        "n_stimuli":      int(n_stimuli),
        "n_pairs":        int(len(v_odd)),
        "r_spearman_raw": r_sp_raw,
        "nc_spearman":    _spearman_brown(r_sp_raw),
        "r_pearson_raw":  r_pe_raw,
        "nc_pearson":     _spearman_brown(r_pe_raw),
    }
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def compute_shared_pool_nc(subjects: Iterable[str], voxel_set: str, seed: int) -> pd.DataFrame:
    rows = []
    for sub in subjects:
        log.info(f"[shared] {sub}")
        bench = LAIONBenchmark(subject=sub, voxel_set=voxel_set, pool="shared")
        rdm_odd, rdm_even, kept = bench.trial_splithalf_rdms(seed=seed)
        rows.append(_nc_row(sub, rdm_odd, rdm_even, len(kept)))
    return pd.DataFrame(rows)


def compute_min_nn_nc(
    subjects: Iterable[str],
    voxel_set: str,
    splits: Optional[Iterable[str]],
    seed: int,
    pool: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for sub in subjects:
        eval_pool = pool or sub
        log.info(f"[min_nn] subject={sub}, pool={eval_pool}")
        bench = LAIONBenchmark(subject=sub, voxel_set=voxel_set, pool=eval_pool)
        sp_names = list(splits) if splits else list_splits()
        for sp_name in sp_names:
            sp = load_split(sp_name, pool=eval_pool)
            for variant in sp.variants:
                # Test set noise ceiling (eve uses this for min_nn)
                rdm_odd, rdm_even, kept = bench.trial_splithalf_rdms(
                    image_ids=variant.test_ids, seed=seed,
                )
                rows.append(_nc_row(
                    sub, rdm_odd, rdm_even, len(kept),
                    extra={"split": sp_name, "variant": variant.variant_id,
                           "score_set": "test", "pool": eval_pool},
                ))
                # Train set NC for completeness (when reporting eve on train)
                rdm_odd_tr, rdm_even_tr, kept_tr = bench.trial_splithalf_rdms(
                    image_ids=variant.train_ids, seed=seed,
                )
                rows.append(_nc_row(
                    sub, rdm_odd_tr, rdm_even_tr, len(kept_tr),
                    extra={"split": sp_name, "variant": variant.variant_id,
                           "score_set": "train", "pool": eval_pool},
                ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("shared", "min_nn"), default="shared")
    ap.add_argument("--subjects", nargs="+",
                    default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"])
    ap.add_argument("--voxel-set", default="hlvis", choices=("hlvis", "visual"))
    ap.add_argument("--splits", nargs="+", default=None,
                    help="Subset of split names to run (default: all 11).")
    ap.add_argument("--pool", default=None,
                    help="Pool to evaluate over (min_nn mode only): "
                         "'shared' or a subject id like 'sub-01'. If "
                         "omitted, uses each --subject's own pool.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Permutation seed for odd/even trial splitting.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output CSV (one row per (subject[, split, variant, score_set])).")
    return ap.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.mode == "shared":
        df = compute_shared_pool_nc(args.subjects, args.voxel_set, args.seed)
    else:
        df = compute_min_nn_nc(
            args.subjects, args.voxel_set, args.splits, args.seed, pool=args.pool,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows → {args.out}")
    print(df.to_string(index=False, max_rows=20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
