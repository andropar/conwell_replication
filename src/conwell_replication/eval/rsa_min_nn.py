#!/usr/bin/env python3
"""min_nn generalization-split RSA evaluation.

For each subject × split × variant × model × layer:

  * Subset the (image_id-indexed) feature matrix to the split's train_ids
    and test_ids.
  * Subset the brain response matrix to the same image_ids.
  * cRSA  — corr(model_RDM[set], brain_RDM[set]) for set ∈ {train, test}.
            Uses raw model features (no ridge).
  * WRSA  — fit ridge on train → predict test → corr(predicted_RDM, brain_RDM).
            Also computes the analogous "train" score from CV predictions.
  * SRPR  — single-voxel Pearson r between predicted and observed responses,
            averaged across voxels.

Best-layer selection (Conwell convention): pick the layer with highest TRAIN
score, report the TEST score for that same layer. This is done downstream by
``conwell_replication.analysis.prepare_scores``; the evaluator simply emits
all layer × score_set rows.

Each split is treated as one evaluation. ``random_*`` and ``cluster_k5_*``
splits each have a single variant; ``tau_*`` splits also have a single
variant_id=0. So per subject we run 13 evaluations.

Outputs (under ``--out``):

  * ``results_<subject>_<split>.parquet`` — per (subject, split) parquet
  * ``results_all.parquet``               — all rows concatenated
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from laion_fmri.splits import list_splits, load_split

from conwell_replication.data import LAIONBenchmark
from conwell_replication.eval._common import (
    align_features,
    compare_rdms,
    fit_ridge,
    load_features,
    rdm_from_responses,
    score_func,
)


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Per (subject, split, variant, model) evaluation
# ---------------------------------------------------------------------------
def evaluate_model_on_split(
    h5_path: Path,
    benchmark: LAIONBenchmark,
    train_ids: List[str],
    test_ids: List[str],
    alphas: List[float],
    do_crsa: bool,
    do_wrsa: bool,
    do_srpr: bool,
    split_name: str,
    variant_id: int,
) -> List[dict]:
    """Run cRSA/WRSA/SRPR for one model on one (subject, split, variant)."""
    model_name = h5_path.stem.replace("-", "/")
    t0 = time.time()

    features, h5_ids = load_features(h5_path)

    # Slice features by the split's image_ids.
    feats_train_full = {ln: align_features(f, h5_ids, train_ids) for ln, f in features.items()}
    feats_test_full  = {ln: align_features(f, h5_ids, test_ids)  for ln, f in features.items()}

    # Brain responses on the same image_ids.
    Y_train = benchmark.betas(train_ids, as_array=True).T   # (n_train, n_voxels)
    Y_test  = benchmark.betas(test_ids,  as_array=True).T

    rdm_train_brain = rdm_from_responses(Y_train)
    rdm_test_brain  = rdm_from_responses(Y_test)

    region = benchmark.voxel_set
    subject = benchmark.subject
    rows: List[dict] = []

    for layer_idx, layer_name in enumerate(features.keys()):
        ftrain = feats_train_full[layer_name]
        ftest  = feats_test_full[layer_name]

        # cRSA on raw features
        if do_crsa:
            for ss, mfeats, target in (
                ("train", ftrain, rdm_train_brain),
                ("test",  ftest,  rdm_test_brain),
            ):
                m_rdm = rdm_from_responses(mfeats)
                rows.append({
                    "score":             float(compare_rdms(m_rdm, target)),
                    "score_set":         ss,
                    "eval_type":         "crsa",
                    "region":            region,
                    "model":             model_name,
                    "model_layer":       layer_name,
                    "model_layer_index": layer_idx,
                    "subject":           subject,
                    "split":             split_name,
                    "variant":           variant_id,
                })

        # Ridge once per layer; reused for WRSA + SRPR
        if do_wrsa or do_srpr:
            try:
                pred_train, pred_test, _ridge = fit_ridge(ftrain, ftest, Y_train, alphas)
            except Exception as e:
                _log(f"    Ridge failed for {layer_name}: {e}")
                continue

            if do_wrsa:
                for ss, pred, target in (
                    ("train", pred_train, rdm_train_brain),
                    ("test",  pred_test,  rdm_test_brain),
                ):
                    m_rdm = rdm_from_responses(pred)
                    rows.append({
                        "score":             float(compare_rdms(m_rdm, target)),
                        "score_set":         ss,
                        "eval_type":         "wrsa",
                        "region":            region,
                        "model":             model_name,
                        "model_layer":       layer_name,
                        "model_layer_index": layer_idx,
                        "subject":           subject,
                        "split":             split_name,
                        "variant":           variant_id,
                    })

            if do_srpr:
                for ss, pred, ytrue in (
                    ("train", pred_train, Y_train),
                    ("test",  pred_test,  Y_test),
                ):
                    per_voxel = score_func(ytrue, pred, score_type="pearson_r")
                    rows.append({
                        "score":             float(np.mean(per_voxel)),
                        "score_set":         ss,
                        "eval_type":         "srpr",
                        "region":            region,
                        "model":             model_name,
                        "model_layer":       layer_name,
                        "model_layer_index": layer_idx,
                        "subject":           subject,
                        "split":             split_name,
                        "variant":           variant_id,
                    })

    _log(f"  {model_name} [{split_name} v{variant_id}]: "
         f"{len(rows)} rows in {time.time() - t0:.1f}s")
    return rows


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True,
                    help="Directory of per-model SRP feature .h5 files.")
    ap.add_argument("--pool", default=None,
                    help="Pool to evaluate over: 'shared' or a subject id "
                         "like 'sub-01'. If omitted, uses each --subject's "
                         "own pool (e.g. sub-01 evaluates on the sub-01 pool).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for results parquets.")
    ap.add_argument("--subjects", nargs="+",
                    default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"])
    ap.add_argument("--splits", nargs="+", default=None,
                    help="Subset of split names to run (default: all 13).")
    ap.add_argument("--voxel-set", default="hlvis", choices=("hlvis", "visual"))
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    ap.add_argument("--no-crsa", action="store_true")
    ap.add_argument("--no-wrsa", action="store_true")
    ap.add_argument("--no-srpr", action="store_true")
    ap.add_argument("--model-glob", default="*.h5")
    ap.add_argument("--model-batch", type=int, default=None,
                    help="0-based batch index when running in parallel.")
    ap.add_argument("--models-per-batch", type=int, default=10)
    return ap.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    feature_files = sorted(args.features.glob(args.model_glob))
    if not feature_files:
        raise SystemExit(f"No feature files matching {args.features}/{args.model_glob}")

    if args.model_batch is not None:
        s = args.model_batch * args.models_per_batch
        feature_files = feature_files[s : s + args.models_per_batch]
        _log(f"Batch {args.model_batch}: {len(feature_files)} models")

    do_crsa = not args.no_crsa
    do_wrsa = not args.no_wrsa
    do_srpr = not args.no_srpr

    all_rows: List[dict] = []
    for subject in args.subjects:
        # Pool may be the subject's own (default) or "shared" — set via
        # --pool. Every split's train/test image_ids are a subset of the
        # benchmark's response_data, since pool selection happens in the
        # benchmark.
        pool = args.pool or subject
        _log(f"=== Subject {subject}, pool={pool} ===")

        bench = LAIONBenchmark(
            subject=subject,
            voxel_set=args.voxel_set,
            pool=pool,
        )
        _log(f"  loaded benchmark: {bench.n_stimuli} stimuli, "
             f"{bench.response_data.shape[0]} voxels")

        split_names = args.splits or list_splits()
        _log(f"  running {len(split_names)} splits: {split_names}")

        for split_name in split_names:
            sp = load_split(split_name, pool=pool)
            for variant in sp.variants:
                rows: List[dict] = []
                for h5_path in tqdm(
                    feature_files,
                    desc=f"{subject} {split_name} v{variant.variant_id}",
                ):
                    try:
                        rows.extend(evaluate_model_on_split(
                            h5_path, bench,
                            variant.train_ids, variant.test_ids,
                            args.alphas, do_crsa, do_wrsa, do_srpr,
                            split_name=split_name,
                            variant_id=variant.variant_id,
                        ))
                    except Exception as e:
                        _log(f"  ERROR {h5_path.name} on {split_name} v{variant.variant_id}: {e}")
                        traceback.print_exc()

                df = pd.DataFrame(rows)
                if args.model_batch is not None:
                    out_path = args.out / (
                        f"results_{subject}_{split_name}_v{variant.variant_id}_"
                        f"batch{args.model_batch}.parquet"
                    )
                else:
                    out_path = args.out / (
                        f"results_{subject}_{split_name}_v{variant.variant_id}.parquet"
                    )
                df.to_parquet(out_path, index=False)
                _log(f"  wrote {len(df)} rows → {out_path}")
                all_rows.extend(rows)

    if args.model_batch is None:
        all_df = pd.DataFrame(all_rows)
        all_path = args.out / "results_all.parquet"
        all_df.to_parquet(all_path, index=False)
        _log(f"wrote {len(all_df)} rows → {all_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
