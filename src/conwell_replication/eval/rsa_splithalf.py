#!/usr/bin/env python3
"""Conwell-style split-half RSA evaluation (sanity-replication of rsa_20260223_154344).

For each subject × model × layer:

  * cRSA  — corr(model_RDM[half], brain_RDM[half]) for each half
  * WRSA  — fit ridge on half_a → predict half_b → corr(predicted_RDM, brain_RDM)
  * SRPR  — single-voxel Pearson r between predicted and observed responses

The splits are "even" / "odd" indices of the SHARED stimulus pool (mirroring
``benchmark.response_data[:, ::2]`` vs ``[:, 1::2]`` from the original code).
This evaluator is meant for internal sanity-checking against
``rsa_20260223_154344/results_all.parquet`` — confirm the new pipeline
reproduces those numbers within tolerance before trusting the min_nn results.

Outputs (under ``--out``):

  * ``results_<subject>.parquet`` — long-form rows with columns
        score, score_set, eval_type, region, model, model_layer,
        model_layer_index, subject
  * ``results_all.parquet``       — concatenation of all subjects (only
                                    written when --all-subjects pass completes)
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

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
# Per-(subject, model) evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    h5_path: Path,
    benchmark: LAIONBenchmark,
    alphas: List[float],
    do_crsa: bool,
    do_wrsa: bool,
    do_srpr: bool,
) -> List[dict]:
    """Run cRSA/WRSA/SRPR for one model on one subject's split-half data."""
    model_name = h5_path.stem.replace("-", "/")
    t0 = time.time()

    features, h5_ids = load_features(h5_path)
    bench_ids = benchmark.stimulus_data["image_name"].tolist()
    aligned = {ln: align_features(f, h5_ids, bench_ids) for ln, f in features.items()}

    # Brain split-half data
    Y = benchmark.response_data.to_numpy()       # (n_voxels, n_stimuli)
    Y_train = Y[:, ::2].T                         # (n_train, n_voxels)
    Y_test  = Y[:, 1::2].T
    rdm_train = rdm_from_responses(Y_train)       # brain RDM, half_a
    rdm_test  = rdm_from_responses(Y_test)        # brain RDM, half_b

    region = benchmark.voxel_set
    subject = benchmark.subject
    rows: List[dict] = []

    for layer_idx, (layer_name, feats) in enumerate(aligned.items()):
        feats_train = feats[::2, :]
        feats_test  = feats[1::2, :]

        # cRSA: classical RSA (no ridge)
        if do_crsa:
            for ss, mfeats, target in (
                ("train", feats_train, rdm_train),
                ("test",  feats_test,  rdm_test),
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
                })

        # Ridge once per layer; reused for WRSA + SRPR
        if do_wrsa or do_srpr:
            try:
                pred_train, pred_test, _ridge = fit_ridge(
                    feats_train, feats_test, Y_train, alphas
                )
            except Exception as e:
                _log(f"    Ridge failed for {layer_name}: {e}")
                continue

            if do_wrsa:
                for ss, pred, target in (
                    ("train", pred_train, rdm_train),
                    ("test",  pred_test,  rdm_test),
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
                    })

    _log(f"  {model_name}: {len(rows)} rows in {time.time() - t0:.1f}s")
    return rows


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True,
                    help="Directory of per-model SRP feature .h5 files.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for results parquets.")
    ap.add_argument("--subjects", nargs="+",
                    default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"])
    ap.add_argument("--voxel-set", default="hlvis", choices=("hlvis", "visual"))
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    ap.add_argument("--no-crsa", action="store_true")
    ap.add_argument("--no-wrsa", action="store_true")
    ap.add_argument("--no-srpr", action="store_true")
    ap.add_argument("--model-glob", default="*.h5",
                    help="Restrict to a glob (e.g. 'timm_*.h5').")
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
        _log(f"=== Subject {subject} ===")
        bench = LAIONBenchmark(
            subject=subject,
            voxel_set=args.voxel_set,
            image_pool="shared",
        )
        _log(f"  loaded benchmark: {bench.n_stimuli} stimuli, "
             f"{bench.response_data.shape[0]} voxels")

        rows: List[dict] = []
        for h5_path in tqdm(feature_files, desc=f"{subject} models"):
            try:
                rows.extend(evaluate_model(
                    h5_path, bench, args.alphas, do_crsa, do_wrsa, do_srpr,
                ))
            except Exception as e:
                _log(f"  ERROR on {h5_path.name}: {e}")
                traceback.print_exc()

        df = pd.DataFrame(rows)
        if args.model_batch is not None:
            out_path = args.out / f"results_{subject}_batch{args.model_batch}.parquet"
        else:
            out_path = args.out / f"results_{subject}.parquet"
        df.to_parquet(out_path, index=False)
        _log(f"  wrote {len(df)} rows → {out_path}")
        all_rows.extend(rows)

    if args.model_batch is None and len(args.subjects) > 1:
        all_df = pd.DataFrame(all_rows)
        all_path = args.out / "results_all.parquet"
        all_df.to_parquet(all_path, index=False)
        _log(f"wrote {len(all_df)} rows → {all_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
