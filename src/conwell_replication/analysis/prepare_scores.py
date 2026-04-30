#!/usr/bin/env python3
"""Best-layer score selection (Conwell convention).

For each (model × eval_type × subject [× split × variant]):
  1. find the layer with the highest TRAIN score
  2. report the TEST score for that same layer

Inputs:
  --results        Path to a parquet produced by ``rsa_splithalf`` /
                   ``rsa_min_nn``. Required columns:
                       score, score_set, eval_type, model, model_layer,
                       model_layer_index, subject
                   Optional (min_nn): split, variant
  --metadata       model_metadata.csv (defaults to packaged resource)
  --noise-ceiling  noise_ceilings.csv (per subject, with nc_pearson column)

Outputs (under --out):
  best_layer_scores.csv   — long form, one row per group above
  mean_scores.csv         — mean test score across subjects per
                            (model × eval_type [× split × variant])
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

_PKG_RESOURCES = Path(__file__).resolve().parents[3] / "resources"


def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--metadata", type=Path,
                    default=_PKG_RESOURCES / "model_metadata.csv")
    ap.add_argument("--noise-ceiling", type=Path, default=None,
                    help="Optional path to noise_ceilings.csv. If provided "
                         "we add nc_pearson and eve columns.")
    ap.add_argument("--region", default="hlvis")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args(argv)


def _group_keys(df: pd.DataFrame) -> List[str]:
    keys = ["model", "eval_type", "subject"]
    for k in ("split", "variant"):
        if k in df.columns:
            keys.append(k)
    return keys


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.results)
    if "region" in df.columns:
        df = df[df["region"] == args.region].copy()

    keys = _group_keys(df)
    print(f"Grouping by: {keys}")
    print(f"Input rows: {len(df)}, models: {df['model'].nunique()}, "
          f"eval_types: {df['eval_type'].unique().tolist()}")

    rows = []
    for tup, group in df.groupby(keys, sort=False):
        train = group[group["score_set"] == "train"]
        test = group[group["score_set"] == "test"]
        if train.empty or test.empty:
            continue
        best = train.loc[train["score"].idxmax()]
        best_layer = best["model_layer"]
        test_row = test[test["model_layer"] == best_layer]
        if test_row.empty:
            continue

        out = dict(zip(keys, tup if isinstance(tup, tuple) else (tup,)))
        out.update({
            "best_layer":       best_layer,
            "best_layer_index": int(best["model_layer_index"]),
            "train_score":      float(best["score"]),
            "test_score":       float(test_row["score"].iloc[0]),
        })
        rows.append(out)

    scores = pd.DataFrame(rows)
    print(f"Best-layer scores: {len(scores)} rows")

    # Merge metadata
    if args.metadata.exists():
        meta = pd.read_csv(args.metadata)
        scores = scores.merge(meta, on="model", how="left")
    else:
        print(f"WARNING: metadata not found at {args.metadata}; skipping merge")

    # Merge noise ceilings
    if args.noise_ceiling is not None and args.noise_ceiling.exists():
        nc = pd.read_csv(args.noise_ceiling)
        if "nc_pearson" in nc.columns:
            if {"split", "variant", "score_set"}.issubset(nc.columns) and {"split", "variant"}.issubset(scores.columns):
                # Per-(subject, split, variant) noise ceiling using TEST set
                test_nc = nc[nc["score_set"] == "test"][["subject", "split", "variant", "nc_pearson"]]
                scores = scores.merge(test_nc, on=["subject", "split", "variant"], how="left")
            else:
                # Per-subject noise ceiling
                scores = scores.merge(
                    nc[["subject", "nc_pearson"]], on="subject", how="left",
                )
            scores["eve"] = scores["test_score"] ** 2 / scores["nc_pearson"] ** 2

    scores.to_csv(args.out / "best_layer_scores.csv", index=False)
    print(f"Saved → {args.out / 'best_layer_scores.csv'}")

    # Mean across subjects (and variant, if present)
    mean_keys = ["model", "eval_type"]
    if "split" in scores.columns:
        mean_keys.append("split")
    mean_scores = (
        scores.groupby(mean_keys)
        .agg(
            mean_test=("test_score", "mean"),
            std_test=("test_score", "std"),
            mean_train=("train_score", "mean"),
            n_subjects=("subject", "nunique"),
        )
        .reset_index()
    )
    if args.metadata.exists():
        mean_scores = mean_scores.merge(meta, on="model", how="left")
    mean_scores.to_csv(args.out / "mean_scores.csv", index=False)
    print(f"Saved → {args.out / 'mean_scores.csv'}")

    for et in ("crsa", "wrsa"):
        sub = mean_scores[mean_scores["eval_type"] == et].sort_values("mean_test", ascending=False)
        print(f"\n=== Top 10 {et.upper()} ===")
        for _, row in sub.head(10).iterrows():
            split_label = f" [{row['split']}]" if "split" in row else ""
            print(f"  {str(row['model'])[:60]:60s} {row['mean_test']:.4f}{split_label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
