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
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from conwell_replication.data import LAIONBenchmark
from conwell_replication.data.roi_masks import available_roi_columns
from conwell_replication.eval._common import (
    columnwise_pearson,
    compare_rdms,
    fit_ridge,
    mean_columnwise_pearson,
    rdm_from_responses,
    take_h5_rows,
)
from conwell_replication.eval._voxelwise import VoxelwiseScoreWriter


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _is_ood_image_id(image_id: str) -> bool:
    return image_id.startswith("shared_4rep_OOD_")


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
    voxelwise_writer: Optional[VoxelwiseScoreWriter] = None,
    rois: Optional[List[str]] = None,
    min_roi_voxels: int = 10,
    include_ood: bool = False,
    extra_features_dir: Optional[Path] = None,
) -> List[dict]:
    """Run cRSA/WRSA/SRPR for one model on one subject's split-half data."""
    model_name = h5_path.stem.replace("-", "/")
    t0 = time.time()

    # Brain split-half data. LAIONBenchmark(pool="shared") includes OOD images
    # since the min-nn OOD split needs them. We drop them by default to match
    # Conwell's clean train/test split, but ``include_ood=True`` keeps them
    # — useful for parity with the older deepvision pipeline that ran wRSA
    # over all 1,492 shared images.
    if include_ood:
        bench_ids = list(benchmark.stimulus_data["image_name"])
    else:
        bench_ids = [
            iid for iid in benchmark.stimulus_data["image_name"].tolist()
            if not _is_ood_image_id(iid)
        ]
    Y = benchmark.response_data.loc[:, bench_ids].to_numpy()  # (n_voxels, n_stimuli)
    Y_train = Y[:, ::2].T                         # (n_train, n_voxels)
    Y_test  = Y[:, 1::2].T
    roi_names = available_roi_columns(
        benchmark.metadata, requested=rois, min_voxels=min_roi_voxels,
    )
    if not roi_names:
        raise ValueError(
            f"No ROI columns with >= {min_roi_voxels} voxels. "
            f"Requested={rois}; metadata columns={list(benchmark.metadata.columns)}"
        )
    roi_indices = {
        roi: benchmark.voxel_indices(roi).astype(np.int64)
        for roi in roi_names
    }
    rdm_train_by_roi = {
        roi: rdm_from_responses(Y_train[:, idx])
        for roi, idx in roi_indices.items()
    }
    rdm_test_by_roi = {
        roi: rdm_from_responses(Y_test[:, idx])
        for roi, idx in roi_indices.items()
    }

    fit_region = benchmark.voxel_set
    subject = benchmark.subject
    rows: List[dict] = []
    verbose_layers = os.environ.get("CONWELL_EVAL_VERBOSE_LAYERS") == "1"

    extra_path = extra_features_dir / h5_path.name if extra_features_dir else None
    extra_f = None
    try:
        if extra_path is not None and extra_path.exists():
            extra_f = h5py.File(extra_path, "r")

        with h5py.File(h5_path, "r") as f:
            if "features_srp" not in f:
                raise ValueError(f"{h5_path} has no /features_srp group")
            if "image_ids" not in f:
                raise ValueError(f"{h5_path} has no /image_ids dataset")

            raw_ids = f["image_ids"][:]
            h5_ids = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw_ids
            ]
            id_to_idx = {iid: i for i, iid in enumerate(h5_ids)}

            extra_id_to_idx: Dict[str, int] = {}
            if extra_f is not None:
                if "features_srp" not in extra_f:
                    raise ValueError(f"{extra_path} has no /features_srp group")
                if "image_ids" not in extra_f:
                    raise ValueError(f"{extra_path} has no /image_ids dataset")
                raw_extra_ids = extra_f["image_ids"][:]
                extra_id_to_idx = {
                    (x.decode("utf-8") if isinstance(x, bytes) else str(x)): i
                    for i, x in enumerate(raw_extra_ids)
                }

            missing = [
                iid for iid in bench_ids
                if iid not in id_to_idx and iid not in extra_id_to_idx
            ]
            if missing:
                raise KeyError(
                    f"{len(missing)} benchmark image_ids missing from "
                    f"{h5_path.name} (and extra={extra_path}); e.g. {missing[:3]}. "
                    "Re-extract features over the relevant pool."
                )
            primary_rows = np.fromiter(
                (id_to_idx.get(iid, -1) for iid in bench_ids),
                dtype=np.int64,
                count=len(bench_ids),
            )
            extra_rows = np.fromiter(
                (extra_id_to_idx.get(iid, -1) for iid in bench_ids),
                dtype=np.int64,
                count=len(bench_ids),
            )

            grp = f["features_srp"]
            extra_grp = extra_f["features_srp"] if extra_f is not None else None
            layer_names = list(grp.keys())
            for layer_idx, layer_name in enumerate(layer_names):
                t_layer = time.time()
                primary_dataset = grp[layer_name]
                primary_mask = primary_rows >= 0
                if primary_mask.all():
                    feats = take_h5_rows(primary_dataset, primary_rows)
                else:
                    if extra_grp is None or layer_name not in extra_grp:
                        raise KeyError(
                            f"missing rows for {h5_path.name} layer {layer_name} "
                            "and no compatible extra-features layer"
                        )
                    extra_dataset = extra_grp[layer_name]
                    feats = np.empty(
                        (len(bench_ids), primary_dataset.shape[1]),
                        dtype=primary_dataset.dtype,
                    )
                    if primary_mask.any():
                        feats[primary_mask] = take_h5_rows(
                            primary_dataset, primary_rows[primary_mask]
                        )
                    extra_mask = extra_rows >= 0
                    if extra_mask.any():
                        feats[extra_mask] = take_h5_rows(
                            extra_dataset, extra_rows[extra_mask]
                        )
                feats_train = feats[::2, :]
                feats_test  = feats[1::2, :]
                if verbose_layers:
                    _log(f"    {model_name} {layer_name}: read/slice {time.time() - t_layer:.1f}s")

                # cRSA: classical RSA (no ridge)
                t_stage = time.time()
                if do_crsa:
                    for ss, mfeats, targets in (
                        ("train", feats_train, rdm_train_by_roi),
                        ("test",  feats_test,  rdm_test_by_roi),
                    ):
                        m_rdm = rdm_from_responses(mfeats)
                        for roi, target in targets.items():
                            rows.append({
                                "score":             float(compare_rdms(m_rdm, target)),
                                "score_set":         ss,
                                "eval_type":         "crsa",
                                "region":            roi,
                                "fit_region":        fit_region,
                                "n_voxels":          int(roi_indices[roi].size),
                                "model":             model_name,
                                "model_layer":       layer_name,
                                "model_layer_index": layer_idx,
                                "subject":           subject,
                            })
                    if verbose_layers:
                        _log(f"    {model_name} {layer_name}: crsa {time.time() - t_stage:.1f}s")

                # Ridge once per layer; reused for WRSA + SRPR
                if do_wrsa or do_srpr:
                    t_stage = time.time()
                    try:
                        pred_train, pred_test, _ridge = fit_ridge(
                            feats_train, feats_test, Y_train, alphas
                        )
                    except Exception as e:
                        _log(f"    Ridge failed for {layer_name}: {e}")
                        continue
                    if verbose_layers:
                        _log(f"    {model_name} {layer_name}: ridge {time.time() - t_stage:.1f}s")

                    if do_wrsa:
                        t_stage = time.time()
                        for roi, voxel_idx in roi_indices.items():
                            for ss, pred, targets in (
                                ("train", pred_train, rdm_train_by_roi),
                                ("test",  pred_test,  rdm_test_by_roi),
                            ):
                                m_rdm = rdm_from_responses(pred[:, voxel_idx])
                                rows.append({
                                    "score":             float(compare_rdms(m_rdm, targets[roi])),
                                    "score_set":         ss,
                                    "eval_type":         "wrsa",
                                    "region":            roi,
                                    "fit_region":        fit_region,
                                    "n_voxels":          int(voxel_idx.size),
                                    "model":             model_name,
                                    "model_layer":       layer_name,
                                    "model_layer_index": layer_idx,
                                    "subject":           subject,
                                })
                        if verbose_layers:
                            _log(f"    {model_name} {layer_name}: wrsa {time.time() - t_stage:.1f}s")

                    if do_srpr:
                        t_stage = time.time()
                        for ss, pred, ytrue in (
                            ("train", pred_train, Y_train),
                            ("test",  pred_test,  Y_test),
                        ):
                            scores = columnwise_pearson(ytrue, pred)
                            if voxelwise_writer is not None and voxelwise_writer.enabled:
                                voxelwise_writer.add(
                                    scores,
                                    {
                                        "score_set":         ss,
                                        "eval_type":         "srpr",
                                        "region":            fit_region,
                                        "fit_region":        fit_region,
                                        "n_voxels":          int(ytrue.shape[1]),
                                        "model":             model_name,
                                        "model_layer":       layer_name,
                                        "model_layer_index": layer_idx,
                                        "subject":           subject,
                                    },
                                )
                            for roi, voxel_idx in roi_indices.items():
                                rows.append({
                                    "score":             float(scores[voxel_idx].mean(dtype=np.float64)),
                                    "score_set":         ss,
                                    "eval_type":         "srpr",
                                    "region":            roi,
                                    "fit_region":        fit_region,
                                    "n_voxels":          int(voxel_idx.size),
                                    "model":             model_name,
                                    "model_layer":       layer_name,
                                    "model_layer_index": layer_idx,
                                    "subject":           subject,
                                })
                        if verbose_layers:
                            _log(f"    {model_name} {layer_name}: srpr {time.time() - t_stage:.1f}s")

                if verbose_layers:
                    _log(f"    {model_name} {layer_name}: {time.time() - t_layer:.1f}s")
    finally:
        if extra_f is not None:
            extra_f.close()

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
    ap.add_argument("--voxel-set", default="hlvis", choices=("hlvis", "visual", "otc"))
    ap.add_argument("--rois", nargs="+", default=None,
                    help="ROI columns to score. Default: Conwell-style OTC "
                         "and component category/object ROIs present in the cache.")
    ap.add_argument("--min-roi-voxels", type=int, default=10,
                    help="Skip ROI columns with fewer voxels. Default: 10.")
    ap.add_argument("--ncsnr-threshold", type=float, default=0.2,
                    help="Drop voxels with ncsnr <= this from the response "
                         "matrix (matches Conwell's NCSNR > 0.2 selection). "
                         "Set to a negative value to disable. Default: 0.2.")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    ap.add_argument("--no-crsa", action="store_true")
    ap.add_argument("--no-wrsa", action="store_true")
    ap.add_argument("--no-srpr", action="store_true")
    ap.add_argument("--model-glob", default="*.h5",
                    help="Restrict to a glob (e.g. 'timm_*.h5').")
    ap.add_argument("--model-list", type=Path, default=None,
                    help="Optional newline-delimited list of .h5 paths or "
                         "basenames. Preserves listed order and overrides "
                         "--model-glob.")
    ap.add_argument("--model-batch", type=int, default=None,
                    help="0-based batch index when running in parallel.")
    ap.add_argument("--models-per-batch", type=int, default=10)
    ap.add_argument("--srpr-voxelwise", choices=("none", "all"), default="none",
                    help="Write voxel-wise SRPR Pearson-r vectors as sidecars.")
    ap.add_argument("--srpr-voxelwise-dir", type=Path, default=None,
                    help="Sidecar directory (default: OUT/srpr_voxelwise).")
    ap.add_argument("--include-ood", action="store_true",
                    help="Keep shared_4rep_OOD_* images in the train/test split. "
                         "Requires --extra-features-dir if model h5s don't include "
                         "OOD rows. Mirrors the older deepvision-fmri eval.")
    ap.add_argument("--extra-features-dir", type=Path, default=None,
                    help="Optional sidecar feature directory with the same h5 "
                         "basenames; used for image_ids absent from --features "
                         "(typically OOD features).")
    return ap.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.model_list is not None:
        feature_files = []
        for line in args.model_list.read_text().splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            p = Path(item)
            if not p.is_absolute():
                p = args.features / p
            if p.suffix != ".h5":
                p = p.with_suffix(".h5")
            feature_files.append(p)
        missing = [str(p) for p in feature_files if not p.exists()]
        if missing:
            raise SystemExit(
                f"{len(missing)} model-list files do not exist "
                f"(e.g. {missing[:3]})"
            )
    else:
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
            pool="shared",
            ncsnr_threshold=(args.ncsnr_threshold if args.ncsnr_threshold >= 0 else None),
        )
        _log(f"  loaded benchmark: {bench.n_stimuli} stimuli, "
             f"{bench.response_data.shape[0]} voxels")

        rows: List[dict] = []
        if args.model_batch is not None:
            out_path = args.out / f"results_{subject}_batch{args.model_batch}.parquet"
        else:
            out_path = args.out / f"results_{subject}.parquet"
        voxelwise_dir = args.srpr_voxelwise_dir or (args.out / "srpr_voxelwise")
        voxelwise_writer = VoxelwiseScoreWriter(
            voxelwise_dir,
            out_path.stem,
            enabled=do_srpr and args.srpr_voxelwise == "all",
        )
        for h5_path in tqdm(feature_files, desc=f"{subject} models"):
            try:
                rows.extend(evaluate_model(
                    h5_path, bench, args.alphas, do_crsa, do_wrsa, do_srpr,
                    voxelwise_writer=voxelwise_writer,
                    rois=args.rois,
                    min_roi_voxels=args.min_roi_voxels,
                    include_ood=args.include_ood,
                    extra_features_dir=args.extra_features_dir,
                ))
            except Exception as e:
                _log(f"  ERROR on {h5_path.name}: {e}")
                traceback.print_exc()

        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        _log(f"  wrote {len(df)} rows → {out_path}")
        sidecars = voxelwise_writer.write()
        if sidecars is not None:
            _log(f"  wrote voxel-wise SRPR → {sidecars[0]} and {sidecars[1]}")
        all_rows.extend(rows)

    if args.model_batch is None and len(args.subjects) > 1:
        all_df = pd.DataFrame(all_rows)
        all_path = args.out / "results_all.parquet"
        all_df.to_parquet(all_path, index=False)
        _log(f"wrote {len(all_df)} rows → {all_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
