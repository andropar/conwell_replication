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
splits each have a single variant; ``tau`` and ``ood`` also have a single
variant_id=0. The ``ood`` split fits once on the pool's regular images and
then reports test scores both on all OOD images and on each OOD category.

Outputs (under ``--out``):

  * ``results_<subject>_<split>.parquet`` — per (subject, split) parquet
  * ``results_all.parquet``               — all rows concatenated
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from laion_fmri.splits import list_ood_types, list_splits, load_split

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


def _ood_type_for_image_id(image_id: str) -> Optional[str]:
    """Return the LAION-fMRI OOD category encoded in an image_id."""
    for ood_type in list_ood_types():
        if image_id.startswith(f"shared_4rep_OOD_{ood_type}_"):
            return ood_type
    return None


def _make_test_slices(
    test_ids: List[str],
    split_name: str,
) -> List[Tuple[Optional[str], np.ndarray]]:
    """Return test subsets to report, without changing the fit."""
    full = np.arange(len(test_ids), dtype=np.int64)
    if split_name != "ood":
        return [(None, full)]

    slices: List[Tuple[Optional[str], np.ndarray]] = [("all", full)]
    image_types = [_ood_type_for_image_id(iid) for iid in test_ids]
    for ood_type in list_ood_types():
        idx = np.fromiter(
            (i for i, typ in enumerate(image_types) if typ == ood_type),
            dtype=np.int64,
        )
        if idx.size:
            slices.append((ood_type, idx))
    return slices


def _is_full_slice(indices: np.ndarray, n: int) -> bool:
    return (
        indices.size == n
        and (n == 0 or (int(indices[0]) == 0 and int(indices[-1]) == n - 1))
    )


def _rdm_subset(rdm: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Take an RDM submatrix, returning the input for the full slice."""
    if _is_full_slice(indices, rdm.shape[0]):
        return rdm
    return rdm[np.ix_(indices, indices)]


def _response_subset(Y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Take response rows, returning the input for the full slice."""
    if _is_full_slice(indices, Y.shape[0]):
        return Y
    return Y[indices]


# ---------------------------------------------------------------------------
# Per-(subject, split, variant) brain-side context — built once, reused for
# all models on that split. Hoisting the brain ops out of the per-model loop
# saved ~50% of wall time on the smoke test (one set of (Y_train, Y_test,
# rdm_train_brain, rdm_test_brain) per split, vs recomputing 271×).
# ---------------------------------------------------------------------------
class SplitContext:
    """Brain-side state for one (subject, split, variant)."""
    __slots__ = (
        "train_ids", "test_ids",
        "Y_train", "Y_test",
        "roi_indices", "rdm_train_brain_by_roi", "rdm_test_brain_by_roi",
        "rdm_test_brain_by_roi_and_ood_type", "test_slices",
        "fit_region", "subject", "pool", "split_name", "variant_id",
    )

    def __init__(
        self,
        benchmark: LAIONBenchmark,
        train_ids: List[str],
        test_ids: List[str],
        split_name: str,
        variant_id: int,
        rois: Optional[List[str]] = None,
        min_roi_voxels: int = 10,
    ):
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.Y_train = benchmark.betas(train_ids, as_array=True).T  # (n_train, n_voxels)
        self.Y_test = benchmark.betas(test_ids, as_array=True).T
        roi_names = available_roi_columns(
            benchmark.metadata, requested=rois, min_voxels=min_roi_voxels,
        )
        if not roi_names:
            raise ValueError(
                f"No ROI columns with >= {min_roi_voxels} voxels. "
                f"Requested={rois}; metadata columns={list(benchmark.metadata.columns)}"
            )
        self.roi_indices = {
            roi: benchmark.voxel_indices(roi).astype(np.int64)
            for roi in roi_names
        }
        self.rdm_train_brain_by_roi = {
            roi: rdm_from_responses(self.Y_train[:, idx])
            for roi, idx in self.roi_indices.items()
        }
        self.rdm_test_brain_by_roi = {
            roi: rdm_from_responses(self.Y_test[:, idx])
            for roi, idx in self.roi_indices.items()
        }
        self.test_slices = _make_test_slices(test_ids, split_name)
        self.rdm_test_brain_by_roi_and_ood_type = {
            roi: {
                ood_type: _rdm_subset(rdm, indices)
                for ood_type, indices in self.test_slices
            }
            for roi, rdm in self.rdm_test_brain_by_roi.items()
        }
        self.fit_region = benchmark.voxel_set
        self.subject = benchmark.subject
        self.pool = benchmark.pool
        self.split_name = split_name
        self.variant_id = variant_id


def evaluate_model_on_split(
    h5_path: Path,
    ctx: SplitContext,
    alphas: List[float],
    do_crsa: bool,
    do_wrsa: bool,
    do_srpr: bool,
    voxelwise_writer: Optional[VoxelwiseScoreWriter] = None,
    extra_features_dir: Optional[Path] = None,
) -> List[dict]:
    """Run cRSA/WRSA/SRPR for one model on a SplitContext (brain-side cached)."""
    model_name = h5_path.stem.replace("-", "/")
    t0 = time.time()

    rows: List[dict] = []

    extra_path = extra_features_dir / h5_path.name if extra_features_dir else None
    extra_f = None
    try:
        with h5py.File(h5_path, "r") as f:
            if extra_path is not None and extra_path.exists():
                extra_f = h5py.File(extra_path, "r")
            elif extra_path is not None and split_name == "ood":
                raise FileNotFoundError(
                    f"OOD split needs extra feature rows but {extra_path} does not exist"
                )

            rows = _evaluate_model_on_split_open_h5(
                h5_path=h5_path,
                f=f,
                extra_f=extra_f,
                ctx=ctx,
                alphas=alphas,
                do_crsa=do_crsa,
                do_wrsa=do_wrsa,
                do_srpr=do_srpr,
                voxelwise_writer=voxelwise_writer,
                model_name=model_name,
                t0=t0,
            )
    finally:
        if extra_f is not None:
            extra_f.close()

    return rows


def _evaluate_model_on_split_open_h5(
    h5_path: Path,
    f: h5py.File,
    extra_f: Optional[h5py.File],
    ctx: SplitContext,
    alphas: List[float],
    do_crsa: bool,
    do_wrsa: bool,
    do_srpr: bool,
    voxelwise_writer: Optional[VoxelwiseScoreWriter],
    model_name: str,
    t0: float,
) -> List[dict]:
    """Implementation for one model with primary and optional extra H5 open."""
    Y_train = ctx.Y_train
    Y_test = ctx.Y_test
    roi_indices = ctx.roi_indices
    rdm_train_brain_by_roi = ctx.rdm_train_brain_by_roi
    rdm_test_brain_by_roi_and_ood_type = ctx.rdm_test_brain_by_roi_and_ood_type
    test_slices = ctx.test_slices
    fit_region = ctx.fit_region
    subject = ctx.subject
    pool = ctx.pool
    split_name = ctx.split_name
    variant_id = ctx.variant_id
    rows: List[dict] = []
    verbose_layers = os.environ.get("CONWELL_EVAL_VERBOSE_LAYERS") == "1"

    with f:
        if "features_srp" not in f:
            raise ValueError(f"{h5_path} has no /features_srp group")
        if "image_ids" not in f:
            raise ValueError(f"{h5_path} has no /image_ids dataset")
        if extra_f is not None:
            if "features_srp" not in extra_f:
                raise ValueError(f"{extra_f.filename} has no /features_srp group")
            if "image_ids" not in extra_f:
                raise ValueError(f"{extra_f.filename} has no /image_ids dataset")

        raw_ids = f["image_ids"][:]
        h5_ids = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw_ids
        ]
        id_to_idx = {iid: i for i, iid in enumerate(h5_ids)}
        extra_id_to_idx = {}
        if extra_f is not None:
            raw_extra_ids = extra_f["image_ids"][:]
            extra_ids = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in raw_extra_ids
            ]
            extra_id_to_idx = {iid: i for i, iid in enumerate(extra_ids)}

        def rows_for(target_ids: List[str]) -> np.ndarray:
            missing = [
                iid for iid in target_ids
                if iid not in id_to_idx and iid not in extra_id_to_idx
            ]
            if missing:
                raise KeyError(
                    f"{len(missing)} target image_ids missing from {h5_path.name} "
                    f"(e.g. {missing[:3]}). Re-extract features over this pool "
                    "or pass --extra-features-dir."
                )
            return np.fromiter(
                (id_to_idx.get(iid, -1) for iid in target_ids),
                dtype=np.int64,
                count=len(target_ids),
            )

        def extra_rows_for(target_ids: List[str]) -> np.ndarray:
            return np.fromiter(
                (extra_id_to_idx.get(iid, -1) for iid in target_ids),
                dtype=np.int64,
                count=len(target_ids),
            )

        def take_layer_rows(
            primary_dataset,
            layer_name: str,
            target_ids: List[str],
            primary_rows: np.ndarray,
            extra_rows: np.ndarray,
        ) -> np.ndarray:
            primary_mask = primary_rows >= 0
            if primary_mask.all():
                return take_h5_rows(primary_dataset, primary_rows)
            if extra_f is None:
                missing = [iid for iid, ok in zip(target_ids, primary_mask) if not ok]
                raise KeyError(
                    f"{len(missing)} target image_ids missing from {h5_path.name} "
                    f"(e.g. {missing[:3]}). Re-extract features over this pool."
                )
            extra_mask = extra_rows >= 0
            if not (primary_mask | extra_mask).all():
                missing = [
                    iid for iid, ok in zip(target_ids, primary_mask | extra_mask)
                    if not ok
                ]
                raise KeyError(
                    f"{len(missing)} target image_ids missing from primary+extra "
                    f"features (e.g. {missing[:3]})."
                )
            if layer_name not in extra_f["features_srp"]:
                raise KeyError(f"{extra_f.filename} has no layer {layer_name!r}")

            extra_dataset = extra_f["features_srp"][layer_name]
            out = np.empty(
                (len(target_ids), primary_dataset.shape[1]),
                dtype=primary_dataset.dtype,
            )
            if primary_mask.any():
                out[primary_mask, :] = take_h5_rows(
                    primary_dataset, primary_rows[primary_mask],
                )
            if extra_mask.any():
                out[extra_mask, :] = take_h5_rows(
                    extra_dataset, extra_rows[extra_mask],
                )
            return out

        train_rows = rows_for(ctx.train_ids)
        test_rows = rows_for(ctx.test_ids)
        train_extra_rows = extra_rows_for(ctx.train_ids)
        test_extra_rows = extra_rows_for(ctx.test_ids)
        grp = f["features_srp"]
        layer_names = list(grp.keys())

        for layer_idx, layer_name in enumerate(layer_names):
            t_layer = time.time()
            layer_dataset = grp[layer_name]
            ftrain = take_layer_rows(
                layer_dataset, layer_name, ctx.train_ids, train_rows, train_extra_rows,
            )
            ftest = take_layer_rows(
                layer_dataset, layer_name, ctx.test_ids, test_rows, test_extra_rows,
            )
            if verbose_layers:
                _log(f"    {model_name} {layer_name}: read/slice {time.time() - t_layer:.1f}s")

            # cRSA on raw features
            t_stage = time.time()
            if do_crsa:
                m_rdm_train = rdm_from_responses(ftrain)
                for roi, target in rdm_train_brain_by_roi.items():
                    rows.append({
                        "score":             float(compare_rdms(m_rdm_train, target)),
                        "score_set":         "train",
                        "eval_type":         "crsa",
                        "region":            roi,
                        "fit_region":        fit_region,
                        "n_voxels":          int(roi_indices[roi].size),
                        "model":             model_name,
                        "model_layer":       layer_name,
                        "model_layer_index": layer_idx,
                        "subject":           subject,
                        "pool":              pool,
                        "split":             split_name,
                        "variant":           variant_id,
                        "ood_type":          None,
                    })

                m_rdm_test = rdm_from_responses(ftest)
                for roi in roi_indices:
                    for ood_type, test_idx in test_slices:
                        rows.append({
                            "score":             float(compare_rdms(
                                _rdm_subset(m_rdm_test, test_idx),
                                rdm_test_brain_by_roi_and_ood_type[roi][ood_type],
                            )),
                            "score_set":         "test",
                            "eval_type":         "crsa",
                            "region":            roi,
                            "fit_region":        fit_region,
                            "n_voxels":          int(roi_indices[roi].size),
                            "model":             model_name,
                            "model_layer":       layer_name,
                            "model_layer_index": layer_idx,
                            "subject":           subject,
                            "pool":              pool,
                            "split":             split_name,
                            "variant":           variant_id,
                            "ood_type":          ood_type,
                        })
                if verbose_layers:
                    _log(f"    {model_name} {layer_name}: crsa {time.time() - t_stage:.1f}s")

            # Ridge once per layer; reused for WRSA + SRPR
            if do_wrsa or do_srpr:
                t_stage = time.time()
                try:
                    pred_train, pred_test, _ridge = fit_ridge(ftrain, ftest, Y_train, alphas)
                except Exception as e:
                    _log(f"    Ridge failed for {layer_name}: {e}")
                    continue
                if verbose_layers:
                    _log(f"    {model_name} {layer_name}: ridge {time.time() - t_stage:.1f}s")

                if do_wrsa:
                    t_stage = time.time()
                    for roi, voxel_idx in roi_indices.items():
                        pred_rdm_train = rdm_from_responses(pred_train[:, voxel_idx])
                        rows.append({
                            "score":             float(compare_rdms(
                                pred_rdm_train,
                                rdm_train_brain_by_roi[roi],
                            )),
                            "score_set":         "train",
                            "eval_type":         "wrsa",
                            "region":            roi,
                            "fit_region":        fit_region,
                            "n_voxels":          int(voxel_idx.size),
                            "model":             model_name,
                            "model_layer":       layer_name,
                            "model_layer_index": layer_idx,
                            "subject":           subject,
                            "pool":              pool,
                            "split":             split_name,
                            "variant":           variant_id,
                            "ood_type":          None,
                        })
                        pred_rdm_test = rdm_from_responses(pred_test[:, voxel_idx])
                        for ood_type, test_idx in test_slices:
                            rows.append({
                                "score":             float(compare_rdms(
                                    _rdm_subset(pred_rdm_test, test_idx),
                                    rdm_test_brain_by_roi_and_ood_type[roi][ood_type],
                                )),
                                "score_set":         "test",
                                "eval_type":         "wrsa",
                                "region":            roi,
                                "fit_region":        fit_region,
                                "n_voxels":          int(voxel_idx.size),
                                "model":             model_name,
                                "model_layer":       layer_name,
                                "model_layer_index": layer_idx,
                                "subject":           subject,
                                "pool":              pool,
                                "split":             split_name,
                                "variant":           variant_id,
                                "ood_type":          ood_type,
                            })
                    if verbose_layers:
                        _log(f"    {model_name} {layer_name}: wrsa {time.time() - t_stage:.1f}s")

                if do_srpr:
                    t_stage = time.time()
                    train_scores = columnwise_pearson(Y_train, pred_train)
                    if voxelwise_writer is not None and voxelwise_writer.enabled:
                        voxelwise_writer.add(
                            train_scores,
                            {
                                "score_set":         "train",
                                "eval_type":         "srpr",
                                "region":            fit_region,
                                "fit_region":        fit_region,
                                "n_voxels":          int(Y_train.shape[1]),
                                "model":             model_name,
                                "model_layer":       layer_name,
                                "model_layer_index": layer_idx,
                                "subject":           subject,
                                "pool":              pool,
                                "split":             split_name,
                                "variant":           variant_id,
                                "ood_type":          None,
                            },
                        )
                    for roi, voxel_idx in roi_indices.items():
                        rows.append({
                            "score":             float(train_scores[voxel_idx].mean(dtype=np.float64)),
                            "score_set":         "train",
                            "eval_type":         "srpr",
                            "region":            roi,
                            "fit_region":        fit_region,
                            "n_voxels":          int(voxel_idx.size),
                            "model":             model_name,
                            "model_layer":       layer_name,
                            "model_layer_index": layer_idx,
                            "subject":           subject,
                            "pool":              pool,
                            "split":             split_name,
                            "variant":           variant_id,
                            "ood_type":          None,
                        })
                    for ood_type, test_idx in test_slices:
                        ytrue = _response_subset(Y_test, test_idx)
                        pred = _response_subset(pred_test, test_idx)
                        test_scores = columnwise_pearson(ytrue, pred)
                        if voxelwise_writer is not None and voxelwise_writer.enabled:
                            voxelwise_writer.add(
                                test_scores,
                                {
                                    "score_set":         "test",
                                    "eval_type":         "srpr",
                                    "region":            fit_region,
                                    "fit_region":        fit_region,
                                    "n_voxels":          int(Y_train.shape[1]),
                                    "model":             model_name,
                                    "model_layer":       layer_name,
                                    "model_layer_index": layer_idx,
                                    "subject":           subject,
                                    "pool":              pool,
                                    "split":             split_name,
                                    "variant":           variant_id,
                                    "ood_type":          ood_type,
                                },
                            )
                        for roi, voxel_idx in roi_indices.items():
                            rows.append({
                                "score":             float(test_scores[voxel_idx].mean(dtype=np.float64)),
                                "score_set":         "test",
                                "eval_type":         "srpr",
                                "region":            roi,
                                "fit_region":        fit_region,
                                "n_voxels":          int(voxel_idx.size),
                                "model":             model_name,
                                "model_layer":       layer_name,
                                "model_layer_index": layer_idx,
                                "subject":           subject,
                                "pool":              pool,
                                "split":             split_name,
                                "variant":           variant_id,
                                "ood_type":          ood_type,
                            })
                    if verbose_layers:
                        _log(f"    {model_name} {layer_name}: srpr {time.time() - t_stage:.1f}s")

            if verbose_layers:
                _log(f"    {model_name} {layer_name}: {time.time() - t_layer:.1f}s")

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
                    help="Subset of split names to run (default: all LAION-fMRI splits).")
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
    ap.add_argument("--model-glob", default="*.h5")
    ap.add_argument("--model-list", type=Path, default=None,
                    help="Optional newline-delimited list of .h5 paths or "
                         "basenames. Preserves listed order and overrides "
                         "--model-glob.")
    ap.add_argument("--model-batch", type=int, default=None,
                    help="0-based batch index when running in parallel.")
    ap.add_argument("--models-per-batch", type=int, default=10)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip split outputs that already exist and are nonempty.")
    ap.add_argument("--srpr-voxelwise", choices=("none", "all"), default="none",
                    help="Write voxel-wise SRPR Pearson-r vectors as sidecars.")
    ap.add_argument("--srpr-voxelwise-dir", type=Path, default=None,
                    help="Sidecar directory (default: OUT/srpr_voxelwise).")
    ap.add_argument("--extra-features-dir", type=Path, default=None,
                    help="Optional feature directory with the same H5 basenames "
                         "for image_ids absent from --features, e.g. OOD-only "
                         "features.")
    return ap.parse_args(argv)


def _pool_slug(pool: str) -> str:
    return pool.replace("/", "-")


def _result_path(
    out_dir: Path,
    subject: str,
    pool: str,
    split_name: str,
    variant_id: int,
    model_batch: Optional[int],
) -> Path:
    prefix = (
        f"results_{subject}_pool-{_pool_slug(pool)}_"
        f"{split_name}_v{variant_id}"
    )
    if model_batch is not None:
        prefix += f"_batch{model_batch}"
    return out_dir / f"{prefix}.parquet"


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
            ncsnr_threshold=(args.ncsnr_threshold if args.ncsnr_threshold >= 0 else None),
        )
        _log(f"  loaded benchmark: {bench.n_stimuli} stimuli, "
             f"{bench.response_data.shape[0]} voxels")

        split_names = args.splits or list_splits()
        _log(f"  running {len(split_names)} splits: {split_names}")

        for split_name in split_names:
            sp = load_split(split_name, pool=pool)
            for variant in sp.variants:
                out_path = _result_path(
                    args.out, subject, pool, split_name,
                    variant.variant_id, args.model_batch,
                )
                voxelwise_dir = args.srpr_voxelwise_dir or (args.out / "srpr_voxelwise")
                voxelwise_writer = VoxelwiseScoreWriter(
                    voxelwise_dir,
                    out_path.stem,
                    enabled=do_srpr and args.srpr_voxelwise == "all",
                )
                voxelwise_done = (
                    not voxelwise_writer.enabled
                    or (
                        voxelwise_writer.score_path.exists()
                        and voxelwise_writer.score_path.stat().st_size > 0
                        and voxelwise_writer.index_path.exists()
                        and voxelwise_writer.index_path.stat().st_size > 0
                    )
                )
                if (
                    args.skip_existing
                    and out_path.exists()
                    and out_path.stat().st_size > 0
                    and voxelwise_done
                ):
                    _log(f"  skipping existing {out_path}")
                    continue

                # Build the brain-side context ONCE per (subject, split, variant)
                # and reuse it for every model. With 271 models this avoids 270
                # redundant brain-RDM computations per split.
                t_ctx = time.time()
                ctx = SplitContext(
                    bench,
                    variant.train_ids, variant.test_ids,
                    split_name=split_name, variant_id=variant.variant_id,
                    rois=args.rois, min_roi_voxels=args.min_roi_voxels,
                )
                _log(
                    f"  {subject} {split_name} v{variant.variant_id}: brain ctx ready "
                    f"(n_train={len(ctx.train_ids)}, n_test={len(ctx.test_ids)}, "
                    f"rois={list(ctx.roi_indices)}, {time.time() - t_ctx:.1f}s)"
                )

                rows: List[dict] = []
                for h5_path in tqdm(
                    feature_files,
                    desc=f"{subject} {split_name} v{variant.variant_id}",
                ):
                    try:
                        rows.extend(evaluate_model_on_split(
                            h5_path, ctx,
                            args.alphas, do_crsa, do_wrsa, do_srpr,
                            voxelwise_writer=voxelwise_writer,
                            extra_features_dir=args.extra_features_dir,
                        ))
                    except Exception as e:
                        _log(f"  ERROR {h5_path.name} on {split_name} v{variant.variant_id}: {e}")
                        traceback.print_exc()

                df = pd.DataFrame(rows)
                df.to_parquet(out_path, index=False)
                _log(f"  wrote {len(df)} rows → {out_path}")
                sidecars = voxelwise_writer.write()
                if sidecars is not None:
                    _log(f"  wrote voxel-wise SRPR → {sidecars[0]} and {sidecars[1]}")
                all_rows.extend(rows)

    if args.model_batch is None:
        all_df = pd.DataFrame(all_rows)
        all_path = args.out / "results_all.parquet"
        all_df.to_parquet(all_path, index=False)
        _log(f"wrote {len(all_df)} rows → {all_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
