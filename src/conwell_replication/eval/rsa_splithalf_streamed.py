#!/usr/bin/env python3
"""Colin-style streamed split-half RSA evaluation.

This evaluator is a validation path for the layer-selection question. It keeps
the current LAION split-half scoring logic, but avoids durable per-layer HDF5
feature stores:

  1. load one DeepNSD model,
  2. discover hook names and retain the full second half, matching Colin's
     released ``main_analysis.py`` convention,
  3. extract a chunk of retained layers over the split-half stimulus images,
  4. SRP-reduce those features in memory,
  5. immediately run cRSA / wRSA / SRPR and write only score rows.

The output schema matches ``rsa_splithalf.py`` closely enough for the existing
best-layer/plotting code to consume the resulting parquet shards.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.random_projection import SparseRandomProjection
from tqdm import tqdm

from conwell_replication import _vendor  # noqa: F401  side effect: model_opts importable
from conwell_replication.data import LAIONBenchmark
from conwell_replication.data.roi_masks import available_roi_columns
from conwell_replication.eval._common import (
    columnwise_pearson,
    compare_rdms,
    fit_ridge,
    rdm_from_responses,
)
from conwell_replication.extract.extract_features import load_model
from model_opts.feature_extraction import get_empty_feature_maps, get_feature_maps
from model_opts.model_options import get_model_options, get_recommended_transforms


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _is_ood_image_id(image_id: str) -> bool:
    return image_id.startswith("shared_4rep_OOD_")


def conwell_second_half_layers(feature_map_names: List[str]) -> List[str]:
    """Return the full second half of DeepNSD-discovered layer names.

    Colin's released ``source_code/pressures/main_analysis.py`` uses:

        layer_subset = model_layer_names[int(len(model_layer_names) / 2):]

    It does not skip pass-through module types. We intentionally preserve that
    behavior here because pass-through hooks can mark useful representational
    boundaries even when the module itself has no learned parameters.
    """
    return list(feature_map_names[len(feature_map_names) // 2 :])


def _chunks(items: List[str], chunk_size: int) -> Iterable[List[str]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


class _LazyImagesDataset(torch.utils.data.Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            return self.transform(img.convert("RGB"))


@dataclass
class SplitHalfContext:
    subject: str
    fit_region: str
    bench_ids: List[str]
    y_train: np.ndarray
    y_test: np.ndarray
    roi_indices: Dict[str, np.ndarray]
    rdm_train_by_roi: Dict[str, np.ndarray]
    rdm_test_by_roi: Dict[str, np.ndarray]


def _make_context(
    benchmark: LAIONBenchmark,
    canonical_ids: Optional[List[str]],
    rois: Optional[List[str]],
    min_roi_voxels: int,
    include_ood: bool,
) -> SplitHalfContext:
    if include_ood:
        bench_ids = list(benchmark.stimulus_data["image_name"])
    else:
        bench_ids = [
            iid
            for iid in benchmark.stimulus_data["image_name"].tolist()
            if not _is_ood_image_id(iid)
        ]

    if canonical_ids is None:
        canonical_ids = bench_ids
    elif set(bench_ids) != set(canonical_ids):
        missing = sorted(set(canonical_ids) - set(bench_ids))[:5]
        extra = sorted(set(bench_ids) - set(canonical_ids))[:5]
        raise ValueError(
            f"{benchmark.subject} shared image set differs from canonical set; "
            f"missing examples={missing}, extra examples={extra}"
        )

    # Reorder every subject's response matrix to the canonical extraction order.
    y = benchmark.response_data.loc[:, canonical_ids].to_numpy()
    y_train = y[:, ::2].T
    y_test = y[:, 1::2].T

    roi_names = available_roi_columns(
        benchmark.metadata, requested=rois, min_voxels=min_roi_voxels
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
    return SplitHalfContext(
        subject=benchmark.subject,
        fit_region=benchmark.voxel_set,
        bench_ids=canonical_ids,
        y_train=y_train,
        y_test=y_test,
        roi_indices=roi_indices,
        rdm_train_by_roi={
            roi: rdm_from_responses(y_train[:, idx])
            for roi, idx in roi_indices.items()
        },
        rdm_test_by_roi={
            roi: rdm_from_responses(y_test[:, idx])
            for roi, idx in roi_indices.items()
        },
    )


def _load_contexts(
    subjects: List[str],
    voxel_set: str,
    rois: Optional[List[str]],
    min_roi_voxels: int,
    ncsnr_threshold: float,
    include_ood: bool,
) -> List[SplitHalfContext]:
    contexts: List[SplitHalfContext] = []
    canonical_ids: Optional[List[str]] = None
    for subject in subjects:
        _log(f"=== Loading brain context {subject} ===")
        bench = LAIONBenchmark(
            subject=subject,
            voxel_set=voxel_set,
            pool="shared",
            ncsnr_threshold=(ncsnr_threshold if ncsnr_threshold >= 0 else None),
        )
        ctx = _make_context(
            bench,
            canonical_ids=canonical_ids,
            rois=rois,
            min_roi_voxels=min_roi_voxels,
            include_ood=include_ood,
        )
        canonical_ids = ctx.bench_ids
        contexts.append(ctx)
        _log(
            f"  {subject}: {len(ctx.bench_ids)} stimuli, "
            f"{ctx.y_train.shape[1]} voxels, rois={list(ctx.roi_indices)}"
        )
        del bench
        gc.collect()
    return contexts


def _parse_image_root_maps(values: Optional[List[str]]) -> List[tuple[str, str]]:
    maps: List[tuple[str, str]] = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"--image-root-map expects FROM=TO entries, got {value!r}"
            )
        src, dst = value.split("=", 1)
        if not src:
            raise ValueError(f"--image-root-map source prefix is empty: {value!r}")
        maps.append((src, dst))
    return maps


def _rewrite_image_path(path: str, root_maps: List[tuple[str, str]]) -> str:
    for src, dst in root_maps:
        if path.startswith(src):
            return dst + path[len(src) :]
    return path


def _image_paths_from_pool(
    pool_csv: Path,
    image_ids: List[str],
    root_maps: Optional[List[tuple[str, str]]] = None,
) -> List[str]:
    pool = pd.read_csv(pool_csv)
    if "image_id" not in pool.columns or "image_path" not in pool.columns:
        raise ValueError(
            f"{pool_csv} must have image_id and image_path columns for streamed eval"
        )
    maps = root_maps or []
    image_paths = [
        _rewrite_image_path(path, maps)
        for path in pool["image_path"].astype(str)
    ]
    id_to_path = dict(zip(pool["image_id"].astype(str), image_paths))
    missing = [iid for iid in image_ids if iid not in id_to_path]
    if missing:
        raise KeyError(
            f"{len(missing)} image_ids missing from {pool_csv}; e.g. {missing[:5]}"
        )
    paths = [id_to_path[iid] for iid in image_ids]
    absent = [p for p in paths if not Path(p).exists()]
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} image paths from {pool_csv} do not exist; e.g. {absent[:3]}"
        )
    return paths


def _fit_srps_for_layer_chunk(
    model: torch.nn.Module,
    transform,
    image_paths: List[str],
    layer_names: List[str],
    batch_size: int,
    num_workers: int,
    device: str,
    n_projections: int,
    srp_seed: int,
) -> Dict[str, np.ndarray]:
    n_samples = len(image_paths)
    dataloader = torch.utils.data.DataLoader(
        _LazyImagesDataset(image_paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    srp_per_layer: Dict[str, SparseRandomProjection] = {}
    reduced_per_layer: Dict[str, np.ndarray] = {}
    write_idx = 0

    for batch in tqdm(dataloader, desc="Extract/SRP batches", leave=False):
        batch = batch.to(device, non_blocking=True)
        with torch.no_grad():
            batch_maps = get_feature_maps(
                model,
                batch,
                layers_to_retain=layer_names,
                remove_duplicates=False,
                enforce_input_shape=True,
            )
        del batch

        batch_n: Optional[int] = None
        for layer_name in layer_names:
            feats = batch_maps.get(layer_name)
            if feats is None:
                continue
            feats_np = feats.reshape(feats.shape[0], -1).numpy()
            if batch_n is None:
                batch_n = feats_np.shape[0]

            if layer_name not in srp_per_layer:
                srp = SparseRandomProjection(
                    n_components=n_projections,
                    random_state=srp_seed,
                    dense_output=True,
                )
                srp.fit(feats_np)
                srp_per_layer[layer_name] = srp
                reduced_per_layer[layer_name] = np.empty(
                    (n_samples, n_projections), dtype=np.float32
                )

            reduced = srp_per_layer[layer_name].transform(feats_np)
            reduced_per_layer[layer_name][
                write_idx : write_idx + feats_np.shape[0], :
            ] = np.asarray(reduced, dtype=np.float32)

        del batch_maps
        if batch_n is not None:
            write_idx += batch_n

    if write_idx != n_samples:
        raise RuntimeError(f"wrote {write_idx} feature rows but expected {n_samples}")
    missing_layers = [layer for layer in layer_names if layer not in reduced_per_layer]
    if missing_layers:
        _log(f"  WARNING: no outputs for {len(missing_layers)} layers: {missing_layers[:5]}")
    return reduced_per_layer


def _score_layer(
    feats: np.ndarray,
    model_name: str,
    layer_name: str,
    layer_idx: int,
    contexts: List[SplitHalfContext],
    alphas: List[float],
    do_crsa: bool,
    do_wrsa: bool,
    do_srpr: bool,
) -> List[dict]:
    rows: List[dict] = []
    feats_train = feats[::2, :]
    feats_test = feats[1::2, :]

    model_rdms: Dict[str, np.ndarray] = {}
    if do_crsa:
        model_rdms = {
            "train": rdm_from_responses(feats_train),
            "test": rdm_from_responses(feats_test),
        }

    for ctx in contexts:
        if do_crsa:
            for score_set, m_rdm, targets in (
                ("train", model_rdms["train"], ctx.rdm_train_by_roi),
                ("test", model_rdms["test"], ctx.rdm_test_by_roi),
            ):
                for roi, target in targets.items():
                    rows.append(
                        {
                            "score": float(compare_rdms(m_rdm, target)),
                            "score_set": score_set,
                            "eval_type": "crsa",
                            "region": roi,
                            "fit_region": ctx.fit_region,
                            "n_voxels": int(ctx.roi_indices[roi].size),
                            "model": model_name,
                            "model_layer": layer_name,
                            "model_layer_index": layer_idx,
                            "subject": ctx.subject,
                        }
                    )

        if do_wrsa or do_srpr:
            try:
                pred_train, pred_test, _ridge = fit_ridge(
                    feats_train, feats_test, ctx.y_train, alphas
                )
            except Exception as exc:
                _log(
                    f"    Ridge failed for {model_name} {layer_name} "
                    f"{ctx.subject}: {exc}"
                )
                continue

            if do_wrsa:
                for roi, voxel_idx in ctx.roi_indices.items():
                    for score_set, pred, target in (
                        ("train", pred_train, ctx.rdm_train_by_roi[roi]),
                        ("test", pred_test, ctx.rdm_test_by_roi[roi]),
                    ):
                        pred_rdm = rdm_from_responses(pred[:, voxel_idx])
                        rows.append(
                            {
                                "score": float(compare_rdms(pred_rdm, target)),
                                "score_set": score_set,
                                "eval_type": "wrsa",
                                "region": roi,
                                "fit_region": ctx.fit_region,
                                "n_voxels": int(voxel_idx.size),
                                "model": model_name,
                                "model_layer": layer_name,
                                "model_layer_index": layer_idx,
                                "subject": ctx.subject,
                            }
                        )

            if do_srpr:
                for score_set, pred, ytrue in (
                    ("train", pred_train, ctx.y_train),
                    ("test", pred_test, ctx.y_test),
                ):
                    scores = columnwise_pearson(ytrue, pred)
                    for roi, voxel_idx in ctx.roi_indices.items():
                        rows.append(
                            {
                                "score": float(scores[voxel_idx].mean(dtype=np.float64)),
                                "score_set": score_set,
                                "eval_type": "srpr",
                                "region": roi,
                                "fit_region": ctx.fit_region,
                                "n_voxels": int(voxel_idx.size),
                                "model": model_name,
                                "model_layer": layer_name,
                                "model_layer_index": layer_idx,
                                "subject": ctx.subject,
                            }
                        )

            del pred_train, pred_test
            gc.collect()

    return rows


def _resolve_model_items(
    models_csv: Optional[Path],
    model_list: Optional[Path],
    model_options: dict,
) -> List[str]:
    if model_list is not None:
        items: List[str] = []
        for line in model_list.read_text().splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            key = Path(item).stem if item.endswith(".h5") else Path(item).name
            if key not in model_options and key.replace("-", "/") in model_options:
                key = key.replace("-", "/")
            items.append(key)
        return items

    if models_csv is None:
        raise ValueError("Either --models or --model-list is required")
    df = pd.read_csv(models_csv)
    if "option_key" not in df.columns:
        raise ValueError(f"{models_csv} must have an option_key column")
    return df["option_key"].astype(str).tolist()


def evaluate_model_streamed(
    option_key: str,
    model_options: dict,
    contexts: List[SplitHalfContext],
    image_paths: List[str],
    batch_size: int,
    num_workers: int,
    layer_chunk_size: int,
    device: str,
    n_projections: int,
    srp_seed: int,
    alphas: List[float],
    do_crsa: bool,
    do_wrsa: bool,
    do_srpr: bool,
) -> tuple[List[dict], pd.DataFrame]:
    model_name = option_key.replace("-", "/") if option_key not in model_options else option_key
    t0 = time.time()
    _log(f"=== {option_key} ===")

    _log("  Loading model...")
    model = load_model(option_key, model_options)
    transform = get_recommended_transforms(option_key, input_type="PIL")
    model = model.to(device)

    try:
        with Image.open(image_paths[0]) as img0:
            sample = transform(img0.convert("RGB"))
        input_shape = tuple(sample.shape)
        del sample

        _log("  Discovering DeepNSD hook layers...")
        all_layers = get_empty_feature_maps(
            model,
            inputs=None,
            input_shape=input_shape,
            remove_duplicates=True,
            names_only=True,
        )
        selected_layers = conwell_second_half_layers(all_layers)
        _log(
            f"  Selected {len(selected_layers)} second-half layers "
            f"from {len(all_layers)} discovered layers"
        )

        all_rows: List[dict] = []
        layer_meta: List[dict] = []
        for chunk_idx, layer_chunk in enumerate(_chunks(selected_layers, layer_chunk_size)):
            _log(
                f"  Layer chunk {chunk_idx + 1}/"
                f"{int(np.ceil(len(selected_layers) / layer_chunk_size))}: "
                f"{layer_chunk[0]} ... {layer_chunk[-1]}"
            )
            t_chunk = time.time()
            reduced = _fit_srps_for_layer_chunk(
                model=model,
                transform=transform,
                image_paths=image_paths,
                layer_names=layer_chunk,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                n_projections=n_projections,
                srp_seed=srp_seed,
            )

            for layer_name in layer_chunk:
                if layer_name not in reduced:
                    continue
                layer_idx = selected_layers.index(layer_name)
                t_layer = time.time()
                feats = reduced[layer_name]
                rows = _score_layer(
                    feats,
                    model_name=option_key,
                    layer_name=layer_name,
                    layer_idx=layer_idx,
                    contexts=contexts,
                    alphas=alphas,
                    do_crsa=do_crsa,
                    do_wrsa=do_wrsa,
                    do_srpr=do_srpr,
                )
                all_rows.extend(rows)
                layer_meta.append(
                    {
                        "model": option_key,
                        "model_layer": layer_name,
                        "model_layer_index": layer_idx,
                        "n_discovered_layers": len(all_layers),
                        "n_selected_layers": len(selected_layers),
                        "n_features_srp": int(feats.shape[1]),
                        "n_images": int(feats.shape[0]),
                        "n_rows": len(rows),
                        "seconds": time.time() - t_layer,
                    }
                )
                del feats
                gc.collect()

            del reduced
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            _log(f"  Chunk done in {time.time() - t_chunk:.1f}s")

        _log(f"  {option_key}: {len(all_rows)} rows in {time.time() - t0:.1f}s")
        return all_rows, pd.DataFrame(layer_meta)

    finally:
        model = model.cpu()
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=None,
                        help="CSV with option_key column. Use --model-list for "
                             "an eval manifest of h5 basenames.")
    parser.add_argument("--model-list", type=Path, default=None,
                        help="Newline-delimited option keys or .h5 basenames.")
    parser.add_argument("--pool-csv", type=Path, required=True,
                        help="Stimulus pool CSV with image_id and image_path columns.")
    parser.add_argument("--image-root-map", action="append", default=[],
                        help="Rewrite image path prefixes from the pool CSV. "
                             "Format: FROM=TO. May be passed more than once.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+",
                        default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"])
    parser.add_argument("--voxel-set", default="otc", choices=("hlvis", "visual", "otc"))
    parser.add_argument("--rois", nargs="+", default=None)
    parser.add_argument("--min-roi-voxels", type=int, default=10)
    parser.add_argument("--ncsnr-threshold", type=float, default=0.2)
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    parser.add_argument("--no-crsa", action="store_true")
    parser.add_argument("--no-wrsa", action="store_true")
    parser.add_argument("--no-srpr", action="store_true")
    parser.add_argument("--include-ood", action="store_true",
                        help="Keep shared_4rep_OOD_* images in the split-half order.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--layer-chunk-size", type=int, default=8,
                        help="Number of second-half layers to extract per image pass.")
    parser.add_argument("--n-projections", type=int, default=5960)
    parser.add_argument("--srp-seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-batch", type=int, default=None)
    parser.add_argument("--models-per-batch", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    _log(f"Device: {device}")

    _log("Loading DeepNSD model registry...")
    model_options = get_model_options()
    model_keys = _resolve_model_items(args.models, args.model_list, model_options)
    missing = [m for m in model_keys if m not in model_options]
    if missing:
        raise SystemExit(f"{len(missing)} model keys not in DeepNSD registry: {missing[:5]}")

    if args.model_batch is not None:
        start = args.model_batch * args.models_per_batch
        model_keys = model_keys[start : start + args.models_per_batch]
        _log(f"Model batch {args.model_batch}: {len(model_keys)} models")
    if not model_keys:
        raise SystemExit("No models selected")

    out_stem = (
        f"results_streamed_batch{args.model_batch}"
        if args.model_batch is not None
        else "results_streamed_all"
    )
    out_path = args.out / f"{out_stem}.parquet"
    meta_path = args.out / f"{out_stem}_layer_metadata.csv"
    if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        _log(f"Skipping existing output: {out_path}")
        return 0

    if args.dry_run:
        _log("=== DRY RUN ===")
        _log(f"subjects={args.subjects}")
        _log(f"models={model_keys}")
        _log(f"out={out_path}")
        return 0

    contexts = _load_contexts(
        args.subjects,
        voxel_set=args.voxel_set,
        rois=args.rois,
        min_roi_voxels=args.min_roi_voxels,
        ncsnr_threshold=args.ncsnr_threshold,
        include_ood=args.include_ood,
    )
    if not contexts:
        raise SystemExit("No brain contexts loaded")
    image_ids = contexts[0].bench_ids
    image_root_maps = _parse_image_root_maps(args.image_root_map)
    image_paths = _image_paths_from_pool(args.pool_csv, image_ids, image_root_maps)
    _log(f"Resolved {len(image_paths)} split-half images from {args.pool_csv}")

    all_rows: List[dict] = []
    all_meta: List[pd.DataFrame] = []
    failures: List[dict] = []
    for option_key in model_keys:
        try:
            rows, meta = evaluate_model_streamed(
                option_key=option_key,
                model_options=model_options,
                contexts=contexts,
                image_paths=image_paths,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                layer_chunk_size=args.layer_chunk_size,
                device=device,
                n_projections=args.n_projections,
                srp_seed=args.srp_seed,
                alphas=args.alphas,
                do_crsa=not args.no_crsa,
                do_wrsa=not args.no_wrsa,
                do_srpr=not args.no_srpr,
            )
            all_rows.extend(rows)
            all_meta.append(meta)
        except Exception as exc:
            _log(f"ERROR on {option_key}: {exc}")
            traceback.print_exc()
            failures.append({"model": option_key, "error": str(exc)[:500]})

    df = pd.DataFrame(all_rows)
    df.to_parquet(out_path, index=False)
    _log(f"wrote {len(df)} rows -> {out_path}")

    if all_meta:
        meta_df = pd.concat(all_meta, ignore_index=True)
        meta_df.to_csv(meta_path, index=False)
        _log(f"wrote layer metadata -> {meta_path}")
    if failures:
        fail_path = args.out / f"{out_stem}_failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        _log(f"wrote failures -> {fail_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
