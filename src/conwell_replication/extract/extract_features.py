#!/usr/bin/env python3
"""Extract SRP-reduced features using Conwell's DeepNSD protocol.

This is a faithful port of the
``rsa_large_scale_benchmark/extract_features_deepnsd.py`` extractor:

  * Loads each model via DeepNSD's call-string registry
    (:func:`model_opts.model_options.get_model_options`).
  * Resolves preprocessing through
    :func:`model_opts.model_options.get_recommended_transforms` with
    ``input_type="PIL"``.
  * Captures activations via DeepNSD's hook-based
    :func:`model_opts.feature_extraction.get_all_feature_maps`. Layer naming
    follows the ``ModuleType-N`` convention reported in Conwell et al. (2024).
  * Selects ~``--target-n-layers`` layers from the second half of the network
    (skipping uninformative module types like ``Dropout``, ``Identity``,
    ``Flatten``, ``Sequential``, ``ModuleList``).
  * Reduces with sparse random projection (``eps=0.1``,
    :func:`model_opts.feature_reduction.get_feature_map_srps`).

Output: one ``.h5`` per model with the same schema as the prior pipeline,
plus an ``image_ids`` dataset so per-subject indexing is image_id-keyed
(not positional)::

    /attrs/model_name             (str)
    /attrs/layer_names            (list[str])
    /attrs/n_layers               (int)
    /attrs/n_images               (int)
    /attrs/extraction_time        (ISO-8601 string)
    /attrs/extractor              "deepnsd_hooks"
    /image_ids                    utf-8 string dataset, one per row
    /features_srp/<safe_layer>    (n_images, n_proj) float32

The model list (``--models``) must be a CSV with at minimum an
``option_key`` column matching keys in DeepNSD's registry. The default
list shipped at ``resources/conwell_model_list.csv`` is the trained-only
subset (335 models) of DeepNSD's 489-model registry, which is what the
published paper reports against modulo Conwell's post-publication
extensions to the timm catalog.
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Importing _vendor primes sys.path so model_opts.* is importable.
from conwell_replication import _vendor  # noqa: F401
from model_opts.feature_extraction import (
    get_empty_feature_maps,
    get_feature_maps,
)
from model_opts.model_options import (
    get_model_options,
    get_recommended_transforms,
)
from sklearn.random_projection import SparseRandomProjection


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Layer selection — same heuristics as extract_features_deepnsd.py
# ---------------------------------------------------------------------------
_SKIP_TYPES = {"Dropout", "Identity", "Flatten", "Sequential", "ModuleList"}


def select_layers(feature_map_names: List[str], target: int = 10) -> List[str]:
    filtered = [n for n in feature_map_names if not any(n.startswith(s) for s in _SKIP_TYPES)]
    if len(filtered) < 3:
        filtered = list(feature_map_names)

    half = len(filtered) // 2
    second_half = filtered[half:]
    if len(second_half) <= target:
        result = list(second_half)
    else:
        stride = max(1, len(second_half) // target)
        result = list(second_half[::stride])

    if filtered and filtered[-1] not in result:
        result.append(filtered[-1])
    return result


# ---------------------------------------------------------------------------
# Model loading via DeepNSD's call-string registry
# ---------------------------------------------------------------------------
def load_model(option_key: str, model_options: dict) -> torch.nn.Module:
    info = model_options[option_key]
    call_str = info["call"]

    # Build the namespace mirroring extract_features_deepnsd.py.
    from model_opts import model_options as mo

    ns = {
        "torch": torch,
        "get_torchvision_model": mo.get_torchvision_model,
        "get_timm_model": mo.get_timm_model,
        "get_dino_model": mo.get_dino_model,
        "get_clip_model": mo.get_clip_model,
        "get_vissl_model": mo.get_vissl_model,
        "get_taskonomy_encoder": mo.get_taskonomy_encoder,
        "get_detectron_model": mo.get_detectron_model,
        "random_taskonomy_encoder": mo.random_taskonomy_encoder,
    }

    source = info.get("model_source", "")
    if source in ("slip", "seer", "ipcl", "vicreg", "openclip", "cornet",
                   "dinov2", "robustness", "vqgan", "bit_expert"):
        from model_opts.model_code._options import get_custom_model_options
        custom_opts = get_custom_model_options()
        if option_key in custom_opts:
            call_str = custom_opts[option_key]["call"]
            from model_opts.model_code import _options as co
            from model_opts.model_code import custom_extended_models as cem
            for mod in (co, cem):
                for attr in dir(mod):
                    if callable(getattr(mod, attr)):
                        ns[attr] = getattr(mod, attr)

    return eval(call_str, ns).eval()


# ---------------------------------------------------------------------------
# Hook-based extraction + SRP for one model
# ---------------------------------------------------------------------------
def extract_and_reduce_to_h5(
    out_path: Path,
    option_key: str,
    model: torch.nn.Module,
    transform,
    image_paths: List[str],
    image_ids: List[str],
    batch_size: int,
    target_n_layers: int,
    device: str,
    n_projections: int = 5960,
    srp_seed: int = 0,
    num_workers: int = 4,
):
    """Stream feature extraction → SRP → HDF5 dataset slice writes.

    Memory bound is ``batch_size * max(n_features_per_layer)`` plus a small
    constant per layer (the fitted ``SparseRandomProjection`` matrix and the
    open HDF5 dataset handle). Reduced features are written directly into
    ``features_srp/<safe_layer>[start:start+batch]`` and never accumulated
    in CPU RAM as a list.

    SRP target dim defaults to **5960**, matching Conwell et al. (2024)'s
    fixed projection (Johnson-Lindenstrauss bound for n=1000 NSD probe
    images, eps=0.1). The projector for each layer is instantiated and fit
    on the first batch that produces a non-None output for that layer; with
    explicit ``n_components`` and fixed ``random_state`` the projection
    matrix is deterministic and depends only on ``n_features`` and the
    seed, so per-batch fits are equivalent to a single full-dataset fit.

    Image loading is lazy: paths are opened + transformed on demand inside
    DataLoader workers, so the ~19 GB cost of pre-decoding 24,681 RGB
    pixel buffers is never paid.
    """
    n_samples = len(image_paths)
    if n_samples == 0:
        raise ValueError("image_paths is empty")
    if len(image_ids) != n_samples:
        raise ValueError(
            f"image_ids length ({len(image_ids)}) != image_paths "
            f"length ({n_samples})"
        )

    class _LazyImagesDataset(torch.utils.data.Dataset):
        def __init__(self, paths, tfm):
            self.paths = paths
            self.tfm = tfm
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            with Image.open(self.paths[idx]) as img:
                return self.tfm(img.convert("RGB"))

    dataloader = torch.utils.data.DataLoader(
        _LazyImagesDataset(image_paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    model = model.to(device)

    # 1. Discover layer names with a synthetic 1-image forward pass.
    with Image.open(image_paths[0]) as _img0:
        sample = transform(_img0.convert("RGB"))
    input_shape = tuple(sample.shape)
    del sample
    _log("  Discovering layer names...")
    all_layer_names = get_empty_feature_maps(
        model, inputs=None, input_shape=input_shape,
        remove_duplicates=True, names_only=True,
    )
    _log(f"  Total layers: {len(all_layer_names)}")

    selected = select_layers(all_layer_names, target=target_n_layers)
    _log(f"  Selected {len(selected)} of {len(all_layer_names)} layers:")
    for ln in selected:
        _log(f"    - {ln}")
    selected_set = set(selected)

    _log(
        f"  SRP target dim: {n_projections} "
        f"(Conwell-fixed, JL bound for n=1000 probe at eps=0.1)"
    )

    # 2. Open output HDF5 and stream batch-by-batch into datasets.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")

    srp_per_layer: Dict[str, SparseRandomProjection] = {}
    h5_dset_per_layer: Dict[str, h5py.Dataset] = {}
    layer_order: List[str] = []
    write_idx = 0

    h5_chunk_rows = min(batch_size, n_samples)

    try:
        with h5py.File(tmp_path, "w") as f:
            f.attrs["model_name"] = option_key
            f.attrs["n_images"] = n_samples
            f.attrs["extraction_time"] = datetime.now().isoformat()
            f.attrs["extractor"] = "deepnsd_hooks"
            f.attrs["srp_n_projections"] = n_projections
            f.attrs["srp_seed"] = srp_seed

            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset(
                "image_ids",
                data=np.array(image_ids, dtype=object),
                dtype=dt,
            )
            grp = f.create_group("features_srp")

            _log("  Extracting features (streaming SRP → HDF5)...")
            pbar = tqdm(dataloader, desc="Feature Extraction (Batch)")
            for batch in pbar:
                batch = batch.to(device, non_blocking=True)
                with torch.no_grad():
                    batch_maps = get_feature_maps(
                        model, batch,
                        layers_to_retain=selected,
                        remove_duplicates=False,
                        enforce_input_shape=True,
                    )
                del batch

                batch_n = None
                for layer_name in selected:
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
                        safe = layer_name.replace("/", "_").replace(".", "_")
                        dset = grp.create_dataset(
                            safe,
                            shape=(n_samples, n_projections),
                            dtype="float32",
                            chunks=(h5_chunk_rows, n_projections),
                            compression="gzip",
                            compression_opts=4,
                        )
                        dset.attrs["layer_name"] = layer_name
                        h5_dset_per_layer[layer_name] = dset
                        layer_order.append(layer_name)

                    reduced = srp_per_layer[layer_name].transform(feats_np)
                    h5_dset_per_layer[layer_name][
                        write_idx:write_idx + batch_n, :
                    ] = np.asarray(reduced, dtype=np.float32)

                del batch_maps
                if batch_n is not None:
                    write_idx += batch_n
            pbar.close()

            f.attrs["n_layers"] = len(layer_order)
            f.attrs["layer_names"] = layer_order

        if write_idx != n_samples:
            _log(
                f"  WARNING: wrote {write_idx} rows but expected {n_samples}"
            )
        tmp_path.replace(out_path)

    finally:
        # Drop GPU model + caches regardless of success
        model = model.cpu()
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        # Clean up any partial file from a crashed run
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    _log(f"  wrote {len(layer_order)} layer datasets, n_proj={n_projections}")


# ---------------------------------------------------------------------------
# Image loading from union-pool CSV
# ---------------------------------------------------------------------------
def load_image_list(pool_csv: Path, images_root: Optional[Path]) -> tuple:
    """Return (image_ids, image_paths). Images are NOT pre-decoded — opening
    happens lazily inside the per-task DataLoader so we don't carry ~19 GB
    of decoded RGB pixel buffers in CPU memory for the union pool.
    """
    df = pd.read_csv(pool_csv).sort_values("union_index", kind="stable").reset_index(drop=True)
    if "image_path" in df.columns:
        paths = df["image_path"].tolist()
    elif images_root is not None:
        paths = [str(Path(images_root) / iid) for iid in df["image_id"]]
    else:
        raise SystemExit(
            "Pool CSV has no 'image_path' column and --images-root was not "
            "provided. Either add image_path to the pool CSV or pass "
            "--images-root /path/to/images."
        )
    _log(f"Resolved {len(paths)} image paths from {pool_csv} (lazy loading)")
    return df["image_id"].tolist(), paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path,
                    default=Path(__file__).resolve().parents[3] / "resources" / "conwell_model_list.csv",
                    help="CSV with an 'option_key' column listing DeepNSD models. "
                         "Defaults to the packaged 335-model trained-only list.")
    ap.add_argument("--pool", type=Path, required=True,
                    help="Stimulus pool CSV from "
                         "`conwell_replication.data.stimuli build-pool`.")
    ap.add_argument("--images-root", type=Path, default=None,
                    help="Directory containing the stimulus images. If omitted, "
                         "expects an 'image_path' column in the pool CSV.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output features directory (one .h5 per model).")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--sources", nargs="+", default=None,
                    help="Restrict to specific model_source values (e.g. timm torchvision).")
    ap.add_argument("--target-n-layers", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader workers for parallel JPEG decode + transform.")
    ap.add_argument("--n-projections", type=int, default=5960,
                    help="SRP target dimensions. Default 5960 matches Conwell "
                         "et al. (2024) — JL bound for n=1000 NSD probe images "
                         "at eps=0.1, then frozen across all datasets.")
    ap.add_argument("--srp-seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="Re-extract even if output already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be extracted without running.")
    return ap.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _log(f"Device: {device}")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    model_list = pd.read_csv(args.models)
    if "option_key" not in model_list.columns:
        raise SystemExit(f"--models CSV must have an 'option_key' column. Got: {list(model_list.columns)}")

    if args.sources:
        if "model_source" not in model_list.columns:
            raise SystemExit("--sources filter requires a 'model_source' column in the model list CSV.")
        model_list = model_list[model_list["model_source"].isin(args.sources)].reset_index(drop=True)
        _log(f"Filtered to sources {args.sources}: {len(model_list)} models")

    if args.end is not None:
        model_list = model_list.iloc[args.start:args.end]
    else:
        model_list = model_list.iloc[args.start:]
    _log(f"Processing {len(model_list)} models")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _log("\n=== DRY RUN ===")
        for _, row in model_list.iterrows():
            key = row["option_key"]
            h5 = args.out / (key.replace("/", "-") + ".h5")
            status = "EXISTS" if h5.exists() else "NEW"
            _log(f"  [{status}] {key} ({row.get('model_source','?')})")
        new_count = sum(
            1 for _, row in model_list.iterrows()
            if not (args.out / (row["option_key"].replace("/", "-") + ".h5")).exists()
        )
        _log(f"Would extract {new_count} new models")
        return 0

    _log("Loading DeepNSD model registry...")
    model_options = get_model_options()
    _log(f"  registry size: {len(model_options)}")

    image_ids, image_paths = load_image_list(args.pool, args.images_root)

    successes = 0
    skips = 0
    failures: List[tuple] = []

    for idx, (_, row) in enumerate(model_list.iterrows()):
        option_key = str(row["option_key"])
        source = row.get("model_source", "?")
        h5_name = option_key.replace("/", "-") + ".h5"
        out_path = args.out / h5_name

        if out_path.exists() and not args.force:
            _log(f"\n[{idx+1}/{len(model_list)}] SKIP {option_key} (exists)")
            skips += 1
            continue

        _log(f"\n{'='*60}\n[{idx+1}/{len(model_list)}] {option_key} ({source})\n{'='*60}")

        if option_key not in model_options:
            _log(f"  WARNING: {option_key} not in registry; skipping")
            failures.append((option_key, "not in registry"))
            continue

        try:
            _log("  Loading model...")
            model = load_model(option_key, model_options)

            transform = get_recommended_transforms(option_key, input_type="PIL")

            extract_and_reduce_to_h5(
                out_path=out_path,
                option_key=option_key,
                model=model,
                transform=transform,
                image_paths=image_paths,
                image_ids=image_ids,
                batch_size=args.batch_size,
                target_n_layers=args.target_n_layers,
                device=device,
                n_projections=args.n_projections,
                srp_seed=args.srp_seed,
                num_workers=args.num_workers,
            )
            _log(f"  SAVED {h5_name}")
            successes += 1

        except Exception as e:
            _log(f"  ERROR: {e}")
            traceback.print_exc()
            failures.append((option_key, str(e)[:100]))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _log(f"\n{'='*60}\nSummary: {successes} ok / {skips} skipped / {len(failures)} failed")
    if failures:
        fail_path = args.out / "extraction_failures.csv"
        pd.DataFrame(failures, columns=["model", "error"]).to_csv(fail_path, index=False)
        _log(f"  failures → {fail_path}")
        for name, reason in failures[:20]:
            _log(f"    - {name}: {reason}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
