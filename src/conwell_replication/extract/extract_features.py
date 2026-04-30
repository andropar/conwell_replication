#!/usr/bin/env python3
"""Extract SRP-reduced features from the 152 curated models on the union pool.

The pool is the union of every participant's min_nn pool — see
``conwell_replication.data.stimuli.build_pool_csv``. Each output ``.h5`` file
contains the SRP features for one model:

    /attrs/model_name             (str)
    /attrs/layer_names            (list[str], in extraction order)
    /attrs/n_layers               (int)
    /attrs/n_images               (int)
    /attrs/extraction_time        (ISO-8601 string)
    /image_ids                    (uint8 array of utf8-encoded ids; one per row)
    /features_srp/<safe_layer>    (n_images, n_proj) float32

The image_id row index is shared across all layers within a file, so per-
participant subsets can be obtained by gathering rows with the matching
``image_id`` values. The file produced here is the same format as the
existing rsa_large_scale_benchmark, except we always carry an ``image_ids``
dataset so that downstream evaluators can index by image_id rather than by
positional order (which differs between the union pool and any single
participant's pool).
"""

from __future__ import annotations

import argparse
import gc
import re
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
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from tqdm import tqdm

# Local vendored helpers
from conwell_replication._vendor.feature_reduction import get_feature_map_srps
from conwell_replication._vendor.universal_extractor import get_custom_model

try:
    from deepjuice import get_deepjuice_model
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "deepjuice is required for feature extraction but is not installed. "
        "Install Conwell's DeepJuice package on this machine before running. "
        f"(import error: {exc})"
    )


# ---------------------------------------------------------------------------
# Layer selection — same heuristics as the existing extractor
# ---------------------------------------------------------------------------
_GOOD_PATTERNS = (
    ".add", "norm2", "norm1", "ln_2", "ln_1", "bn3", "bn2",
    "fc_norm", "ln_post", "avgpool", "pool", ".fc", "mlp.fc2",
)
_SKIP_PATTERNS = (
    "embed", "patch", "cls_token", "head", "classifier",
    "qkv", "proj", "attn", "weight", "bias", "mask",
    "getattr", "getitem", "view", "reshape", "permute",
    "gamma", "beta", "scale", "eq", "mul", "conv", "relu", "gelu",
    "contiguous", "clone", "expand", "repeat", "squeeze",
    "unsqueeze", "chunk", "cat", "split", "transpose", "truediv",
)


def filter_layers(layer_names: List[str]) -> List[str]:
    out = []
    for layer in layer_names:
        ll = layer.lower()
        if any(s in ll for s in _SKIP_PATTERNS):
            continue
        if any(g in ll for g in _GOOD_PATTERNS):
            out.append(layer)
    if len(out) < 5:
        for layer in layer_names:
            if re.search(r"\.\d+$", layer) and layer not in out:
                out.append(layer)
    return out


def select_layers(all_layers: List[str], target: int = 10) -> List[str]:
    filtered = filter_layers(all_layers)
    if len(filtered) < 3:
        filtered = list(all_layers)
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
# Activation aggregation
# ---------------------------------------------------------------------------
def aggregate_activation(t: torch.Tensor) -> torch.Tensor:
    """Collapse a layer's activation to (batch, features)."""
    if t.ndim == 2:
        return t
    if t.ndim == 3:
        return t[:, 0]                       # ViT: take CLS token
    if t.ndim == 4:
        return t.mean(dim=[2, 3])             # CNN: global average pool
    return t.reshape(t.shape[0], -1)


# ---------------------------------------------------------------------------
# Per-model extraction
# ---------------------------------------------------------------------------
def load_model(model_name: str):
    """Custom-loader fallback for the few non-deepjuice models."""
    custom_models = {
        "cornet_s",
        "dinov3-vitl16-pretrain-lvd1689m",
        "vissl_resnet50_barlowtwins",
    }
    if model_name in custom_models:
        return get_custom_model(model_name)
    return get_deepjuice_model(model_name)


def extract_multilayer_features(
    model,
    preprocess,
    layer_names: List[str],
    images: List[Image.Image],
    batch_size: int,
    device: str,
) -> Dict[str, np.ndarray]:
    """One forward pass per batch captures all requested layers."""
    return_nodes = {ln: ln for ln in layer_names}
    try:
        fe = create_feature_extractor(model, return_nodes=return_nodes).to(device).eval()
    except Exception as e:
        print(f"    FX extraction failed: {e}")
        return {}

    layer_features: Dict[str, list] = {ln: [] for ln in layer_names}
    for i in tqdm(range(0, len(images), batch_size), desc="    Batches", leave=False):
        batch = images[i:i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.inference_mode():
            outputs = fe(tensors)
        for ln in layer_names:
            if ln in outputs:
                agg = aggregate_activation(outputs[ln]).cpu().numpy()
                layer_features[ln].append(agg)
        del tensors, outputs

    result: Dict[str, np.ndarray] = {}
    for ln in layer_names:
        if layer_features[ln]:
            arr = np.concatenate(layer_features[ln], axis=0)
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            result[ln] = arr.astype(np.float32, copy=False)
    return result


def apply_srp(
    features: Dict[str, np.ndarray],
    eps: float = 0.1,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, fmap in tqdm(features.items(), desc="    SRP", leave=False):
        try:
            srp = get_feature_map_srps(
                {"layer": fmap},
                n_projections=None,
                upsampling=True,
                eps=eps,
                seed=seed,
                save_outputs=False,
            )["layer"]
            out[name] = srp.astype(np.float32, copy=False)
        except Exception as e:
            print(f"    SRP failed for {name}: {e}")
            continue
    return out


def write_h5(
    output_path: Path,
    model_name: str,
    image_ids: List[str],
    srp: Dict[str, np.ndarray],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["model_name"] = model_name
        f.attrs["n_layers"] = len(srp)
        f.attrs["layer_names"] = list(srp.keys())
        f.attrs["n_images"] = len(image_ids)
        f.attrs["extraction_time"] = datetime.now().isoformat()
        # h5py wants utf-8 string dtype for variable-length strings
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("image_ids", data=np.array(image_ids, dtype=object), dtype=dt)
        grp = f.create_group("features_srp")
        for name, data in srp.items():
            safe = name.replace("/", "_").replace(".", "_")
            grp.create_dataset(safe, data=data, compression="gzip", compression_opts=4)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, required=True,
                    help="CSV with a 'model_uid' column listing models to extract.")
    ap.add_argument("--pool", type=Path, required=True,
                    help="Stimulus pool CSV from "
                         "`conwell_replication.data.stimuli build-pool`.")
    ap.add_argument("--images-root", type=Path, default=None,
                    help="Directory containing the stimulus images. If None, "
                         "expects an 'image_path' column in the pool CSV.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output features directory (one .h5 per model).")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--target-n-layers", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--srp-eps", type=float, default=0.1)
    ap.add_argument("--srp-seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="Re-extract even if output already exists.")
    return ap.parse_args(argv)


def load_image_list(pool_csv: Path, images_root: Optional[Path]) -> tuple:
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
    print(f"Loading {len(paths)} images from {pool_csv}...")
    images = [Image.open(p).convert("RGB") for p in tqdm(paths, desc="open images")]
    return df["image_id"].tolist(), images


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device.startswith("cuda"):
        torch.cuda.set_device(args.gpu)

    models_df = pd.read_csv(args.models)
    if "model_uid" not in models_df.columns:
        raise SystemExit(f"--models CSV must have a 'model_uid' column. Got: {list(models_df.columns)}")
    if args.end:
        models_df = models_df.iloc[args.start:args.end]
    else:
        models_df = models_df.iloc[args.start:]
    print(f"Processing {len(models_df)} models (rows {args.start} ..)")

    image_ids, images = load_image_list(args.pool, args.images_root)
    args.out.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures: List[str] = []

    for _, row in models_df.iterrows():
        model_name = str(row["model_uid"])
        out_path = args.out / f"{model_name.replace('/', '-')}.h5"
        if out_path.exists() and not args.force:
            print(f"\n[SKIP] {model_name} — exists at {out_path}")
            continue

        print(f"\n{'='*60}\n{model_name}\n{'='*60}")
        try:
            print("  Loading model...")
            model, preprocess = load_model(model_name)
            if hasattr(model, "module"):
                model = model.module
            model = model.to(device).eval()

            print("  Tracing layers...")
            try:
                _, eval_nodes = get_graph_node_names(model)
                all_layers = list(eval_nodes)
            except Exception as e:
                print(f"  ERROR: cannot trace model: {e}")
                failures.append(model_name)
                continue

            layers = select_layers(all_layers, target=args.target_n_layers)
            print(f"  Selected {len(layers)} of {len(all_layers)} layers")
            for ln in layers:
                print(f"    - {ln}")

            print("  Extracting features (multi-layer)...")
            features = extract_multilayer_features(
                model, preprocess, layers, images, args.batch_size, device
            )
            del model
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if not features:
                print("  ERROR: no features extracted")
                failures.append(model_name)
                continue

            print("  Applying SRP...")
            srp = apply_srp(features, eps=args.srp_eps, seed=args.srp_seed)
            del features
            gc.collect()
            if not srp:
                print("  ERROR: SRP failed for all layers")
                failures.append(model_name)
                continue

            print(f"  Writing {len(srp)} layers to {out_path}")
            write_h5(out_path, model_name, image_ids, srp)
            successes += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failures.append(model_name)

        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"\n{'='*60}\nSummary: {successes} ok / {len(failures)} failed")
    if failures:
        print("Failed:")
        for m in failures:
            print(f"  - {m}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
