"""Shared helpers for the RSA evaluators.

- :func:`load_features` reads an ``image_id``-indexed h5 (the format produced
  by ``conwell_replication.extract.extract_features``) and returns
  ``{layer_name: ndarray, ...}`` plus the row-aligned image_ids.

- :func:`align_features` reorders a layer's feature rows to match a target
  list of image_ids (e.g. the benchmark's stimulus order, or a split's
  train/test partition).

- :func:`fit_ridge` runs the ``RidgeCVMod`` from the vendored DeepNSD code
  and returns train/test predictions and per-voxel best alphas.

The vendored ``compare_rdms`` (Spearman by default) and ``score_func``
(Pearson r) are re-exported here for convenience.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

# Vendored DeepNSD imports (importing _vendor primes sys.path for model_opts.*)
from conwell_replication import _vendor  # noqa: F401  — side effect: model_opts importable
from conwell_replication._vendor.deepnsd.model_opts.mapping_methods import (  # noqa: F401
    compare_rdms, score_func,
)
from conwell_replication._vendor.deepnsd.ridge_gcv_mod import RidgeCVMod


def load_features(h5_path: Path) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Load all layer SRP features + the row-aligned image_ids."""
    features: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        if "features_srp" not in f:
            raise ValueError(f"{h5_path} has no /features_srp group")

        # image_ids dataset is required for image_id-indexed h5; fall back to
        # positional ordering for legacy files (with a warning).
        if "image_ids" in f:
            raw_ids = f["image_ids"][:]
            image_ids = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw_ids
            ]
        else:
            image_ids = []
            print(
                f"WARNING: {h5_path} has no /image_ids dataset; "
                "downstream id-based alignment will fail."
            )

        grp = f["features_srp"]
        for layer in grp.keys():
            features[layer] = grp[layer][:]

    return features, image_ids


def align_features(
    features: np.ndarray,
    feature_image_ids: List[str],
    target_image_ids: List[str],
) -> np.ndarray:
    """Reorder ``features`` rows to match ``target_image_ids``."""
    id_to_idx = {iid: i for i, iid in enumerate(feature_image_ids)}
    missing = [iid for iid in target_image_ids if iid not in id_to_idx]
    if missing:
        raise KeyError(
            f"{len(missing)} target image_ids missing from features "
            f"(e.g. {missing[:3]}). Re-extract features over the relevant pool."
        )
    rows = np.fromiter((id_to_idx[i] for i in target_image_ids), dtype=np.int64,
                       count=len(target_image_ids))
    return features[rows, :]


def fit_ridge(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    alphas: List[float],
):
    """Standardize, fit RidgeCVMod, and return (pred_train, pred_test, ridge).

    ``pred_train`` is the cross-validated prediction (per-voxel best alpha,
    no leakage) computed from RidgeCVMod's stored CV values, mirroring the
    convention of run_rsa_eval.py.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ridge = RidgeCVMod(alphas=alphas, alpha_per_target=True, scoring="pearson_r")
    ridge.store_cv_values = True
    ridge.fit(X_train_s, y_train)

    best_alpha_idx = np.array([alphas.index(a) for a in ridge.alpha_])
    cv_values = getattr(ridge, "cv_values_", getattr(ridge, "cv_results_", None))
    if cv_values is None:
        raise RuntimeError("RidgeCVMod produced no cv_values_/cv_results_")
    pred_train = np.take_along_axis(
        cv_values, best_alpha_idx[None, :, None], axis=2
    )[:, :, 0]
    pred_test = X_test_s @ ridge.coef_.T + ridge.intercept_
    return pred_train, pred_test, ridge


def rdm_from_responses(Y: np.ndarray) -> np.ndarray:
    """Build a representational dissimilarity matrix (1 - Pearson r).

    ``Y`` should be (n_stimuli, n_features) — i.e. rows = patterns to be
    compared. Returns an (n_stimuli, n_stimuli) RDM.
    """
    return 1.0 - np.corrcoef(Y)
