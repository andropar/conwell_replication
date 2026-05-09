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
from warnings import warn

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

# Vendored DeepNSD imports (importing _vendor primes sys.path for model_opts.*)
from conwell_replication import _vendor  # noqa: F401  — side effect: model_opts importable
from conwell_replication._vendor.deepnsd.ridge_gcv_mod import RidgeCVMod


pearsonr_vec = np.vectorize(pearsonr, signature="(n),(n)->(),()")


def _pearson_r_score(y_true, y_pred, multioutput=None):
    y_true_ = y_true.transpose()
    y_pred_ = y_pred.transpose()
    return pearsonr_vec(y_true_, y_pred_)[0]


def _pearson_r2_score(y_true, y_pred, multioutput=None):
    return _pearson_r_score(y_true, y_pred) ** 2


def _get_predicted_values(y_true, y_pred, transform=None, multioutput=None):
    if transform is None:
        return y_pred
    raise ValueError(f"unknown transform: {transform}")


def score_func(y_true, y_pred, score_type="pearson_r"):
    """DeepNSD-compatible scoring helper, returning raw voxel values."""
    if isinstance(score_type, list):
        return {
            score_type_i: score_func(y_true, y_pred, score_type_i)
            for score_type_i in score_type
        }
    if score_type == "pearson_r":
        return _pearson_r_score(y_true, y_pred, multioutput="raw_values")
    if score_type == "pearson_r2":
        return _pearson_r2_score(y_true, y_pred, multioutput="raw_values")
    if score_type == "predicted_values":
        return _get_predicted_values(y_true, y_pred, multioutput="raw_values")
    if score_type == "r2":
        from sklearn.metrics import r2_score

        return r2_score(y_true, y_pred, multioutput="raw_values")
    if score_type == "explained_variance":
        from sklearn.metrics import explained_variance_score

        return explained_variance_score(y_true, y_pred, multioutput="raw_values")
    raise KeyError(score_type)


def compare_rdms(rdm1, rdm2, dist_type="pearson"):
    """DeepNSD-compatible RDM comparison over the upper triangle."""
    rdm1_triu = rdm1[np.triu_indices(rdm1.shape[0], k=1)]
    rdm2_triu = rdm2[np.triu_indices(rdm2.shape[0], k=1)]

    if np.sum(np.isnan(rdm1_triu)) > 0 or np.sum(np.isnan(rdm2_triu)) > 0:
        warn("compare_rdms: RDMs contain NaNs; returning NaN.", RuntimeWarning)
        return float("nan")

    if dist_type == "pearson":
        return pearsonr(rdm1_triu, rdm2_triu)[0]
    if dist_type == "spearman":
        return spearmanr(rdm1_triu, rdm2_triu)[0]
    raise ValueError(f"unknown dist_type: {dist_type}")


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


def take_h5_rows(dataset, rows: np.ndarray) -> np.ndarray:
    """Read selected HDF5 rows without materializing the full dataset.

    h5py requires fancy row indices to be sorted and unique. This helper keeps
    the caller's requested row order while reading only the chunks that contain
    the selected rows.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError(f"rows must be 1D, got shape {rows.shape}")
    if rows.size == 0:
        return np.empty((0,) + tuple(dataset.shape[1:]), dtype=dataset.dtype)
    if rows.min() < 0 or rows.max() >= dataset.shape[0]:
        raise IndexError(
            f"row indices outside dataset with {dataset.shape[0]} rows"
        )

    if rows.size == 1:
        return dataset[int(rows[0]) : int(rows[0]) + 1]
    if np.all(np.diff(rows) == 1):
        return dataset[int(rows[0]) : int(rows[-1]) + 1]

    unique_rows, inverse = np.unique(rows, return_inverse=True)
    chunk_shape = getattr(dataset, "chunks", None)
    row_chunk = chunk_shape[0] if chunk_shape else None
    if row_chunk:
        unique_shape = (unique_rows.size,) + tuple(dataset.shape[1:])
        unique_block = np.empty(unique_shape, dtype=dataset.dtype)
        chunk_ids = unique_rows // int(row_chunk)
        starts = np.r_[0, np.flatnonzero(np.diff(chunk_ids)) + 1]
        stops = np.r_[starts[1:], unique_rows.size]
        for start, stop in zip(starts, stops):
            group_rows = unique_rows[start:stop]
            lo = int(group_rows[0])
            hi = int(group_rows[-1]) + 1
            block = dataset[lo:hi]
            unique_block[start:stop] = block[group_rows - lo]
        return unique_block[inverse]

    block = dataset[unique_rows]
    return block[inverse]


def _fit_ridge_single_alpha(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
):
    """SVD-based ridge fit + LOO-CV train predictions for a single alpha.

    Replaces sklearn ``RidgeCVMod`` with ``store_cv_values=True`` for the
    Conwell-fixed-α=1e5 case. The sklearn path stores
    ``(n_train, n_targets, n_alphas)`` LOO predictions in float64 — for
    n_targets=270k voxels that's ~10 GB just for storage, plus another ~10
    GB for the float64 cast of y. With one alpha we don't need any of that.

    Math: RidgeCVMod preprocesses with ``fit_intercept=True``, so it centers
    X and y, then uses an unregularized intercept column for the GCV/LOO
    shortcut. Since centered y is orthogonal to that intercept column, the
    full fitted train values are still H y, where
    H = U diag(sᵢ²/(sᵢ²+α)) Uᵀ for centered X = U Σ Vᵀ. The LOO shortcut,
    however, must use the total leverage h_ii + 1/n from the feature smoother
    plus the intercept. RidgeCVMod stores these centered LOO predictions in
    ``cv_values_``; it does not add the intercept back to train predictions.
    Test predictions use coef = V (s_i / (s_i² + α)) Uᵀ y plus the fitted
    intercept.

    Memory profile for the usual LAION SRP case (n_train <= n_features,
    n_targets=270k, all float32 where safe):
      QTy = Q.T @ y:       (n_train × n_targets)       ~5 GB
      pred_train_loo:      (n_train × n_targets)       ~5 GB
      dual = Q @ (...):    (n_train × n_targets)       ~5 GB
      pred_test:           (n_test × n_targets)        ~1 GB
      Peak: tens of GB rather than the 100+ GB sklearn cv_values path.
    """
    # Match sklearn's _preprocess_data(fit_intercept=True): center X and y.
    # StandardScaler has already centered X, but doing this again keeps this
    # path equivalent to RidgeCVMod rather than relying on roundoff details.
    X_mean = X_train.mean(axis=0, dtype=np.float64)
    X_train_centered = X_train.astype(np.float64, copy=True)
    X_train_centered -= X_mean

    y_mean = y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_centered = (y_train - y_mean).astype(np.float32, copy=False)

    n_train, n_features = X_train_centered.shape
    if n_train <= n_features:
        X_train_centered32 = X_train_centered.astype(np.float32, copy=True)
        gram = X_train_centered @ X_train_centered.T
        eigvals, Q = np.linalg.eigh(gram)
        del gram, X_train_centered
        Q = Q.astype(np.float32, copy=False)

        D = (eigvals / (eigvals + alpha)).astype(np.float32)
        inv = (1.0 / (eigvals + alpha)).astype(np.float32)
        QTy = Q.T @ y_centered

        pred_train_loo = Q @ (D[:, None] * QTy)              # H y
        H_diag = (Q ** 2) @ D                                # feature leverage
        pred_train_loo -= (H_diag + np.float32(1.0 / n_train))[:, None] * y_centered
        H_diag += np.float32(1.0 / n_train)                  # intercept leverage
        pred_train_loo /= (1.0 - H_diag)[:, None]
        del y_centered

        dual = Q @ (inv[:, None] * QTy)
        del Q, QTy
        X_test_centered = X_test.astype(np.float32, copy=False) - X_mean.astype(np.float32)
        K_test = X_test_centered @ X_train_centered32.T
        del X_test_centered, X_train_centered32
        pred_test = K_test @ dual
        del K_test, dual
        pred_test += y_mean
        return pred_train_loo, pred_test, None

    # For n_train > n_features, use the design-matrix SVD path, matching
    # sklearn's dense GCV mode while avoiding cv_values_ storage.
    U, s, Vt = np.linalg.svd(X_train_centered, full_matrices=False)
    del X_train_centered
    U = U.astype(np.float32, copy=False)
    s2 = s * s
    D = (s2 / (s2 + alpha)).astype(np.float32)               # filter weights
    s_over = np.where(s > 0, s / (s2 + alpha), 0.0).astype(np.float32)

    UTy = U.T @ y_centered
    pred_train_loo = U @ (D[:, None] * UTy)                  # H y
    H_diag = (U ** 2) @ D                                    # feature leverage
    H_diag += np.float32(1.0 / X_train.shape[0])             # intercept leverage
    pred_train_loo -= H_diag[:, None] * y_centered
    pred_train_loo /= (1.0 - H_diag)[:, None]
    del y_centered, U

    coef = Vt.astype(np.float32, copy=False).T @ (s_over[:, None] * UTy)
    del UTy, Vt
    intercept = y_mean - (X_mean.astype(np.float32, copy=False) @ coef)
    pred_test = X_test.astype(np.float32, copy=False) @ coef + intercept

    return pred_train_loo, pred_test, None


def fit_ridge(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    alphas: List[float],
):
    """Standardize features, fit ridge with LOO-CV train predictions.

    Single-alpha path uses a memory-efficient SVD implementation
    (:func:`_fit_ridge_single_alpha`) that bypasses sklearn's
    ``RidgeCVMod.cv_values_`` storage. Multi-alpha falls back to
    ``RidgeCVMod`` with per-target alpha selection.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if len(alphas) == 1:
        return _fit_ridge_single_alpha(X_train_s, X_test_s, y_train, float(alphas[0]))

    ridge = RidgeCVMod(alphas=alphas, alpha_per_target=True, scoring="pearson_r")
    ridge.store_cv_values = True
    ridge.fit(X_train_s, y_train)

    best_alpha_idx = np.array([alphas.index(a) for a in ridge.alpha_])
    cv_values = getattr(ridge, "cv_values_", getattr(ridge, "cv_results_", None))
    if cv_values is None:
        raise RuntimeError("RidgeCVMod produced no cv_values_/cv_results_")
    if cv_values.ndim == 2:
        cv_values = cv_values.reshape(X_train_s.shape[0], y_train.shape[1], len(alphas))
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
    Y = np.asarray(Y, dtype=np.float32)
    means = Y.mean(axis=1, dtype=np.float64).astype(np.float32)
    Z = Y - means[:, None]
    norms = np.sqrt(np.einsum("ij,ij->i", Z, Z, dtype=np.float64)).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z /= norms[:, None]
    return 1.0 - (Z @ Z.T)


def mean_columnwise_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_size: int = 16384,
) -> float:
    """Mean Pearson r over columns without scipy's per-column Python loop."""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch for Pearson r: {y_true.shape} vs {y_pred.shape}"
        )

    n_samples, n_targets = y_true.shape
    total = 0.0
    for start in range(0, n_targets, block_size):
        stop = min(start + block_size, n_targets)
        yt = y_true[:, start:stop]
        yp = y_pred[:, start:stop]

        sum_t = yt.sum(axis=0, dtype=np.float64)
        sum_p = yp.sum(axis=0, dtype=np.float64)
        sum_tt = np.einsum("ij,ij->j", yt, yt, dtype=np.float64)
        sum_pp = np.einsum("ij,ij->j", yp, yp, dtype=np.float64)
        sum_tp = np.einsum("ij,ij->j", yt, yp, dtype=np.float64)

        num = n_samples * sum_tp - sum_t * sum_p
        den_t = n_samples * sum_tt - sum_t * sum_t
        den_p = n_samples * sum_pp - sum_p * sum_p
        den = np.sqrt(den_t * den_p)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = num / den
        total += r.sum(dtype=np.float64)

    return float(total / n_targets)


def columnwise_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_size: int = 16384,
) -> np.ndarray:
    """Pearson r for each target column, matching DeepNSD ``score_func``.

    The vendored DeepNSD SRPR path returns ``multioutput='raw_values'``:
    one Pearson r per voxel. This helper computes the same vector without
    scipy's per-voxel Python overhead.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch for Pearson r: {y_true.shape} vs {y_pred.shape}"
        )

    n_samples, n_targets = y_true.shape
    out = np.empty(n_targets, dtype=np.float32)
    for start in range(0, n_targets, block_size):
        stop = min(start + block_size, n_targets)
        yt = y_true[:, start:stop]
        yp = y_pred[:, start:stop]

        sum_t = yt.sum(axis=0, dtype=np.float64)
        sum_p = yp.sum(axis=0, dtype=np.float64)
        sum_tt = np.einsum("ij,ij->j", yt, yt, dtype=np.float64)
        sum_pp = np.einsum("ij,ij->j", yp, yp, dtype=np.float64)
        sum_tp = np.einsum("ij,ij->j", yt, yp, dtype=np.float64)

        num = n_samples * sum_tp - sum_t * sum_p
        den_t = n_samples * sum_tt - sum_t * sum_t
        den_p = n_samples * sum_pp - sum_p * sum_p
        den = np.sqrt(den_t * den_p)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start:stop] = (num / den).astype(np.float32)

    return out
