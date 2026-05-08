#!/usr/bin/env python3
"""Two diagnostics:

1. Within the NEW pipeline (sub-01), how correlated are RDMs computed on
   different OTC sub-ROIs vs the full OTC? Tells us how much of the 0.43
   cross-pipeline correlation could be explained by ROI choice alone.

2. For one representative model on sub-01, what is wRSA on:
     * 1121 non-OOD shared images (the current eval set), and
     * 1492 incl OOD shared images (the OLD eval set)?
   Tells us whether including OOD inflates wRSA.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import h5py


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _rdm(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float32)
    means = Y.mean(axis=1, dtype=np.float64).astype(np.float32)
    Z = Y - means[:, None]
    norms = np.sqrt(np.einsum("ij,ij->i", Z, Z, dtype=np.float64)).astype(np.float32)
    norms[norms == 0] = 1.0
    Z = Z / norms[:, None]
    return 1.0 - (Z @ Z.T)


def _upper(rdm: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(rdm.shape[0], k=1)
    return rdm[iu]


def _is_ood(iid: str) -> bool:
    return iid.startswith("shared_4rep_OOD_")


def aggregate_betas_for_images(betas: np.ndarray, image_ids: np.ndarray, target_ids: list) -> np.ndarray:
    """Average per-trial betas by image; return (n_target, n_voxels) float32."""
    rows_by_image = {iid: [] for iid in target_ids}
    target_set = set(target_ids)
    for i, iid in enumerate(image_ids):
        if iid in target_set:
            rows_by_image[iid].append(i)
    Y = np.empty((len(target_ids), betas.shape[1]), dtype=np.float32)
    for k, iid in enumerate(target_ids):
        rows = rows_by_image[iid]
        if not rows:
            raise KeyError(f"missing trials for {iid}")
        Y[k] = betas[rows].mean(axis=0)
    return Y


def diagnostic_1_cross_roi(
    cache_root: Path,
    subject: str,
    rois: list,
    common_ids: list,
) -> pd.DataFrame:
    """Within-pipeline RDM correlations across ROIs."""
    sub_dir = cache_root / subject
    betas = np.load(sub_dir / "betas.npy")
    image_ids = np.load(sub_dir / "image_ids.npy", allow_pickle=False)
    metadata = pd.read_parquet(sub_dir / "voxel_metadata.parquet")
    Y = aggregate_betas_for_images(betas, image_ids, common_ids)
    del betas

    rdm_per_roi = {}
    for roi in rois:
        if roi not in metadata.columns:
            continue
        mask = metadata[roi].astype(bool).to_numpy()
        if "ncsnr" in metadata.columns:
            mask = mask & (metadata["ncsnr"].astype(float).to_numpy() > 0.2)
        n_vox = int(mask.sum())
        if n_vox < 50:
            continue
        rdm_per_roi[roi] = (_rdm(Y[:, mask]), n_vox)
        _log(f"  {roi}: n_voxels={n_vox}")

    rows = []
    base_roi = "otc"
    base_rdm, base_n = rdm_per_roi[base_roi]
    base_tri = _upper(base_rdm)
    for roi, (rdm, n_vox) in rdm_per_roi.items():
        tri = _upper(rdm)
        finite = np.isfinite(base_tri) & np.isfinite(tri)
        r = float(np.corrcoef(base_tri[finite], tri[finite])[0, 1])
        rows.append({
            "subject": subject,
            "roi_a": base_roi,
            "roi_b": roi,
            "n_voxels_a": base_n,
            "n_voxels_b": n_vox,
            "rdm_correlation": r,
        })
    return pd.DataFrame(rows)


def diagnostic_2_ood_effect(
    cache_root: Path,
    subject: str,
    features_dir: Path,
    features_ood_dir: Path,
    model_h5: str,
) -> pd.DataFrame:
    """For one model on sub-01, wRSA with vs without OOD images."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV

    sub_dir = cache_root / subject
    betas = np.load(sub_dir / "betas.npy")
    image_ids = np.load(sub_dir / "image_ids.npy", allow_pickle=False)
    metadata = pd.read_parquet(sub_dir / "voxel_metadata.parquet")
    otc = metadata["otc"].astype(bool).to_numpy()
    ncsnr = metadata["ncsnr"].astype(float).to_numpy() > 0.2
    voxel_idx = np.where(otc & ncsnr)[0]

    main_h5 = features_dir / model_h5
    ood_h5 = features_ood_dir / model_h5

    with h5py.File(main_h5, "r") as f_main, h5py.File(ood_h5, "r") as f_ood:
        main_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f_main["image_ids"][:]]
        ood_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f_ood["image_ids"][:]]

        # All shared image_ids that appear in main features (1121 non-OOD).
        shared_non_ood = sorted([iid for iid in main_ids if iid.startswith("shared_") and not _is_ood(iid)])
        all_shared = sorted([iid for iid in main_ids if iid.startswith("shared_") and not _is_ood(iid)] +
                            [iid for iid in ood_ids if _is_ood(iid)])
        _log(f"  shared non-OOD: {len(shared_non_ood)}, all shared incl OOD: {len(all_shared)}")

        # Pick one mid-level layer to test on.
        layer_names = list(f_main["features_srp"].keys())
        target_layer = layer_names[len(layer_names) // 2]
        _log(f"  using layer {target_layer} of {len(layer_names)}")

        # Read full layer matrix from main + OOD, then index per condition.
        feats_main = f_main["features_srp"][target_layer][:]
        feats_ood = f_ood["features_srp"][target_layer][:]
        main_id_to_row = {iid: i for i, iid in enumerate(main_ids)}
        ood_id_to_row = {iid: i for i, iid in enumerate(ood_ids)}

    def features_for(target_ids):
        out = np.empty((len(target_ids), feats_main.shape[1]), dtype=np.float32)
        for k, iid in enumerate(target_ids):
            if iid in main_id_to_row:
                out[k] = feats_main[main_id_to_row[iid]]
            else:
                out[k] = feats_ood[ood_id_to_row[iid]]
        return out

    rows = []
    for label, ids in [("non_ood_1121", shared_non_ood), ("all_incl_ood_1492", all_shared)]:
        Y = aggregate_betas_for_images(betas, image_ids, ids)
        Y_roi = Y[:, voxel_idx]
        X = features_for(ids)
        X_train, X_test = X[::2], X[1::2]
        Y_train, Y_test = Y_roi[::2], Y_roi[1::2]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ridge = RidgeCV(
            alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
            alpha_per_target=True,
        )
        t0 = time.time()
        ridge.fit(X_train_s, Y_train)
        pred_train = ridge.predict(X_train_s)
        pred_test = ridge.predict(X_test_s)
        _log(f"  {label}: ridge fit in {time.time() - t0:.1f}s")

        crsa_test = float(np.corrcoef(_upper(_rdm(X_test)), _upper(_rdm(Y_test)))[0, 1])
        wrsa_train = float(np.corrcoef(_upper(_rdm(pred_train)), _upper(_rdm(Y_train)))[0, 1])
        wrsa_test = float(np.corrcoef(_upper(_rdm(pred_test)), _upper(_rdm(Y_test)))[0, 1])
        rows.append({
            "subject": subject,
            "model": Path(model_h5).stem,
            "layer": target_layer,
            "image_set": label,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "wrsa_train": wrsa_train,
            "wrsa_test": wrsa_test,
            "crsa_test": crsa_test,
        })
        _log(f"  {label}: wrsa_test={wrsa_test:.4f}, wrsa_train={wrsa_train:.4f}, crsa_test={crsa_test:.4f}")
    return pd.DataFrame(rows)


def main():
    cache_root = Path("/ptmp/rothj/conwell_replication/brain_cache_otc")
    features_dir = Path("/ptmp/rothj/conwell_replication/features")
    features_ood_dir = Path("/ptmp/rothj/conwell_replication/features_ood")
    out_dir = Path("/u/rothj/conwell_replication/figures")
    subject = "sub-01"
    image_ids_full = np.load(cache_root / subject / "image_ids.npy", allow_pickle=False)
    common = sorted({iid for iid in image_ids_full.tolist() if iid.startswith("shared_") and not _is_ood(iid)})

    _log("== Diagnostic 1: cross-ROI RDM correlation within new pipeline ==")
    rois_to_check = [
        "otc", "otc-streams", "otc-category",
        "laion-ventral", "laion-lateral",
        "v-objects", "l-objects",
        "OFA", "FFA-1", "FFA-2", "EBA", "FBA",
        "VWFA-1", "VWFA-2", "PPA", "OPA",
    ]
    df1 = diagnostic_1_cross_roi(cache_root, subject, rois_to_check, common)
    df1.to_csv(out_dir / "diag1_cross_roi_rdm_corr.csv", index=False)
    print()
    print(df1.to_string(index=False))
    print()

    _log("== Diagnostic 2: OOD-inflation effect on wRSA ==")
    df2 = diagnostic_2_ood_effect(
        cache_root, subject, features_dir, features_ood_dir, "alexnet_classification.h5",
    )
    df2.to_csv(out_dir / "diag2_ood_effect_on_wrsa.csv", index=False)
    print()
    print(df2.to_string(index=False))


if __name__ == "__main__":
    main()
