"""ROI mask helpers for LAION-fMRI / DeepNSD-style OTC evaluation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


OTC_CATEGORY_LABELS = (
    "OFA",
    "FFA-1",
    "FFA-2",
    "EBA",
    "FBA",
    "VWFA-1",
    "VWFA-2",
    "mfs-words",
    "pSTS-words",
    "PPA",
    "OPA",
)


DEFAULT_EVAL_ROIS = (
    "otc",
    "otc-streams",
    "otc-category",
    "laion-ventral",
    "laion-lateral",
    "v-objects",
    "l-objects",
    "OFA",
    "FFA-1",
    "FFA-2",
    "EBA",
    "FBA",
    "VWFA-1",
    "VWFA-2",
    "mfs-words",
    "pSTS-words",
    "PPA",
    "OPA",
)


_LABEL_RE = re.compile(r"_label-(.+?)_mask")


def label_from_mask_path(path: Path) -> str:
    match = _LABEL_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse ROI label from {path}")
    return match.group(1)


def load_t1w_roi_masks(
    subject: str,
    roi_root: Path,
    shape: Tuple[int, int, int],
) -> Dict[str, np.ndarray]:
    """Load all T1w ROI masks for ``subject``, unioning duplicate labels."""
    import nibabel as nib

    masks: Dict[str, np.ndarray] = {}
    subject_root = Path(roi_root) / subject
    for path in sorted(subject_root.glob("*/*space-T1w*_mask.nii.gz")):
        label = label_from_mask_path(path)
        img = nib.load(str(path))
        if img.shape != shape:
            raise RuntimeError(
                f"{path} shape {img.shape} does not match reference shape {shape}"
            )
        arr = np.asarray(img.dataobj) > 0
        if label in masks:
            masks[label] |= arr
        else:
            masks[label] = arr
    return masks


def build_otc_masks(masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Build the closest available analogue of Conwell et al.'s OTC mask."""
    required = ("laion-general", "laion-ventral", "laion-lateral")
    missing = [label for label in required if label not in masks]
    if missing:
        raise KeyError(f"Missing required OTC mask labels: {missing}")

    streams = masks["laion-general"] & (masks["laion-ventral"] | masks["laion-lateral"])
    category = np.zeros_like(streams, dtype=bool)
    for label in OTC_CATEGORY_LABELS:
        if label in masks:
            category |= masks[label]
    return {
        "otc-streams": streams,
        "otc-category": category,
        "otc": streams | category,
    }


def _composite_category_masks(masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    groups = {
        "faces": ("OFA", "FFA-1", "FFA-2"),
        "bodies": ("EBA", "FBA"),
        "words": ("VWFA-1", "VWFA-2", "mfs-words", "pSTS-words"),
        "places": ("PPA", "OPA"),
        "objects": ("v-objects", "l-objects"),
    }
    if not masks:
        return out
    template = next(iter(masks.values()))
    for name, labels in groups.items():
        arr = np.zeros_like(template, dtype=bool)
        any_present = False
        for label in labels:
            if label in masks:
                arr |= masks[label]
                any_present = True
        if any_present:
            out[name] = arr
    return out


def build_roi_metadata(
    final_mask: np.ndarray,
    r2: np.ndarray,
    masks: Dict[str, np.ndarray],
    extra_masks: Dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Create cache-column metadata for a final 3D voxel mask.

    The row order matches numpy boolean indexing ``arr[final_mask]`` used in
    ``brain_cache.py``.
    """
    if final_mask.shape != r2.shape:
        raise RuntimeError(f"mask shape {final_mask.shape} != r2 shape {r2.shape}")

    all_masks: Dict[str, np.ndarray] = {}
    all_masks.update(masks)
    all_masks.update(build_otc_masks(masks))
    all_masks.update(_composite_category_masks(masks))
    if extra_masks:
        all_masks.update(extra_masks)

    flat_idx = np.flatnonzero(final_mask.ravel(order="C"))
    ijk = np.column_stack(np.unravel_index(flat_idx, final_mask.shape, order="C"))
    meta = pd.DataFrame({
        "flat_index": flat_idx.astype(np.int64),
        "i": ijk[:, 0].astype(np.int16),
        "j": ijk[:, 1].astype(np.int16),
        "k": ijk[:, 2].astype(np.int16),
        "r2": r2[final_mask].astype(np.float32),
        "hlvis": np.ones(flat_idx.shape[0], dtype=np.int8),
    })
    for label, mask in sorted(all_masks.items()):
        if mask.shape != final_mask.shape:
            continue
        meta[label] = mask[final_mask].astype(np.int8)
    meta.index = [f"voxel_{i:06d}" for i in range(len(meta))]
    meta.index.name = "voxel_id"
    return meta


def available_roi_columns(
    metadata: pd.DataFrame,
    requested: Iterable[str] | None = None,
    min_voxels: int = 10,
) -> list[str]:
    """Return ROI columns with at least ``min_voxels`` true entries."""
    exclude = {"flat_index", "i", "j", "k", "r2", "hlvis"}
    if requested:
        candidates = list(requested)
    else:
        candidates = [c for c in DEFAULT_EVAL_ROIS if c in metadata.columns]
    out: list[str] = []
    for col in candidates:
        if col in exclude or col not in metadata.columns:
            continue
        n = int(metadata[col].astype(bool).sum())
        if n >= min_voxels:
            out.append(col)
    return out
