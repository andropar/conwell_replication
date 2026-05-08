"""LAION-fMRI benchmark adapter.

A thin wrapper around the ``laion_fmri`` package that exposes a
``DeepVisionBenchmark``-shaped interface so the existing analysis code
can talk to one object regardless of the underlying loader:

  * ``benchmark.subject``         — e.g. ``"sub-01"``
  * ``benchmark.pool``            — ``"shared"`` | ``"sub-XX"`` | ``"all"``
  * ``benchmark.voxel_set``       — ``"hlvis"`` | ``"visual"`` | ``"otc"``
  * ``benchmark.stimulus_data``   — DataFrame: ``image_name``, ``image_path``
  * ``benchmark.response_data``   — DataFrame of betas, voxels × images,
                                     index = ``voxel_id``, columns = image_ids
  * ``benchmark.metadata``        — voxel metadata, indexed by ``voxel_id``
  * ``benchmark.n_stimuli``       — int
  * ``benchmark.image_root``      — str

Pool selection uses :mod:`laion_fmri.splits` (the upstream catalogue): any
``pool`` argument valid for ``laion_fmri.splits.list_pools()`` is accepted
here, plus ``"all"`` to mean "every image with at least one trial".
"""

from __future__ import annotations

import logging
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)


def _resolve_root(root: Optional[Path]) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get("LAION_FMRI_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    raise RuntimeError(
        "LAION-fMRI data root not set. Pass `data_root=` or set the "
        "LAION_FMRI_ROOT environment variable."
    )


@dataclass
class _RawData:
    """Container for the trial-level arrays returned by the laion_fmri loader."""
    betas: np.ndarray            # (n_trials, n_voxels), float32
    trial_image_ids: List[str]   # length n_trials, may contain repeats
    voxel_metadata: pd.DataFrame # index = voxel_id, columns include 'hlvis'
    image_paths: Dict[str, str]  # image_id -> on-disk image path


# ---------------------------------------------------------------------------
# Cache-aware load (preferred path)
# ---------------------------------------------------------------------------
def _try_load_from_cache(subject: str, voxel_set: str) -> Optional[_RawData]:
    """If $CONWELL_BRAIN_CACHE/{subject}/{betas,image_ids}.npy exist, load
    them and return a _RawData. The cache is built by ``conwell-cache-brain``
    and currently only holds the responsive-cortex (R²>0.15) view, so we
    only short-circuit when ``voxel_set == "visual"``.

    We ``np.load`` the betas fully (no ``mmap_mode``) because the immediate
    next step in ``LAIONBenchmark.__init__`` is ``_aggregate`` — a
    sparse-matmul that touches every row. With mmap, the random row access
    pattern of sparse-matmul causes millions of page faults from /ptmp;
    a single sequential read of ~33 GB is dramatically faster
    (~30 s on GPFS) and the trial-level array is freed once aggregation
    completes (the public ``response_data`` is the much-smaller
    image-aggregated form).
    """
    cache_root = os.environ.get("CONWELL_BRAIN_CACHE")
    if not cache_root:
        return None
    cache_dir = Path(cache_root) / subject
    betas_p = cache_dir / "betas.npy"
    ids_p = cache_dir / "image_ids.npy"
    if not (betas_p.exists() and ids_p.exists()):
        return None
    info_p = cache_dir / "cache_info.json"
    if info_p.exists():
        info = json.loads(info_p.read_text())
        if info.get("voxel_set") != voxel_set:
            return None
    elif voxel_set != "visual":
        return None

    betas = np.load(betas_p)  # full read, not mmap
    image_ids = np.load(ids_p, allow_pickle=False).tolist()
    if betas.shape[0] != len(image_ids):
        raise RuntimeError(
            f"Cache row mismatch for {subject}: "
            f"betas rows={betas.shape[0]} vs ids={len(image_ids)}"
        )

    n_voxels = betas.shape[1]
    meta_p = cache_dir / "voxel_metadata.parquet"
    if meta_p.exists():
        voxel_metadata = pd.read_parquet(meta_p)
        if len(voxel_metadata) != n_voxels:
            raise RuntimeError(
                f"Cache metadata mismatch for {subject}: "
                f"metadata rows={len(voxel_metadata)} vs betas cols={n_voxels}"
            )
    else:
        voxel_metadata = pd.DataFrame({"hlvis": np.ones(n_voxels, dtype=int)})
    _LOG.info(
        f"Loaded {subject} from cache {cache_dir}: "
        f"{betas.shape[0]} trials × {n_voxels} voxels "
        f"({betas.nbytes / 1e9:.1f} GB in RAM)"
    )
    return _RawData(
        betas=betas,
        trial_image_ids=list(image_ids),
        voxel_metadata=voxel_metadata,
        image_paths={},
    )


# ---------------------------------------------------------------------------
# Connector to laion_fmri (kept narrow: pure I/O)
# ---------------------------------------------------------------------------
def _laion_fmri_load(
    subject: str,
    data_root: Path,
    voxel_set: str,
    roi_name: str = "hlvis",
) -> _RawData:
    """Load trial-level betas + ROI metadata + image paths via laion_fmri.

    Iterates every session that's been downloaded for ``subject`` (laion_fmri
    only stores betas per-session in the bucket, so cross-session
    aggregation is the caller's job here), pre-filtering voxels by
    ``roi_name`` when ``voxel_set == "hlvis"``.

    Short-circuits to the on-disk cache (``conwell-cache-brain`` output) if
    ``$CONWELL_BRAIN_CACHE`` is set and ``voxel_set == "visual"``; the eval
    sweep relies on that to avoid 30s of NIfTI reads per task.
    """
    cached = _try_load_from_cache(subject, voxel_set)
    if cached is not None:
        return cached

    try:
        from laion_fmri.subject import Subject
        from laion_fmri._paths import stimuli_dir_path
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "laion_fmri is not installed. Install it on the target server "
            "(see https://laion-fmri.hebartlab.com/laion_fmri_package/) and "
            "set LAION_FMRI_ROOT to the downloaded data directory."
        ) from exc

    s = Subject(subject_id=subject, data_dir=str(data_root))

    sessions = s.get_sessions()
    if not sessions:
        raise RuntimeError(
            f"No sessions found for {subject} under {data_root}. "
            f"Did you finish downloading via laion_fmri.download?"
        )

    available_rois = s.get_available_rois()
    if voxel_set == "hlvis":
        if roi_name not in available_rois:
            raise ValueError(
                f"voxel_set='hlvis' requires ROI {roi_name!r} but it is not "
                f"available for {subject}. Available ROIs: {available_rois}. "
                f"Pass a different `roi_name` to LAIONBenchmark, or edit "
                f"_laion_fmri_load if the new dataset uses a different name."
            )
        roi_filter = roi_name
    else:
        roi_filter = None  # "visual" → all brain-mask voxels

    _LOG.info(
        f"Loading betas for {subject}: {len(sessions)} sessions, "
        f"voxel_set={voxel_set}, roi_filter={roi_filter}"
    )

    all_betas: List[np.ndarray] = []
    all_labels: List[str] = []
    for sess in sessions:
        betas_sess = s.get_betas(session=sess, roi=roi_filter)
        trials = s.get_trial_info(session=sess)
        if "label" not in trials.columns:
            raise RuntimeError(
                f"events.tsv for {subject}/{sess} has no 'label' column."
            )
        if betas_sess.shape[0] != len(trials):
            raise RuntimeError(
                f"{subject}/{sess}: betas have {betas_sess.shape[0]} trials "
                f"but events.tsv has {len(trials)} rows."
            )
        all_betas.append(betas_sess.astype(np.float32, copy=False))
        all_labels.extend(trials["label"].astype(str).tolist())

    betas = np.concatenate(all_betas, axis=0)  # (total_trials, n_voxels)
    n_voxels = betas.shape[1]
    _LOG.info(
        f"  {subject}: {betas.shape[0]} trials × {n_voxels} voxels "
        f"({len(set(all_labels))} unique image_ids)"
    )

    # We pre-filtered to the requested ROI, so mark every row hlvis=1
    voxel_metadata = pd.DataFrame({"hlvis": np.ones(n_voxels, dtype=int)})

    image_paths: Dict[str, str] = {}
    if s.has_stimuli():
        meta = s.get_stimulus_metadata()
        stim_dir = stimuli_dir_path(Path(data_root))
        for _, row in meta.iterrows():
            fn = str(row["filename"])
            image_paths[fn] = str(stim_dir / fn)

    return _RawData(
        betas=betas,
        trial_image_ids=all_labels,
        voxel_metadata=voxel_metadata,
        image_paths=image_paths,
    )


# ---------------------------------------------------------------------------
# Pool resolution via laion_fmri.splits
# ---------------------------------------------------------------------------
def _pool_image_ids(pool: str) -> Optional[set]:
    """Return the set of image_ids in ``pool``, or ``None`` for "all"."""
    if pool == "all":
        return None
    from laion_fmri.splits import list_pools, list_splits, get_train_test_ids
    if pool not in list_pools():
        raise ValueError(
            f"Unknown pool {pool!r}. Valid: {list_pools() + ['all']}"
        )
    # The pool's image set is the union of train+test across all splits
    # for that pool. Identical regardless of split chosen, but unioning
    # over a few splits is cheap and avoids relying on any single one.
    out: set = set()
    for name in list_splits():
        train, test = get_train_test_ids(name, pool=pool)
        out.update(train)
        out.update(test)
    return out


# ---------------------------------------------------------------------------
# LAIONBenchmark — DeepVisionBenchmark-shaped adapter
# ---------------------------------------------------------------------------
class LAIONBenchmark:
    """Adapter exposing the DeepVisionBenchmark interface over LAION-fMRI data.

    Parameters
    ----------
    subject:
        ``"sub-01" | "sub-03" | "sub-05" | "sub-06" | "sub-07"``.
    voxel_set:
        ``"hlvis"`` (default), ``"visual"``, or cached ``"otc"``.
    pool:
        Stimulus subset to expose. One of:

        - ``"shared"`` — the 1,121 cross-subject shared images (matches
          ``laion_fmri.splits`` shared pool). Use for shared-only
          replications (NSD/Conwell-style).
        - ``"sub-01"`` / ... / ``"sub-07"`` — that subject's full
          5,833-image pool from ``laion_fmri.splits``.
        - ``"all"`` — every image with at least one trial for this subject.
    aggregate_repeats:
        If ``True`` (default), average betas across trials of the same
        image_id so ``response_data`` is voxels × images. If ``False``,
        ``response_data`` keeps trials separate.
    data_root:
        Path to the laion_fmri data directory. Falls back to
        ``$LAION_FMRI_ROOT``.
    roi_name:
        Name of the high-level-visual ROI in laion_fmri's ``rois/``
        dir, used when ``voxel_set='hlvis'``.
    """

    def __init__(
        self,
        subject: str,
        voxel_set: str = "hlvis",
        pool: str = "shared",
        aggregate_repeats: bool = True,
        data_root: Optional[Path] = None,
        roi_name: str = "hlvis",
        ncsnr_threshold: Optional[float] = 0.2,
    ):
        if voxel_set not in ("hlvis", "visual", "otc"):
            raise ValueError(
                f"voxel_set must be 'hlvis', 'visual', or 'otc', got '{voxel_set}'"
            )

        self.subject = subject
        self.voxel_set = voxel_set
        self.pool = pool
        self.ncsnr_threshold = ncsnr_threshold

        self._data_root = _resolve_root(data_root)
        _LOG.info(f"Loading LAION-fMRI for {subject} pool={pool} from {self._data_root}")

        raw = _laion_fmri_load(subject, self._data_root, voxel_set, roi_name=roi_name)
        self._raw = raw

        # Aggregate trials → per-image betas. After this we no longer need
        # the trial-level array (~33 GB on the responsive-cortex view), so
        # null it out and gc — saves a chunky 33 GB during the rest of
        # __init__ and through every subsequent eval call. Noise-ceiling
        # code that wants trial-level access can re-load via
        # ``_laion_fmri_load`` directly.
        betas, image_ids = self._aggregate(
            raw.betas, raw.trial_image_ids, aggregate=aggregate_repeats,
        )
        raw.betas = None
        import gc
        gc.collect()

        # Pool filter via laion_fmri.splits
        keep_set = _pool_image_ids(pool)
        if keep_set is not None:
            keep_mask = np.array([iid in keep_set for iid in image_ids])
            n_kept = int(keep_mask.sum())
            if n_kept == 0:
                raise ValueError(
                    f"Pool {pool!r} has 0 image_ids matching this subject's "
                    f"trial labels. Check that {pool!r} is consistent with "
                    f"subject {subject!r} (subject pools are subject-specific)."
                )
            image_ids = [iid for iid, k in zip(image_ids, keep_mask) if k]
            betas = betas[:, keep_mask]
            _LOG.info(f"  pool {pool!r}: kept {n_kept} of {len(keep_mask)} images")

        # Voxel mask (we already pre-filtered ROI in the loader; this is a no-op
        # for hlvis, but keeps the metadata column shape honest)
        meta = raw.voxel_metadata.copy()
        if voxel_set == "hlvis":
            voxel_mask = meta["hlvis"].astype(bool).to_numpy()
            betas = betas[voxel_mask, :]
            meta = meta.loc[voxel_mask]

        # Conwell-style NCSNR filter: drop voxels with ncsnr <= threshold to
        # match the paper's NCSNR > 0.2 voxel selection. Requires that
        # compute_ncsnr.py has merged 'ncsnr' into voxel_metadata.parquet.
        if ncsnr_threshold is not None and "ncsnr" in meta.columns:
            ncsnr_mask = (meta["ncsnr"].astype(float).to_numpy() > ncsnr_threshold)
            n_before = int(ncsnr_mask.size)
            n_after = int(ncsnr_mask.sum())
            betas = betas[ncsnr_mask, :]
            meta = meta.loc[ncsnr_mask]
            _LOG.info(
                f"  ncsnr > {ncsnr_threshold}: kept {n_after} of {n_before} voxels "
                f"({100 * n_after / n_before:.1f}%)"
            )
        elif ncsnr_threshold is not None:
            _LOG.warning(
                "  ncsnr_threshold set but voxel_metadata has no 'ncsnr' column; "
                "run scripts/compute_ncsnr.py first."
            )

        # Public-facing frames
        self.metadata: pd.DataFrame = meta
        self.metadata.index.name = "voxel_id"

        self.response_data: pd.DataFrame = pd.DataFrame(
            betas, index=meta.index, columns=image_ids,
        )
        self.response_data.index.name = "voxel_id"

        self.stimulus_data: pd.DataFrame = pd.DataFrame({
            "image_name": image_ids,
            "image_path": [raw.image_paths.get(iid, "") for iid in image_ids],
        })
        self.image_root = str(self._data_root)
        self.n_stimuli = len(image_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate(
        betas: np.ndarray,
        trial_image_ids: List[str],
        aggregate: bool,
    ) -> tuple:
        """Collapse trials → image-mean (or pass through if disabled).

        Implemented as a sparse-dense matmul: build a sparse
        ``(n_unique, n_trials)`` averaging matrix ``W`` whose rows are
        per-image indicator-and-divide-by-count vectors, then ``W @ betas``
        gives the per-image mean. scipy.sparse uses BLAS-threaded sparse
        routines, which scales with ``--cpus-per-task`` — unlike the
        prior ``np.add.at`` (single-threaded) and ``pandas.groupby.mean``
        (column-by-column iteration) versions, both of which took
        ~20 min on 33 GB.
        """
        if not aggregate:
            return betas.T.astype(np.float32), list(trial_image_ids)

        from scipy.sparse import csr_matrix

        ids_arr = np.asarray(trial_image_ids)
        unique_ids, inverse = np.unique(ids_arr, return_inverse=True)
        n_unique = len(unique_ids)
        n_trials = betas.shape[0]

        counts = np.bincount(inverse, minlength=n_unique)
        weights = (1.0 / counts[inverse]).astype(np.float32)

        # CSR is the right format for the row-major (per-unique-image) reduction
        # that follows: each row corresponds to one unique image_id and
        # contains 1/count entries at the trial-indices that map to it.
        averaging = csr_matrix(
            (weights, (inverse, np.arange(n_trials))),
            shape=(n_unique, n_trials),
            dtype=np.float32,
        )

        # If betas is a numpy memmap, the matmul still pages it in once
        # sequentially; that's the unavoidable cost of touching all 33 GB.
        means = (averaging @ betas).astype(np.float32, copy=False)

        # Final shape is (voxels, images), matching response_data convention
        return means.T, list(unique_ids)

    def voxel_indices(self, roi: str = "hlvis") -> np.ndarray:
        """Positional indices (into ``response_data``) of voxels in ``roi``."""
        if roi not in self.metadata.columns:
            raise ValueError(
                f"ROI {roi!r} not in metadata. Have: {list(self.metadata.columns)}"
            )
        return np.where(self.metadata[roi].astype(bool).to_numpy())[0]

    def betas(
        self,
        image_ids: Optional[List[str]] = None,
        as_array: bool = True,
    ):
        """Return betas restricted to ``image_ids`` (defaulting to all)."""
        if image_ids is None:
            df = self.response_data
        else:
            missing = set(image_ids) - set(self.response_data.columns)
            if missing:
                raise KeyError(
                    f"image_ids not in response_data: {sorted(missing)[:5]}..."
                )
            df = self.response_data.loc[:, image_ids]
        return df.to_numpy() if as_array else df

    # ------------------------------------------------------------------
    # Trial-level access (used by noise ceiling)
    # ------------------------------------------------------------------
    def trial_splithalf_rdms(
        self,
        image_ids: Optional[List[str]] = None,
        seed: int = 0,
    ) -> tuple:
        """Compute odd/even-trial split-half RDMs for the given image_ids.

        For each image_id, sort its trials, split into odd/even halves,
        average within each half, then compute (1 − Pearson r) across
        image_ids → two (n_images, n_images) RDMs.

        Returns
        -------
        rdm_odd, rdm_even, kept_ids
        """
        raw = self._raw
        if self.voxel_set == "hlvis":
            voxel_mask = raw.voxel_metadata["hlvis"].astype(bool).to_numpy()
            betas = raw.betas[:, voxel_mask]
        else:
            betas = raw.betas

        trial_iids = list(raw.trial_image_ids)
        target = set(image_ids) if image_ids is not None else set(self.stimulus_data["image_name"])

        rng = np.random.default_rng(seed)
        groups: dict = {}
        for tidx, iid in enumerate(trial_iids):
            if iid in target:
                groups.setdefault(iid, []).append(tidx)

        ordered_ids = sorted(groups.keys())
        odd_means: List[np.ndarray] = []
        even_means: List[np.ndarray] = []
        for iid in ordered_ids:
            trial_indices = np.array(sorted(groups[iid]))
            if len(trial_indices) < 2:
                continue
            perm = rng.permutation(trial_indices)
            odd_means.append(betas[perm[0::2]].mean(axis=0))
            even_means.append(betas[perm[1::2]].mean(axis=0))

        kept = [iid for iid in ordered_ids if iid in groups and len(groups[iid]) >= 2]
        Y_odd = np.stack(odd_means, axis=0)
        Y_even = np.stack(even_means, axis=0)
        rdm_odd = 1.0 - np.corrcoef(Y_odd)
        rdm_even = 1.0 - np.corrcoef(Y_even)
        return rdm_odd, rdm_even, kept
