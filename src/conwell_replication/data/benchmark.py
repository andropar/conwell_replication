"""LAION-fMRI benchmark adapter.

This is a thin wrapper around the ``laion_fmri`` package that exposes the same
interface the existing analysis code expects (mirroring
``cstims.datasets.deepvision.DeepVisionBenchmark``):

  * ``benchmark.subject``         — e.g. ``"sub-01"``
  * ``benchmark.image_set``       — ``"shared"`` | ``"unique"`` | ``"pool"``
  * ``benchmark.voxel_set``       — ``"hlvis"`` | ``"visual"``
  * ``benchmark.stimulus_data``   — ``DataFrame`` with at minimum:
        ``image_name``, ``image_path``, optional ``dataset``
  * ``benchmark.response_data``   — ``DataFrame`` of betas, shape (voxels × stimuli),
        index = ``voxel_id``, columns aligned with ``stimulus_data`` row order
  * ``benchmark.metadata``        — ``DataFrame`` of voxel metadata, index = ``voxel_id``,
        with at least an ``hlvis`` (1/0) column
  * ``benchmark.n_stimuli``       — ``int``
  * ``benchmark.image_root``      — ``str``

The original DeepVisionBenchmark relied on a hard-coded local layout and
GLMsingle HDF5 files. The new server has the data through ``laion_fmri``,
which exposes a different but functionally equivalent API
(``Subject`` for per-subject access, ``Group`` for multi-subject queries —
see https://laion-fmri.hebartlab.com/laion_fmri_package/index.html).

Because the laion_fmri package isn't installed yet on this development host,
the connector below is structured around the documented API and contains
``TODO`` markers for places that need verification once you can point at a
real download. The dataclass + interface are fixed; only the concrete I/O
needs to be filled in, and downstream code (``eval/*``, ``extract/*``) talks
exclusively to the documented attributes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Subject / participant mapping
# ---------------------------------------------------------------------------
# This mirrors the DeepVision-era mapping. If the freshly-preprocessed data
# uses a different convention (e.g. sequential sub-01..sub-05 with no gaps),
# patch this dict and re-export.
SUBJECT_TO_PARTICIPANT: Dict[str, str] = {
    "sub-01": "p01",
    "sub-03": "p02",
    "sub-05": "p03",
    "sub-06": "p04",
    "sub-07": "p05",
}

PARTICIPANT_TO_SUBJECT: Dict[str, str] = {v: k for k, v in SUBJECT_TO_PARTICIPANT.items()}


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
    """Container for the raw arrays returned by the laion_fmri loader."""
    betas: np.ndarray            # (n_trials, n_voxels), float32
    trial_image_ids: List[str]   # length n_trials, may contain repeats
    voxel_metadata: pd.DataFrame # index = voxel_id, columns include 'hlvis'
    image_paths: Dict[str, str]  # image_id -> on-disk image path


# ---------------------------------------------------------------------------
# Connector to laion_fmri (kept narrow: pure I/O)
# ---------------------------------------------------------------------------
def _laion_fmri_load(
    subject: str,
    data_root: Path,
    voxel_set: str,
) -> _RawData:
    """Load betas + metadata + image paths from the laion_fmri package.

    NOTE: filled in once laion_fmri is available on the target server.
    Documented API (per the package README) is approximately::

        from laion_fmri import Subject
        s = Subject(subject_id=subject, data_dir=data_root)
        betas        = s.get_betas()                      # (n_trials, n_voxels) f32
        trial_meta   = s.get_trial_metadata()             # DataFrame, has image_id
        voxel_meta   = s.get_voxel_metadata()             # DataFrame, hlvis mask, etc.
        image_paths  = s.get_image_paths()                # {image_id: path}
        # plus: noise ceiling, ROI masks, anatomicals — see docs

    Replace the body below with the matching laion_fmri calls. The shape
    contract is what matters for the downstream code.
    """
    raise NotImplementedError(
        "laion_fmri integration is not wired up yet. Edit "
        "src/conwell_replication/data/benchmark.py:_laion_fmri_load to "
        "translate the laion_fmri Subject API into a _RawData. The "
        "downstream code only depends on the documented interface."
    )


# ---------------------------------------------------------------------------
# LAIONBenchmark — DeepVisionBenchmark-shaped adapter
# ---------------------------------------------------------------------------
class LAIONBenchmark:
    """Adapter exposing the DeepVisionBenchmark interface over LAION-fMRI data.

    Args
    ----
    subject:
        ``"sub-01" | "sub-03" | "sub-05" | "sub-06" | "sub-07"``.
    voxel_set:
        ``"hlvis"`` (default) or ``"visual"``. Restricts ``response_data`` /
        ``metadata`` to that ROI.
    image_pool:
        Which stimuli to expose:

        - ``"shared"``: only stimuli flagged as shared across subjects
          (~1492 images on the original DeepVision data, ≈1121 on min_nn).
          Aligned with the existing rsa_20260223_154344 split-half pipeline.
        - ``"full"``: the per-subject ``n_train + n_test = 5833`` pool used
          by the min_nn splits.
        - ``"all"``: every image with at least one trial for this subject.
    aggregate_repeats:
        If ``True`` (default), average betas across all trials of the same
        ``image_id`` so ``response_data`` has one column per image. If
        ``False``, ``response_data`` is trial-level and column ordering is
        not guaranteed.
    data_root:
        Path to the laion_fmri data directory. Falls back to
        ``$LAION_FMRI_ROOT``.
    """

    def __init__(
        self,
        subject: str,
        voxel_set: str = "hlvis",
        image_pool: str = "shared",
        aggregate_repeats: bool = True,
        data_root: Optional[Path] = None,
    ):
        if subject not in SUBJECT_TO_PARTICIPANT:
            raise ValueError(
                f"Unknown subject '{subject}'. "
                f"Valid: {list(SUBJECT_TO_PARTICIPANT.keys())}"
            )
        if voxel_set not in ("hlvis", "visual"):
            raise ValueError(f"voxel_set must be 'hlvis' or 'visual', got '{voxel_set}'")
        if image_pool not in ("shared", "full", "all"):
            raise ValueError(f"image_pool must be 'shared'|'full'|'all', got '{image_pool}'")

        self.subject = subject
        self.participant = SUBJECT_TO_PARTICIPANT[subject]
        self.voxel_set = voxel_set
        self.image_set = image_pool          # alias kept for compatibility
        self.image_pool = image_pool

        self._data_root = _resolve_root(data_root)
        _LOG.info(
            f"Loading LAION-fMRI for {subject} ({self.participant}) "
            f"from {self._data_root}"
        )

        raw = _laion_fmri_load(subject, self._data_root, voxel_set)
        self._raw = raw  # keep for trial-level access (e.g. noise ceiling)

        # 1. Aggregate trials → per-image betas, deterministic image_id ordering
        betas, image_ids = self._aggregate(
            raw.betas, raw.trial_image_ids, aggregate=aggregate_repeats
        )

        # 2. Pool selection
        keep_ids = self._select_pool(image_ids, raw.voxel_metadata)
        keep_mask = np.array([iid in keep_ids for iid in image_ids])
        image_ids = [iid for iid, k in zip(image_ids, keep_mask) if k]
        betas = betas[:, keep_mask]

        # 3. Voxel mask
        meta = raw.voxel_metadata.copy()
        if voxel_set == "hlvis":
            voxel_mask = meta["hlvis"].astype(bool).to_numpy()
            betas = betas[voxel_mask, :]
            meta = meta.loc[voxel_mask]

        # 4. Build the public-facing frames
        self.metadata: pd.DataFrame = meta
        self.metadata.index.name = "voxel_id"

        self.response_data: pd.DataFrame = pd.DataFrame(
            betas, index=meta.index, columns=image_ids
        )
        self.response_data.index.name = "voxel_id"

        self.stimulus_data: pd.DataFrame = pd.DataFrame(
            {
                "image_name": image_ids,
                "image_path": [raw.image_paths.get(iid, "") for iid in image_ids],
            }
        )
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
        """Collapse trials → image-mean (or pass through if disabled)."""
        if not aggregate:
            return betas.T.astype(np.float32), list(trial_image_ids)

        df = pd.DataFrame(betas)
        df["__iid__"] = trial_image_ids
        agg = df.groupby("__iid__", sort=True).mean()
        ordered_ids = agg.index.tolist()
        # Shape: voxels × images
        return agg.to_numpy(dtype=np.float32).T, ordered_ids

    def _select_pool(
        self,
        image_ids: List[str],
        voxel_metadata: pd.DataFrame,  # noqa: ARG002 — placeholder for future use
    ) -> set:
        """Select which image_ids to keep based on ``self.image_pool``.

        - ``"all"``: keep every image_id we have betas for.
        - ``"shared"``: keep only images marked shared across subjects.
        - ``"full"``: keep the per-subject train+test pool (~5833 images).

        How "shared" and "full" are identified depends on what ``laion_fmri``
        exposes. By the dataset's naming convention, shared images use the
        prefix ``shared_*`` while subject-unique images use participant-
        specific tags. This is a reasonable default; refine once we know
        the exact metadata schema.
        """
        if self.image_pool == "all":
            return set(image_ids)

        if self.image_pool == "shared":
            # Heuristic: filenames like ``shared_12rep_LAION_cluster_*.jpg``.
            # Adjust if laion_fmri exposes an explicit "shared" flag.
            return {iid for iid in image_ids if iid.startswith("shared_")}

        # image_pool == "full"
        # The per-subject full pool is the union of all min_nn splits
        # (5833 images for this participant). We don't have a direct
        # lookup table here without round-tripping through the splits
        # module; downstream code that wants the full pool should call
        # `LAIONBenchmark(image_pool="all")` and then intersect with the
        # split's image_ids when running the evaluation. Returning the
        # full set here is a safe superset.
        return set(image_ids)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def voxel_indices(self, roi: str = "hlvis") -> np.ndarray:
        """Positional indices (into ``response_data``) of voxels in ``roi``."""
        if roi not in self.metadata.columns:
            raise ValueError(f"ROI '{roi}' not in metadata. Have: {list(self.metadata.columns)}")
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
                    f"image_ids not in response_data: {sorted(list(missing))[:5]}..."
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
        average within each half, then compute (1 - Pearson r) across
        image_ids → two (n_images, n_images) RDMs.

        Returns
        -------
        rdm_odd, rdm_even, kept_ids
        """
        raw = self._raw
        # Same hlvis voxel mask we applied to response_data
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
            # Random permutation, then odd/even (so we don't bias by acquisition order)
            perm = rng.permutation(trial_indices)
            odd_means.append(betas[perm[0::2]].mean(axis=0))
            even_means.append(betas[perm[1::2]].mean(axis=0))

        kept = [iid for iid in ordered_ids if iid in groups and len(groups[iid]) >= 2]
        Y_odd = np.stack(odd_means, axis=0)    # (n_kept, n_voxels)
        Y_even = np.stack(even_means, axis=0)
        rdm_odd = 1.0 - np.corrcoef(Y_odd)
        rdm_even = 1.0 - np.corrcoef(Y_even)
        return rdm_odd, rdm_even, kept
