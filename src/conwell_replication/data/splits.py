"""Load min_nn split JSONs.

Each split JSON has the schema produced by
``experiments/generalization_split/min_nn/build_splits.py``::

    {
      "name":       "random_0",                    # split identifier
      "pool":       "sub-01 full pool (...)",
      "splitter":   "min_nn_stochastic",
      "params":     {...},
      "n_train":    4666,
      "n_test":     1167,
      "variants": [
        {
          "variant_id": 0,
          "train_ids":  [...image_id strings...],
          "test_ids":   [...image_id strings...]
        },
        ...
      ]
    }

Splits are stored under ``resources/splits/p01..p05/{name}.json`` (one file per
participant × split). Per the build script, ``random_*`` and ``cluster_k5_*``
each have a single variant indexed by the seed/cluster, while the ``tau_*``
splits have a single variant_id=0.

This module exposes:

- :func:`list_splits` to enumerate split files for a participant
- :func:`load_split` to load one split → :class:`Split`
- :func:`load_all_splits` to load every split for a participant
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Resolve packaged splits dir at import time. Users can also pass an explicit
# splits root to the loaders below.
_DEFAULT_SPLITS_ROOT = Path(__file__).resolve().parents[3] / "resources" / "splits"


@dataclass
class SplitVariant:
    """A single train/test image-id partition within a Split."""
    variant_id: int
    train_ids: List[str]
    test_ids: List[str]

    def all_ids(self) -> List[str]:
        return list(self.train_ids) + list(self.test_ids)


@dataclass
class Split:
    """One named split (e.g. ``random_0``) for one participant."""
    name: str
    participant: str
    pool: str
    splitter: str
    params: Dict
    n_train: int
    n_test: int
    variants: List[SplitVariant] = field(default_factory=list)

    @property
    def split_family(self) -> str:
        """Coarse family: 'random', 'cluster_k5', 'tau' (used for grouping)."""
        n = self.name
        if n.startswith("random_"):
            return "random"
        if n.startswith("cluster_k5_"):
            return "cluster_k5"
        if n.startswith("tau_"):
            return "tau"
        return "other"


def _resolve_splits_root(splits_root: Optional[Path]) -> Path:
    if splits_root is None:
        return _DEFAULT_SPLITS_ROOT
    return Path(splits_root)


def list_splits(
    participant: str,
    splits_root: Optional[Path] = None,
) -> List[str]:
    """Return the available split names for a participant (e.g. ``"p01"``)."""
    root = _resolve_splits_root(splits_root) / participant
    if not root.is_dir():
        raise FileNotFoundError(
            f"No splits directory for participant '{participant}' at {root}"
        )
    return sorted(p.stem for p in root.glob("*.json"))


def load_split(
    participant: str,
    name: str,
    splits_root: Optional[Path] = None,
) -> Split:
    """Load a single split JSON → :class:`Split`."""
    path = _resolve_splits_root(splits_root) / participant / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    raw = json.loads(path.read_text())
    variants = [
        SplitVariant(
            variant_id=int(v["variant_id"]),
            train_ids=list(v["train_ids"]),
            test_ids=list(v["test_ids"]),
        )
        for v in raw["variants"]
    ]
    return Split(
        name=raw["name"],
        participant=participant,
        pool=raw.get("pool", ""),
        splitter=raw.get("splitter", ""),
        params=raw.get("params", {}),
        n_train=int(raw["n_train"]),
        n_test=int(raw["n_test"]),
        variants=variants,
    )


def load_all_splits(
    participant: str,
    splits_root: Optional[Path] = None,
) -> Dict[str, Split]:
    """Load every split JSON for a participant → ``{name: Split}``."""
    return {
        name: load_split(participant, name, splits_root)
        for name in list_splits(participant, splits_root)
    }


def collect_pool_ids(
    splits_by_participant: Dict[str, Dict[str, Split]],
) -> Dict[str, List[str]]:
    """Build {participant: sorted-unique-image-ids} from a nested mapping.

    Use this to compute the per-participant stimulus pool needed by feature
    extraction. The pool for a participant is the union of all train_ids ∪
    test_ids across all variants of all splits — in practice this is the
    ``n_train + n_test`` images repeated identically across splits, but we do
    a union to be safe.
    """
    out: Dict[str, List[str]] = {}
    for participant, splits in splits_by_participant.items():
        ids: set = set()
        for split in splits.values():
            for v in split.variants:
                ids.update(v.train_ids)
                ids.update(v.test_ids)
        out[participant] = sorted(ids)
    return out


def union_pool(
    pool_by_participant: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, List[int]]]:
    """Build the union-of-pools image list and per-participant index lists.

    Returns
    -------
    union_ids:
        Sorted, deduplicated list of image_ids across all participants.
    indices_by_participant:
        ``{participant: [union_index_for_each_id_in_pool_by_participant]}``
        — i.e. for each id in the participant's pool, the index into
        ``union_ids``. Useful when slicing feature arrays per subject.
    """
    union: set = set()
    for ids in pool_by_participant.values():
        union.update(ids)
    union_ids = sorted(union)
    id_to_idx = {iid: i for i, iid in enumerate(union_ids)}
    indices_by_participant = {
        p: [id_to_idx[iid] for iid in ids]
        for p, ids in pool_by_participant.items()
    }
    return union_ids, indices_by_participant
