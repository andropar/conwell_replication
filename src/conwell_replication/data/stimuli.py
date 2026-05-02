"""Stimulus pool helpers (union of per-subject pools for feature extraction).

Building features for the replication requires extracting representations
for the **union of every per-subject pool** across the 5 subjects, so that
the same .h5 file can be sliced per subject. The image set is:

    1,121 cross-subject shared images
  + 5 × 4,712 subject-unique images
  = 24,681 unique images total.

The per-subject and shared pools live in :mod:`laion_fmri.splits`. This
module wraps those into one CSV that drives :mod:`conwell_replication.extract`.

CLI::

    python -m conwell_replication.data.stimuli build-pool \\
        --stimuli-root $WORK/conwell_replication/stimuli/image_sets \\
        --output       features/stimulus_pool.csv

The output CSV has columns::

    image_id   subject_pool   union_index   [image_path]

where ``subject_pool`` is a comma-separated list of subject IDs that
include this image in their pool (always includes ``"shared"`` for
cross-subject images), and ``union_index`` is the row position of the
image in the deduplicated, sorted union list.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# image_id → cache subdir resolver
# ---------------------------------------------------------------------------
# The dev-host stimulus cache layout (also used by the rsync-from-iris
# recipe). Filenames in ``deepvision_unique_<sub>/`` carry whichever
# participant-code suffix the source HDF5 used at acquisition time, which
# differs from the BIDS subject id for sub-03 ↔ sub-06 (swapped pair).
_UNIQUE_FILE_SUFFIX_TO_SUBJECT: Dict[str, str] = {
    "p01": "sub-01",
    "p02": "sub-06",  # swapped vs subject id
    "p03": "sub-05",
    "p04": "sub-03",  # swapped vs subject id
    "p05": "sub-07",
}


def resolve_image_path(image_id: str, stimuli_root: Path) -> Path:
    """Return the absolute on-disk path for ``image_id``.

    ``stimuli_root`` should mirror the development host's
    ``deepvision_fmri/cache/image_sets/`` layout (see the rsync command
    in the README).
    """
    stimuli_root = Path(stimuli_root)
    if image_id.startswith("shared_"):
        return stimuli_root / "deepvision_shared" / image_id
    if image_id.startswith("unique_"):
        suffix = image_id.rsplit("_", 1)[-1].split(".")[0]
        if suffix not in _UNIQUE_FILE_SUFFIX_TO_SUBJECT:
            raise ValueError(
                f"Unrecognized participant suffix in image_id {image_id!r}: "
                f"{suffix!r}. Expected one of "
                f"{list(_UNIQUE_FILE_SUFFIX_TO_SUBJECT)}."
            )
        sub = _UNIQUE_FILE_SUFFIX_TO_SUBJECT[suffix]
        return stimuli_root / f"deepvision_unique_{sub}" / image_id
    raise ValueError(
        f"image_id {image_id!r} doesn't start with 'shared_' or 'unique_' — "
        "don't know how to resolve."
    )


# ---------------------------------------------------------------------------
# Pool builder
# ---------------------------------------------------------------------------
def build_pool_csv(
    pools: Iterable[str] = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07"),
    stimuli_root: Optional[Path] = None,
    require_existence: bool = False,
) -> pd.DataFrame:
    """Build the union stimulus-pool DataFrame.

    Pulls per-pool image_ids from :mod:`laion_fmri.splits` (the union of
    train + test across all 11 splits per pool, which is identical to the
    pool itself).
    """
    from laion_fmri.splits import get_train_test_ids, list_splits

    pool_ids: Dict[str, set] = {}
    for pool in pools:
        ids: set = set()
        for split_name in list_splits():
            train, test = get_train_test_ids(split_name, pool=pool)
            ids.update(train)
            ids.update(test)
        pool_ids[pool] = ids

    union: set = set()
    for ids in pool_ids.values():
        union.update(ids)
    union_ids = sorted(union)

    by_id: Dict[str, List[str]] = {iid: [] for iid in union_ids}
    for pool, ids in pool_ids.items():
        for iid in ids:
            by_id[iid].append(pool)

    rows = []
    missing: List[str] = []
    for idx, iid in enumerate(union_ids):
        row = {
            "image_id":     iid,
            "subject_pool": ",".join(by_id[iid]),
            "union_index":  idx,
        }
        if stimuli_root is not None:
            path = resolve_image_path(iid, stimuli_root)
            row["image_path"] = str(path)
            if require_existence and not path.exists():
                missing.append(iid)
        rows.append(row)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} resolved paths do not exist on disk under "
            f"{stimuli_root} (e.g. {missing[:3]})."
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("build-pool", help="Build the union stimulus-pool CSV.")
    bp.add_argument("--output", type=Path, required=True,
                    help="Output CSV path.")
    bp.add_argument("--pools", nargs="+",
                    default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
                    help="Subject pools to include (default: all 5).")
    bp.add_argument("--stimuli-root", type=Path, default=None,
                    help="Path to deepvision-style image_sets directory. "
                         "If given, an 'image_path' column is added and "
                         "existence is verified.")
    bp.add_argument("--no-require-existence", action="store_true",
                    help="Skip the existence check on resolved image paths.")

    args = ap.parse_args(argv)

    if args.cmd == "build-pool":
        df = build_pool_csv(
            pools=tuple(args.pools),
            stimuli_root=args.stimuli_root,
            require_existence=(args.stimuli_root is not None and not args.no_require_existence),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows to {args.output}")
        for p in args.pools:
            n = (df["subject_pool"].str.contains(p)).sum()
            print(f"  {p}: {n} images in pool")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
