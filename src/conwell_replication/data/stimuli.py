"""Stimulus pool helpers.

Compute the union of per-subject min_nn pools to drive feature extraction.
Each participant has a 5833-image pool (1121 shared + 4712 subject-unique).
The pools differ per subject. The union over all 5 participants is **24,681
images**.

Empirically, the on-disk layout (on iris.cbs.mpg.de / the development host)
is::

    /SSD/jroth/deepvision_fmri/cache/image_sets/
        deepvision_shared/                 # 1492 jpgs, every shared_*.jpg
        deepvision_unique_sub-01/          # 4712 jpgs, files end _p01.jpg
        deepvision_unique_sub-03/          # 4712 jpgs, files end _p04.jpg  (!)
        deepvision_unique_sub-05/          # 4712 jpgs, files end _p03.jpg
        deepvision_unique_sub-06/          # 4712 jpgs, files end _p02.jpg  (!)
        deepvision_unique_sub-07/          # 4712 jpgs, files end _p05.jpg

The unique-cache directories are named after the **subject** but their
filenames carry whichever participant code the source HDF5 used (so the
sub-03 cache contains files ending in ``_p04.jpg`` because in the original
DeepVision dataset, sub-03's unique images were drawn from
``stimuli_participant_p04.hdf5``).

This module exposes:

- :func:`build_pool_csv` (plus the ``build-pool`` CLI) — emit the union
  pool with optional ``image_path`` column resolved against a stimuli root.
- :func:`resolve_image_path` — image_id → on-disk path lookup.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .splits import collect_pool_ids, load_all_splits, union_pool


# ---------------------------------------------------------------------------
# image_id → cache subdir
# ---------------------------------------------------------------------------
# A min_nn unique image_id encodes the participant code as ``_pNN.jpg``.
# Map that suffix → the deepvision cache subdirectory that physically
# contains the file (see module docstring above).
_UNIQUE_PARTICIPANT_TO_SUBJECT: Dict[str, str] = {
    "p01": "sub-01",
    "p02": "sub-06",  # swapped vs shared mapping
    "p03": "sub-05",
    "p04": "sub-03",  # swapped vs shared mapping
    "p05": "sub-07",
}


def resolve_image_path(image_id: str, stimuli_root: Path) -> Path:
    """Return the absolute on-disk path for ``image_id``.

    ``stimuli_root`` should point at a directory laid out like the
    development host's ``deepvision_fmri/cache/image_sets/`` (see module
    docstring). Symlinks and rsync targets that mirror that layout work
    the same way.
    """
    stimuli_root = Path(stimuli_root)
    if image_id.startswith("shared_"):
        return stimuli_root / "deepvision_shared" / image_id
    if image_id.startswith("unique_"):
        # Filename ends in _pNN.jpg; map to subject's cache dir
        suffix = image_id.rsplit("_", 1)[-1].split(".")[0]  # 'p04'
        if suffix not in _UNIQUE_PARTICIPANT_TO_SUBJECT:
            raise ValueError(
                f"Unrecognized participant suffix in image_id {image_id!r}: "
                f"{suffix!r}. Expected one of {list(_UNIQUE_PARTICIPANT_TO_SUBJECT)}."
            )
        sub = _UNIQUE_PARTICIPANT_TO_SUBJECT[suffix]
        return stimuli_root / f"deepvision_unique_{sub}" / image_id
    raise ValueError(
        f"image_id {image_id!r} doesn't start with 'shared_' or 'unique_' — "
        "don't know how to resolve."
    )


# ---------------------------------------------------------------------------
# Pool builder
# ---------------------------------------------------------------------------
def build_pool_csv(
    splits_root: Optional[Path] = None,
    participants: Iterable[str] = ("p01", "p02", "p03", "p04", "p05"),
    stimuli_root: Optional[Path] = None,
    require_existence: bool = False,
) -> pd.DataFrame:
    """Build the union stimulus-pool DataFrame.

    Returns one row per unique image_id across all participants, sorted by
    image_id. Columns:

      image_id, participant_pool, union_index
      [, image_path]   only if ``stimuli_root`` is given
    """
    splits_by_participant: Dict[str, dict] = {
        p: load_all_splits(p, splits_root=splits_root) for p in participants
    }
    pool_by_p = collect_pool_ids(splits_by_participant)

    union_ids, _indices = union_pool(pool_by_p)

    by_id: Dict[str, List[str]] = {iid: [] for iid in union_ids}
    for participant, ids in pool_by_p.items():
        for iid in ids:
            by_id[iid].append(participant)

    rows = []
    missing: List[str] = []
    for idx, iid in enumerate(union_ids):
        row = {
            "image_id":         iid,
            "participant_pool": ",".join(by_id[iid]),
            "union_index":      idx,
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
    bp.add_argument("--splits-root", type=Path, default=None,
                    help="Override resources/splits root.")
    bp.add_argument("--output", type=Path, required=True,
                    help="Output CSV path.")
    bp.add_argument(
        "--participants", nargs="+",
        default=["p01", "p02", "p03", "p04", "p05"],
        help="Participants to include (default: p01..p05).",
    )
    bp.add_argument("--stimuli-root", type=Path, default=None,
                    help="Path to the directory containing "
                         "deepvision_shared/ and deepvision_unique_sub-XX/. "
                         "If given, an 'image_path' column is added to the "
                         "output and existence is verified.")
    bp.add_argument("--no-require-existence", action="store_true",
                    help="Don't fail if some paths don't exist on disk "
                         "(useful when generating the CSV before rsync).")

    args = ap.parse_args(argv)

    if args.cmd == "build-pool":
        df = build_pool_csv(
            splits_root=args.splits_root,
            participants=tuple(args.participants),
            stimuli_root=args.stimuli_root,
            require_existence=(args.stimuli_root is not None and not args.no_require_existence),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows to {args.output}")
        for p in args.participants:
            n = (df["participant_pool"].str.contains(p)).sum()
            print(f"  {p}: {n} images in pool")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
