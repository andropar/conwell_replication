"""Stimulus pool helpers.

Compute the union of per-subject min_nn pools to drive feature extraction.
Each participant has a 5833-image pool defined by the union of train+test ids
across all 13 splits. The pools differ per subject: only ~1121 images overlap
between participants. The union over all 5 participants is roughly 24k images.

CLI usage::

    python -m conwell_replication.data.stimuli build-pool \\
        --splits-root resources/splits \\
        --output      features/stimulus_pool.csv

The output CSV has columns:

    image_id   participant_pool   union_index

where ``participant_pool`` is a comma-separated list of participant codes
that include this image in their pool, and ``union_index`` is the row
position into the union list (used by feature extraction to index the
.h5 array per subject).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .splits import collect_pool_ids, load_all_splits, union_pool


def build_pool_csv(
    splits_root: Optional[Path] = None,
    participants: Iterable[str] = ("p01", "p02", "p03", "p04", "p05"),
) -> pd.DataFrame:
    """Build the union stimulus-pool DataFrame.

    Returns a DataFrame with one row per unique image_id across all
    participants, sorted by image_id.
    """
    splits_by_participant: Dict[str, dict] = {
        p: load_all_splits(p, splits_root=splits_root) for p in participants
    }
    pool_by_p = collect_pool_ids(splits_by_participant)

    union_ids, _indices = union_pool(pool_by_p)

    # Reverse map: image_id -> participants that include it
    by_id: Dict[str, List[str]] = {iid: [] for iid in union_ids}
    for participant, ids in pool_by_p.items():
        for iid in ids:
            by_id[iid].append(participant)

    rows = []
    for idx, iid in enumerate(union_ids):
        rows.append(
            {
                "image_id": iid,
                "participant_pool": ",".join(by_id[iid]),
                "union_index": idx,
            }
        )
    return pd.DataFrame(rows)


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

    args = ap.parse_args(argv)

    if args.cmd == "build-pool":
        df = build_pool_csv(
            splits_root=args.splits_root,
            participants=tuple(args.participants),
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
