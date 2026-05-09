#!/usr/bin/env python3
"""Build the OTC brain cache from a startslurm-launched Python job."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_workers() -> int:
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    return max(1, min(4, cpus))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("LAION_FMRI_ROOT", "/raven/ptmp/rothj/laion_fmri")),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            os.environ.get(
                "CONWELL_BRAIN_CACHE",
                "/raven/ptmp/rothj/conwell_replication/brain_cache_otc",
            )
        ),
    )
    parser.add_argument("--roi-root", type=Path, default=None)
    parser.add_argument("--r2-threshold", type=float, default=0.15)
    parser.add_argument("--n-workers", type=int, default=_default_workers())
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--repo",
        type=Path,
        default=repo,
        help="Repository root to add to sys.path. Defaults to this script's repo.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    os.chdir(repo)

    os.environ["LAION_FMRI_ROOT"] = str(args.data_root)
    os.environ["CONWELL_BRAIN_CACHE"] = str(args.cache_root)

    from conwell_replication.data.brain_cache import main as cache_main

    cache_argv = [
        "--data-root",
        str(args.data_root),
        "--cache-root",
        str(args.cache_root),
        "--subjects",
        *args.subjects,
        "--voxel-set",
        "otc",
        "--r2-threshold",
        str(args.r2_threshold),
        "--n-workers",
        str(args.n_workers),
    ]
    roi_root = args.roi_root or (args.data_root / "derivatives" / "rois")
    cache_argv.extend(["--roi-root", str(roi_root)])
    if args.force:
        cache_argv.append("--force")
    return int(cache_main(cache_argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
