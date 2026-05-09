#!/usr/bin/env python3
"""Submit OTC cache jobs through /u/rothj/laion_natural startslurm."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
    )
    parser.add_argument(
        "--startslurm",
        type=Path,
        default=Path("/u/rothj/laion_natural/scripts/start_as_slurm_job.py"),
    )
    parser.add_argument("--conda-env", default="conwell_replication")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--mem", type=int, default=119000)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/raven/ptmp/rothj/laion_fmri"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/raven/ptmp/rothj/conwell_replication/brain_cache_otc"),
    )
    parser.add_argument("--roi-root", type=Path, default=None)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--r2-threshold", type=float, default=0.15)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--runner",
        type=Path,
        default=repo / "scripts" / "archive" / "cluster" / "run_cache_brain_otc_raven.py",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for subject in args.subjects:
        cmd = [
            sys.executable,
            str(args.startslurm),
            "--conda-env",
            args.conda_env,
            "--n-jobs",
            str(args.n_jobs),
            "--mem",
            str(args.mem),
            str(args.runner),
            "--subjects",
            subject,
            "--data-root",
            str(args.data_root),
            "--cache-root",
            str(args.cache_root),
            "--n-workers",
            str(args.n_workers),
            "--r2-threshold",
            str(args.r2_threshold),
        ]
        if args.roi_root is not None:
            cmd.extend(["--roi-root", str(args.roi_root)])
        if args.force:
            cmd.append("--force")
        _run(cmd, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
