#!/usr/bin/env python3
"""Submit streamed split-half model batches through startslurm."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _count_models(models_csv: Path | None, model_list: Path | None) -> int:
    if model_list is not None:
        return sum(
            1
            for line in model_list.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if models_csv is None:
        raise ValueError("Either models_csv or model_list is required")
    with models_csv.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _run(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--startslurm",
        type=Path,
        default=Path("/u/rothj/laion_natural/scripts/start_as_slurm_job.py"),
    )
    parser.add_argument("--conda-env", default="conwell_replication")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--mem", type=int, default=92000)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--dependency-afterok", default=None)
    parser.add_argument("--dependency-afterany", default=None)
    parser.add_argument(
        "--runner",
        type=Path,
        default=repo
        / "scripts"
        / "archive"
        / "cluster"
        / "run_splithalf_streamed_batch_raven.py",
    )
    parser.add_argument(
        "--models-csv",
        type=Path,
        default=repo / "resources" / "conwell_model_list_replication.csv",
    )
    parser.add_argument("--model-list", type=Path, default=None)
    parser.add_argument("--models-per-batch", type=int, default=1)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--stop-batch", type=int, default=None)
    parser.add_argument("--max-submit", type=int, default=None)
    parser.add_argument("--submit-delay", type=float, default=0.0)
    parser.add_argument(
        "--pool-csv",
        type=Path,
        default=repo / "features" / "stimulus_pool.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/raven/ptmp/rothj/conwell_replication/eval_splithalf_streamed_otc"),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
    )
    parser.add_argument(
        "--brain-cache",
        type=Path,
        default=Path("/raven/ptmp/rothj/conwell_replication/brain_cache_otc"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/raven/ptmp/rothj/laion_fmri"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--layer-chunk-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runner-dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.models_per_batch <= 0:
        raise SystemExit("--models-per-batch must be positive")
    n_models = _count_models(args.models_csv, args.model_list)
    n_batches = (n_models + args.models_per_batch - 1) // args.models_per_batch
    stop = args.stop_batch if args.stop_batch is not None else n_batches
    stop = min(stop, n_batches)
    batches = list(range(args.start_batch, stop))
    if args.max_submit is not None:
        batches = batches[: args.max_submit]

    print(
        f"n_models={n_models} models_per_batch={args.models_per_batch} "
        f"n_batches={n_batches} submitting={len(batches)}"
    )

    for batch in batches:
        cmd = [
            sys.executable,
            str(args.startslurm),
            "--conda-env",
            args.conda_env,
            "--gpu",
            "--n-gpus",
            str(args.n_gpus),
            "--n-jobs",
            str(args.n_jobs),
            "--mem",
            str(args.mem),
        ]
        if args.dependency_afterok:
            cmd.extend(["--dependency-afterok", args.dependency_afterok])
        if args.dependency_afterany:
            cmd.extend(["--dependency-afterany", args.dependency_afterany])
        cmd.extend(
            [
                str(args.runner),
                "--model-batch",
                str(batch),
                "--models-per-batch",
                str(args.models_per_batch),
                "--pool-csv",
                str(args.pool_csv),
                "--out",
                str(args.out),
                "--subjects",
                *args.subjects,
                "--brain-cache",
                str(args.brain_cache),
                "--data-root",
                str(args.data_root),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--layer-chunk-size",
                str(args.layer_chunk_size),
            ]
        )
        if args.model_list is not None:
            cmd.extend(["--model-list", str(args.model_list)])
        else:
            cmd.extend(["--models-csv", str(args.models_csv)])
        if args.runner_dry_run:
            cmd.append("--dry-run")
        _run(cmd, args.dry_run)
        if args.submit_delay:
            time.sleep(args.submit_delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
