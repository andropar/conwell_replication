#!/usr/bin/env python3
"""Run one streamed split-half model batch from a startslurm GPU job."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_workers() -> int:
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    return max(1, min(8, cpus))


def _check_cache(cache_root: Path, voxel_set: str, subjects: list[str]) -> None:
    bad: list[str] = []
    for subject in subjects:
        subject_dir = cache_root / subject
        info_path = subject_dir / "cache_info.json"
        betas_path = subject_dir / "betas.npy"
        image_ids_path = subject_dir / "image_ids.npy"
        if not betas_path.exists() or not image_ids_path.exists() or not info_path.exists():
            bad.append(f"{subject}: missing betas.npy, image_ids.npy, or cache_info.json")
            continue
        info = json.loads(info_path.read_text())
        if info.get("voxel_set") != voxel_set:
            bad.append(
                f"{subject}: cache voxel_set={info.get('voxel_set')!r}, "
                f"expected {voxel_set!r}"
            )
    if bad:
        lines = "\n  - ".join(bad)
        raise SystemExit(
            "Brain cache preflight failed:\n"
            f"  - {lines}\n"
            "Build it first with submit_cache_brain_otc_startslurm.py."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-batch", type=int, required=True)
    parser.add_argument("--models-per-batch", type=int, default=1)
    parser.add_argument(
        "--models-csv",
        type=Path,
        default=repo / "resources" / "conwell_model_list_replication.csv",
    )
    parser.add_argument("--model-list", type=Path, default=None)
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
    parser.add_argument("--voxel-set", default="otc", choices=("hlvis", "visual", "otc"))
    parser.add_argument("--rois", nargs="+", default=None)
    parser.add_argument("--min-roi-voxels", type=int, default=10)
    parser.add_argument("--ncsnr-threshold", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=_default_workers())
    parser.add_argument("--layer-chunk-size", type=int, default=8)
    parser.add_argument("--n-projections", type=int, default=5960)
    parser.add_argument("--srp-seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--include-ood", action="store_true")
    parser.add_argument("--no-crsa", action="store_true")
    parser.add_argument("--no-wrsa", action="store_true")
    parser.add_argument("--no-srpr", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-brain-cache", action="store_true", default=True)
    parser.add_argument("--no-require-brain-cache", dest="require_brain_cache", action="store_false")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("LAION_FMRI_ROOT", "/raven/ptmp/rothj/laion_fmri")),
    )
    parser.add_argument(
        "--brain-cache",
        type=Path,
        default=Path(
            os.environ.get(
                "CONWELL_BRAIN_CACHE",
                "/raven/ptmp/rothj/conwell_replication/brain_cache_otc",
            )
        ),
    )
    parser.add_argument(
        "--image-root-map",
        action="append",
        default=["/ptmp/rothj/=/raven/ptmp/rothj/"],
    )
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
    os.environ["CONWELL_BRAIN_CACHE"] = str(args.brain_cache)
    if args.require_brain_cache:
        _check_cache(args.brain_cache, args.voxel_set, args.subjects)

    from conwell_replication.eval.rsa_splithalf_streamed import main as eval_main

    eval_argv: list[str] = []
    if args.model_list is not None:
        eval_argv.extend(["--model-list", str(args.model_list)])
    else:
        eval_argv.extend(["--models", str(args.models_csv)])
    eval_argv.extend(
        [
            "--pool-csv",
            str(args.pool_csv),
            "--out",
            str(args.out),
            "--subjects",
            *args.subjects,
            "--voxel-set",
            args.voxel_set,
            "--min-roi-voxels",
            str(args.min_roi_voxels),
            "--ncsnr-threshold",
            str(args.ncsnr_threshold),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--layer-chunk-size",
            str(args.layer_chunk_size),
            "--n-projections",
            str(args.n_projections),
            "--srp-seed",
            str(args.srp_seed),
            "--model-batch",
            str(args.model_batch),
            "--models-per-batch",
            str(args.models_per_batch),
            "--gpu",
            str(args.gpu),
        ]
    )
    if args.rois:
        eval_argv.extend(["--rois", *args.rois])
    for root_map in args.image_root_map:
        eval_argv.extend(["--image-root-map", root_map])
    if args.skip_existing:
        eval_argv.append("--skip-existing")
    if args.include_ood:
        eval_argv.append("--include-ood")
    if args.no_crsa:
        eval_argv.append("--no-crsa")
    if args.no_wrsa:
        eval_argv.append("--no-wrsa")
    if args.no_srpr:
        eval_argv.append("--no-srpr")
    if args.dry_run:
        eval_argv.append("--dry-run")
    return int(eval_main(eval_argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
