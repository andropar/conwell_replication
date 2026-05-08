#!/usr/bin/env python3
"""Build the model manifest for the DeepNSD-controlled replication evals.

The feature directory may contain many extracted models. For the replication
figures requested here, the evals only need models that participate in the
DeepNSD model_contrasts.csv controlled comparisons for Figs. 2-4.
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = Path(os.environ.get("CONWELL_FEATURES", REPO_ROOT / "features"))
DEFAULT_MODEL_CONTRASTS = REPO_ROOT / "resources" / "model_contrasts.csv"
DEFAULT_OUT = Path(
    os.environ.get(
        "CONWELL_EVAL_MODEL_MANIFEST",
        REPO_ROOT / "results" / "eval_model_manifest_replication.txt",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model-contrasts", type=Path, default=DEFAULT_MODEL_CONTRASTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="CSV audit path. Default: OUT with .audit.csv suffix.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any needed model has no complete .h5 feature file.",
    )
    return parser.parse_args()


def unique_ordered(values):
    seen = set()
    out = []
    for value in values:
        if pd.isna(value):
            continue
        value = str(value)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def feature_stem_candidates(model_string: str) -> list[str]:
    raw = str(model_string)
    candidates = [raw, raw.replace("/", "-")]
    for candidate in list(candidates):
        if candidate.endswith("_ipcl"):
            candidates.append(candidate[: -len("_ipcl")])
    return unique_ordered(candidates)


def result_model_name(feature_stem: str) -> str:
    return feature_stem.replace("-", "/")


def needed_rows(contrasts: pd.DataFrame) -> list[tuple[int, str, pd.Series]]:
    comparisons = [
        (
            "fig2_architecture_cnn_transformer",
            contrasts["compare_architecture"].isin(["Convolutional", "Transformer"]),
        ),
        ("fig3_taskonomy", contrasts["compare_goal_taskonomy_tasks"].notna()),
        ("fig3_selfsupervised_resnet50", contrasts["compare_goal_selfsupervised"].notna()),
        ("fig3_slip", contrasts["compare_goal_slip"].notna()),
        ("fig4_imagenet_size", contrasts["compare_diet_imagenetsize"].notna()),
        ("fig4_ipcl", contrasts["compare_diet_ipcl"].notna()),
    ]
    rows: list[tuple[int, str, pd.Series]] = []
    for reason, mask in comparisons:
        for idx, row in contrasts.loc[mask].iterrows():
            rows.append((int(idx), reason, row))

    # Fig. 3B uses a supervised ResNet50 reference in addition to SSL models.
    ref = contrasts.loc[contrasts["model_string"] == "resnet50_classification"]
    if not ref.empty:
        rows.append((int(ref.index[0]), "fig3_supervised_resnet50_reference", ref.iloc[0]))
    return rows


def first_existing_candidate(model_string: str, available_stems: set[str]) -> tuple[str, bool]:
    candidates = feature_stem_candidates(model_string)
    for candidate in candidates:
        if candidate in available_stems:
            return candidate, True
    return candidates[0], False


def build_manifest(
    features: Path,
    contrasts_path: Path,
) -> tuple[list[str], pd.DataFrame]:
    contrasts = pd.read_csv(contrasts_path)
    required = {"model_string", "model_display_name"}
    missing_cols = required.difference(contrasts.columns)
    if missing_cols:
        raise ValueError(f"{contrasts_path} lacks required columns: {sorted(missing_cols)}")

    feature_files = {path.stem: path.name for path in sorted(features.glob("*.h5"))}
    available_stems = set(feature_files)

    needed_by_stem: OrderedDict[str, dict] = OrderedDict()
    reasons_by_stem: dict[str, list[str]] = defaultdict(list)
    source_strings_by_stem: dict[str, list[str]] = defaultdict(list)

    for source_order, reason, row in needed_rows(contrasts):
        stem, exists = first_existing_candidate(row["model_string"], available_stems)
        if stem not in needed_by_stem:
            needed_by_stem[stem] = {
                "source_order": source_order,
                "feature_stem": stem,
                "feature_file": feature_files.get(stem, f"{stem}.h5"),
                "result_model": result_model_name(stem),
                "feature_exists": bool(exists),
                "in_manifest": bool(exists),
                "status": "included" if exists else "missing_needed",
                "deepnsd_model_string": row["model_string"],
                "model_display_name": row.get("model_display_name", stem),
                "model_class": row.get("model_class"),
                "architecture": row.get("architecture"),
                "train_type": row.get("train_type"),
                "train_data": row.get("train_data"),
                "compare_architecture": row.get("compare_architecture"),
                "compare_goal_taskonomy_tasks": row.get("compare_goal_taskonomy_tasks"),
                "compare_goal_taskonomy_cluster": row.get("compare_goal_taskonomy_cluster"),
                "compare_goal_selfsupervised": row.get("compare_goal_selfsupervised"),
                "compare_goal_contrastive": row.get("compare_goal_contrastive"),
                "compare_goal_slip": row.get("compare_goal_slip"),
                "compare_diet_imagenetsize": row.get("compare_diet_imagenetsize"),
                "compare_diet_ipcl": row.get("compare_diet_ipcl"),
            }
        reasons_by_stem[stem].append(reason)
        source_strings_by_stem[stem].append(str(row["model_string"]))

    for stem, info in needed_by_stem.items():
        info["needed_for"] = ";".join(unique_ordered(reasons_by_stem[stem]))
        info["all_deepnsd_model_strings"] = ";".join(unique_ordered(source_strings_by_stem[stem]))

    audit_rows = list(needed_by_stem.values())
    needed_stems = set(needed_by_stem)
    for stem, file_name in sorted(feature_files.items()):
        if stem in needed_stems:
            continue
        audit_rows.append(
            {
                "source_order": pd.NA,
                "feature_stem": stem,
                "feature_file": file_name,
                "result_model": result_model_name(stem),
                "feature_exists": True,
                "in_manifest": False,
                "status": "available_unneeded",
                "needed_for": "",
                "deepnsd_model_string": "",
                "all_deepnsd_model_strings": "",
                "model_display_name": "",
            }
        )

    audit = pd.DataFrame(audit_rows)
    included = [
        info["feature_file"]
        for info in needed_by_stem.values()
        if info["feature_exists"]
    ]
    return included, audit


def main() -> int:
    args = parse_args()
    audit_path = args.audit
    if audit_path is None:
        audit_path = args.out.with_suffix(".audit.csv")

    included, audit = build_manifest(args.features, args.model_contrasts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(included) + ("\n" if included else ""))
    audit.to_csv(audit_path, index=False)

    n_needed = int((audit["status"] != "available_unneeded").sum())
    n_included = len(included)
    n_missing = int((audit["status"] == "missing_needed").sum())
    n_unneeded = int((audit["status"] == "available_unneeded").sum())

    print(f"Wrote {n_included} eval models to {args.out}")
    print(f"Wrote audit to {audit_path}")
    print(
        "Replication model audit: "
        f"needed={n_needed}, included={n_included}, "
        f"missing_needed={n_missing}, available_unneeded={n_unneeded}"
    )
    if n_missing:
        missing = audit.loc[audit["status"] == "missing_needed", "feature_file"].tolist()
        print("Missing needed feature files:")
        for item in missing:
            print(f"  - {item}")
        if args.strict:
            return 1
    if not included:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
