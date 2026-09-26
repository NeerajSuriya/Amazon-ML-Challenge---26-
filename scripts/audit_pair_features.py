#!/usr/bin/env python3
"""Audit baseline pair features and default-union model errors.

The audit uses the existing nested S1 split. Development rows are used for
feature/coefficient diagnostics; held-out validation errors are emitted only as
post-hoc diagnostics and never drive feature or threshold selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_blocking_features import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    _feature_rows,
    _read,
    _prediction_map,
    fit_matching_model,
    link_counts,
    score_features,
    select_threshold,
    threshold_report,
)
from src.blocking import BlockingConfig, CandidateBlocker  # noqa: E402
from src.feature_experiments import (  # noqa: E402
    coefficient_diagnostics,
    diagnose_features,
    feature_correlations,
)
from src.pair_features import (  # noqa: E402
    CANDIDATE_ID,
    SOURCE1_ID,
    PairFeatureGenerator,
    build_candidate_pairs_from_ground_truth,
    split_ground_truth_by_s1,
)


def _add_record_context(frame: pd.DataFrame, source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame) -> pd.DataFrame:
    records = pd.concat([source1, source2, source3], ignore_index=True).copy()
    records["entity_id"] = records["entity_id"].astype(str).str.strip()
    record_by_id = records.set_index("entity_id", drop=False)
    output = frame.copy()
    fields = ("name_norm", "address_norm", "country_norm")
    for prefix, id_column in (("s1", SOURCE1_ID), ("candidate", CANDIDATE_ID)):
        for field in fields:
            output[f"{prefix}_{field}"] = [
                record_by_id.loc[str(entity_id).strip(), field]
                for entity_id in output[id_column]
            ]
    return output


def _error_rows(scored: pd.DataFrame, threshold: float, *, error_type: str) -> pd.DataFrame:
    if error_type == "model_false_negative":
        mask = (scored["label"] == 1) & (scored["score"] < threshold)
    elif error_type == "false_positive":
        mask = (scored["label"] == 0) & (scored["score"] >= threshold)
    else:
        raise ValueError(f"Unknown error type: {error_type}")
    errors = scored.loc[mask].copy()
    errors["error_type"] = error_type
    errors["selected_threshold"] = threshold
    return errors.sort_values(
        ["score", SOURCE1_ID, CANDIDATE_ID], ascending=[False, True, True], kind="stable"
    ).reset_index(drop=True)


def _error_summary(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame()
    summary = errors.copy()
    summary["name_missing_s1"] = summary["s1_name_norm"].fillna("").eq("")
    summary["name_missing_candidate"] = summary["candidate_name_norm"].fillna("").eq("")
    summary["address_missing_s1"] = summary["s1_address_norm"].fillna("").eq("")
    summary["address_missing_candidate"] = summary["candidate_address_norm"].fillna("").eq("")
    summary["country_missing_either"] = (
        summary["s1_country_norm"].fillna("").eq("")
        | summary["candidate_country_norm"].fillna("").eq("")
    )
    summary["score_band"] = pd.cut(
        summary["score"],
        bins=[-float("inf"), 0.25, 0.5, 0.75, 0.9, 0.95, float("inf")],
        labels=["<0.25", "0.25-0.50", "0.50-0.75", "0.75-0.90", "0.90-0.95", ">=0.95"],
        right=False,
    )
    dimensions = [
        "error_type",
        "candidate_source",
        "score_band",
        "name_missing_s1",
        "name_missing_candidate",
        "address_missing_s1",
        "address_missing_candidate",
        "country_missing_either",
        "name_exact",
        "address_exact",
    ]
    rows = []
    for dimension in dimensions:
        counts = summary.groupby(dimension, dropna=False, observed=False).size()
        for value, count in counts.items():
            rows.append({"dimension": dimension, "value": str(value), "count": int(count)})
    return pd.DataFrame(rows)


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    source1 = _read(args.processed_root / args.source1_name)
    source2 = _read(args.processed_root / args.source2_name)
    source3 = _read(args.processed_root / args.source3_name)
    ground_truth = _read(args.ground_truth)
    train_truth, validation_truth = split_ground_truth_by_s1(
        ground_truth, validation_fraction=args.validation_fraction, seed=args.seed
    )
    model_truth, development_truth = split_ground_truth_by_s1(
        train_truth, validation_fraction=args.development_fraction, seed=args.seed + 1
    )

    blocker = CandidateBlocker(BlockingConfig()).fit(source1, source2, source3)
    candidates = blocker.generate()
    generator = PairFeatureGenerator().fit(source1, source2, source3)
    all_features = _feature_rows(generator, candidates, ground_truth)
    model_features = all_features.loc[
        all_features[SOURCE1_ID].isin(set(model_truth[SOURCE1_ID].astype(str)))
    ].copy()
    development_features = all_features.loc[
        all_features[SOURCE1_ID].isin(set(development_truth[SOURCE1_ID].astype(str)))
    ].copy()
    validation_features = all_features.loc[
        all_features[SOURCE1_ID].isin(set(validation_truth[SOURCE1_ID].astype(str)))
    ].copy()

    provisional = fit_matching_model(model_features, generator.feature_names, args.seed)
    development_scored = score_features(provisional, development_features, generator.feature_names)
    development_report = threshold_report(
        development_scored,
        development_truth,
        DEFAULT_THRESHOLDS,
        stage="development",
        configuration="default_union",
    )
    threshold = select_threshold(development_report)

    final_model = fit_matching_model(all_features.loc[
        all_features[SOURCE1_ID].isin(set(train_truth[SOURCE1_ID].astype(str)))
    ].copy(), generator.feature_names, args.seed)
    validation_scored = score_features(final_model, validation_features, generator.feature_names)
    validation_report = threshold_report(
        validation_scored,
        validation_truth,
        DEFAULT_THRESHOLDS,
        stage="validation",
        configuration="default_union",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnose_features(development_features, generator.feature_names).to_csv(
        output_dir / "feature_diagnostics.tsv", sep="\t", index=False
    )
    feature_correlations(development_features, generator.feature_names).to_csv(
        output_dir / "feature_correlations.tsv", sep="\t", index=False
    )
    coefficient_diagnostics(final_model, generator.feature_names).to_csv(
        output_dir / "model_coefficients.tsv", sep="\t", index=False
    )
    development_report.to_csv(output_dir / "threshold_development.tsv", sep="\t", index=False)
    validation_report.to_csv(output_dir / "threshold_validation.tsv", sep="\t", index=False)

    validation_scored = _add_record_context(validation_scored, source1, source2, source3)
    false_negatives = _error_rows(validation_scored, threshold, error_type="model_false_negative")
    false_positives = _error_rows(validation_scored, threshold, error_type="false_positive")
    false_negatives.to_csv(output_dir / "validation_model_false_negatives.tsv", sep="\t", index=False)
    false_positives.head(args.top_errors).to_csv(
        output_dir / "validation_false_positives_high_confidence.tsv", sep="\t", index=False
    )
    near_threshold = false_positives.loc[
        (false_positives["score"] - threshold).abs() <= args.near_threshold_window
    ].copy()
    near_threshold.to_csv(output_dir / "validation_false_positives_near_threshold.tsv", sep="\t", index=False)
    all_errors = pd.concat([false_negatives, false_positives], ignore_index=True)
    _error_summary(all_errors).to_csv(output_dir / "validation_error_summary.tsv", sep="\t", index=False)

    validation_counts = link_counts(
        validation_scored, threshold, validation_truth
    )
    summary = {
        "candidate_count": int(len(candidates)),
        "feature_count": len(generator.feature_names),
        "feature_names": generator.feature_names,
        "development_threshold": threshold,
        "development_selected_row": development_report.loc[
            development_report["threshold"] == threshold
        ].iloc[0].to_dict(),
        "validation_selected_row": validation_report.loc[
            validation_report["threshold"] == threshold
        ].iloc[0].to_dict(),
        "validation_model_false_negatives": int(len(false_negatives)),
        "validation_false_positives": int(len(false_positives)),
        "validation_high_confidence_false_positives_emitted": int(min(args.top_errors, len(false_positives))),
        "validation_near_threshold_false_positives": int(len(near_threshold)),
        "selection_uses_validation_labels": False,
        "validation_link_counts": validation_counts,
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "processed" / "sample")
    parser.add_argument("--source1-name", default="train_source1_sample.tsv")
    parser.add_argument("--source2-name", default="train_source2_sample.tsv")
    parser.add_argument("--source3-name", default="train_source3_sample.tsv")
    parser.add_argument("--ground-truth", type=Path, default=PROJECT_ROOT / "dataset" / "sample" / "train_ground_truth_sample.tsv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "pair_feature_audit")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--development-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-errors", type=int, default=25)
    parser.add_argument("--near-threshold-window", type=float, default=0.05)
    return parser


def main() -> None:
    summary = run_audit(_parser().parse_args())
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
