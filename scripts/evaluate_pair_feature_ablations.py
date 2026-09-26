#!/usr/bin/env python3
"""Run controlled pair-feature ablations on fixed default-union candidates.

Feature sets are selected from development results only. Every seed gets an
independent deterministic S1-level outer/inner split; held-out validation rows
are used only for reporting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_blocking_features import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    _feature_rows,
    _read,
    fit_matcher,
    link_counts,
    score_matcher,
    select_threshold,
    threshold_report,
)
from src.blocking import BlockingConfig, CandidateBlocker, evaluate_candidate_pairs  # noqa: E402
from src.feature_experiments import (  # noqa: E402
    aggregate_results,
    default_feature_sets,
    paired_differences,
    resolve_feature_set,
)
from src.pair_features import (  # noqa: E402
    CANDIDATE_ID,
    PairFeatureGenerator,
    SOURCE1_ID,
    build_candidate_pairs_from_ground_truth,
    split_ground_truth_by_s1,
)

ROBUSTNESS_FEATURE_SET_NAMES = (
    "baseline",
    "baseline_plus_address",
    "baseline_plus_cross_field",
    "baseline_plus_address_cross_field",
    "baseline_plus_name_address_cross",
)


def _feature_sets(args: argparse.Namespace):
    specification = getattr(args, "feature_sets", None)
    if specification:
        names = tuple(value.strip() for value in specification.split(",") if value.strip())
        if not names:
            raise ValueError("--feature-sets must contain at least one feature-set name")
        return tuple(resolve_feature_set(name) for name in names)
    if getattr(args, "robustness", False):
        return tuple(resolve_feature_set(name) for name in ROBUSTNESS_FEATURE_SET_NAMES)
    return default_feature_sets()


def _add_record_context(
    frame: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    records = pd.concat([source1, source2, source3], ignore_index=True).copy()
    records["entity_id"] = records["entity_id"].astype(str).str.strip()
    record_by_id = records.set_index("entity_id", drop=False)
    output = frame.copy()
    for prefix, id_column in (("s1", SOURCE1_ID), ("candidate", CANDIDATE_ID)):
        for field in ("name_norm", "address_norm", "country_norm"):
            output[f"{prefix}_{field}"] = [
                record_by_id.loc[str(entity_id).strip(), field]
                for entity_id in output[id_column]
            ]
    return output


def _error_artifact(
    *,
    feature_set,
    validation_scored: pd.DataFrame,
    validation_truth: pd.DataFrame,
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    threshold: float,
    seed: int,
) -> pd.DataFrame:
    """Build post-hoc errors without altering model or threshold selection."""
    scored = _add_record_context(validation_scored, source1, source2, source3)
    false_positive = scored.loc[
        (scored["label"] == 0) & (scored["score"] >= threshold)
    ].copy()
    false_positive["error_type"] = "false_positive"
    false_positive["selected_threshold"] = threshold
    false_negative = scored.loc[
        (scored["label"] == 1) & (scored["score"] < threshold)
    ].copy()
    false_negative["error_type"] = "model_false_negative"
    false_negative["selected_threshold"] = threshold

    truth_pairs = build_candidate_pairs_from_ground_truth(validation_truth)
    candidate_pairs = set(zip(candidates[SOURCE1_ID], candidates[CANDIDATE_ID]))
    blocked = truth_pairs.loc[
        ~truth_pairs.apply(
            lambda row: (row[SOURCE1_ID], row[CANDIDATE_ID]) in candidate_pairs,
            axis=1,
        )
    ].copy()
    blocked["label"] = 1
    blocked["score"] = float("nan")
    blocked["error_type"] = "blocked_true_link"
    blocked["source1_source"] = "S1"
    blocked["candidate_source"] = blocked[CANDIDATE_ID].astype(str).str[:2]
    blocked["selected_threshold"] = threshold
    blocked = _add_record_context(blocked, source1, source2, source3)

    columns = [
        SOURCE1_ID,
        CANDIDATE_ID,
        "source1_source",
        "candidate_source",
        "error_type",
        "score",
        "selected_threshold",
        "label",
        "s1_name_norm",
        "s1_address_norm",
        "s1_country_norm",
        "candidate_name_norm",
        "candidate_address_norm",
        "candidate_country_norm",
        *feature_set.columns,
    ]
    rows = pd.concat([false_positive, false_negative, blocked], ignore_index=True)
    rows["feature_set"] = feature_set.name
    rows["seed"] = seed
    rows["selection_uses_validation_labels"] = False
    return rows.reindex(columns=[*columns, "feature_set", "seed", "selection_uses_validation_labels"])


def _evaluate_set(
    feature_set,
    *,
    all_features: pd.DataFrame,
    train_truth: pd.DataFrame,
    validation_truth: pd.DataFrame,
    model_truth: pd.DataFrame,
    development_truth: pd.DataFrame,
    seed: int,
    candidate_count: int,
    blocking_metrics: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, float]:
    started = time.perf_counter()
    feature_names = feature_set.columns
    model_ids = set(model_truth[SOURCE1_ID].astype(str))
    development_ids = set(development_truth[SOURCE1_ID].astype(str))
    train_ids = set(train_truth[SOURCE1_ID].astype(str))
    validation_ids = set(validation_truth[SOURCE1_ID].astype(str))

    model_features = all_features.loc[all_features[SOURCE1_ID].isin(model_ids)].copy()
    development_features = all_features.loc[all_features[SOURCE1_ID].isin(development_ids)].copy()
    train_model_features = all_features.loc[all_features[SOURCE1_ID].isin(train_ids)].copy()
    validation_features = all_features.loc[all_features[SOURCE1_ID].isin(validation_ids)].copy()

    provisional = fit_matcher(model_features, feature_names, seed)
    development_scored = score_matcher(provisional, development_features, feature_names)
    development_report = threshold_report(
        development_scored,
        development_truth,
        DEFAULT_THRESHOLDS,
        stage="development",
        configuration=feature_set.name,
    )
    threshold = select_threshold(development_report)

    final_model = fit_matcher(train_model_features, feature_names, seed)
    validation_scored = score_matcher(final_model, validation_features, feature_names)
    validation_report = threshold_report(
        validation_scored,
        validation_truth,
        DEFAULT_THRESHOLDS,
        stage="validation",
        configuration=feature_set.name,
    )
    counts = link_counts(validation_scored, threshold, validation_truth)
    selected_development = development_report.loc[development_report["threshold"] == threshold].iloc[0]
    selected_validation = validation_report.loc[validation_report["threshold"] == threshold].iloc[0]
    row = {
        "feature_set": feature_set.name,
        "feature_names": json.dumps(list(feature_names)),
        "feature_count": feature_set.count,
        "candidate_count": candidate_count,
        "blocking_recall": float(blocking_metrics["blocking_recall"]),
        "s2_blocking_recall": float(blocking_metrics["s2_recall"]),
        "s3_blocking_recall": float(blocking_metrics["s3_recall"]),
        "seed": seed,
        "outer_seed": seed,
        "development_seed": seed + 1,
        "threshold": threshold,
        "development_precision": float(selected_development["precision"]),
        "development_recall": float(selected_development["recall"]),
        "development_macro_f0_5": float(selected_development["macro_f0_5"]),
        "validation_precision": float(selected_validation["precision"]),
        "validation_recall": float(selected_validation["recall"]),
        "validation_macro_f0_5": float(selected_validation["macro_f0_5"]),
        "predicted_links": counts["predicted_links"],
        "true_positive_links": counts["true_positive_links"],
        "false_positive_links": counts["false_positive_links"],
        "false_negative_links": counts["false_negative_links"],
        "model_fn_among_candidates": counts["model_fn_among_candidates"],
        "blocked_out_validation_links": counts["candidate_loss_links"],
        "candidate_true_links": counts["candidate_true_links"],
        "runtime_seconds": time.perf_counter() - started,
        "selection_uses_validation_labels": False,
    }
    return (
        row,
        pd.concat([development_report, validation_report], ignore_index=True),
        validation_scored,
        threshold,
    )


def run_ablation(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source1 = _read(args.processed_root / args.source1_name)
    source2 = _read(args.processed_root / args.source2_name)
    source3 = _read(args.processed_root / args.source3_name)
    ground_truth = _read(args.ground_truth)
    blocker = CandidateBlocker(BlockingConfig()).fit(source1, source2, source3)
    candidates = blocker.generate()
    blocking_metrics = evaluate_candidate_pairs(candidates, ground_truth, s1_ids=blocker.s1_ids)

    generator = PairFeatureGenerator(include_experimental=True).fit(source1, source2, source3)
    all_features = _feature_rows(generator, candidates, ground_truth)
    feature_sets = _feature_sets(args)
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")

    rows = []
    threshold_rows = []
    scored_by_key = {}
    for seed in seeds:
        train_truth, validation_truth = split_ground_truth_by_s1(
            ground_truth, validation_fraction=args.validation_fraction, seed=seed
        )
        model_truth, development_truth = split_ground_truth_by_s1(
            train_truth, validation_fraction=args.development_fraction, seed=seed + 1
        )
        for feature_set in feature_sets:
            row, reports, validation_scored, selected_threshold = _evaluate_set(
                feature_set,
                all_features=all_features,
                train_truth=train_truth,
                validation_truth=validation_truth,
                model_truth=model_truth,
                development_truth=development_truth,
                seed=seed,
                candidate_count=len(candidates),
                blocking_metrics=blocking_metrics,
            )
            rows.append(row)
            scored_by_key[(seed, feature_set.name)] = (
                feature_set,
                validation_scored,
                validation_truth,
                selected_threshold,
            )
            reports = reports.copy()
            reports["seed"] = seed
            reports["outer_seed"] = seed
            reports["development_seed"] = seed + 1
            threshold_rows.append(reports)

    per_seed = pd.DataFrame(rows)
    aggregate = aggregate_results(per_seed, group_columns=("feature_set",))
    aggregate["seeds_requested"] = len(seeds)
    aggregate["seed_values"] = ",".join(str(seed) for seed in seeds)
    aggregate["selection_uses_validation_labels"] = False
    threshold_report_frame = pd.concat(threshold_rows, ignore_index=True)
    error_output = getattr(args, "error_output", None)
    if error_output is not None:
        error_feature_set = getattr(args, "error_feature_set", "baseline_plus_address")
        error_seed = getattr(args, "error_seed", None)
        error_seed = seeds[0] if error_seed is None else int(error_seed)
        error_key = (error_seed, error_feature_set)
        if error_key not in scored_by_key:
            raise ValueError(
                f"Error artifact target is not in this run: seed={error_seed}, feature_set={error_feature_set}"
            )
        error_set, error_scored, error_truth, error_threshold = scored_by_key[error_key]
        error_frame = _error_artifact(
            feature_set=error_set,
            validation_scored=error_scored,
            validation_truth=error_truth,
            candidates=candidates,
            source1=source1,
            source2=source2,
            source3=source3,
            threshold=error_threshold,
            seed=error_seed,
        )
        error_output.parent.mkdir(parents=True, exist_ok=True)
        error_frame.to_csv(error_output, sep="\t", index=False)
    summary = pd.DataFrame(
        [{
            "candidate_count": len(candidates),
            "feature_matrix_rows": len(all_features),
            "feature_matrix_columns": len(generator.feature_names),
            "blocking_recall": blocking_metrics["blocking_recall"],
            "s2_blocking_recall": blocking_metrics["s2_recall"],
            "s3_blocking_recall": blocking_metrics["s3_recall"],
            "seeds": ",".join(str(seed) for seed in seeds),
            "selection_uses_validation_labels": False,
        }]
    )
    return per_seed, aggregate, threshold_report_frame, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "processed" / "sample")
    parser.add_argument("--source1-name", default="train_source1_sample.tsv")
    parser.add_argument("--source2-name", default="train_source2_sample.tsv")
    parser.add_argument("--source3-name", default="train_source3_sample.tsv")
    parser.add_argument("--ground-truth", type=Path, default=PROJECT_ROOT / "dataset" / "sample" / "train_ground_truth_sample.tsv")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--development-fraction", type=float, default=0.2)
    parser.add_argument("--seeds", default="42", help="Comma-separated S1 split seeds, e.g. 42,123,2024")
    parser.add_argument("--robustness", action="store_true", help="Run the focused five-configuration robustness set")
    parser.add_argument("--feature-sets", help="Comma-separated named feature sets overriding the default ablation list")
    parser.add_argument("--per-seed-output", type=Path, default=PROJECT_ROOT / "artifacts" / "pair_feature_ablations" / "per_seed.tsv")
    parser.add_argument("--aggregate-output", type=Path, default=PROJECT_ROOT / "artifacts" / "pair_feature_ablations" / "aggregate.tsv")
    parser.add_argument("--threshold-output", type=Path, default=PROJECT_ROOT / "artifacts" / "pair_feature_ablations" / "thresholds.tsv")
    parser.add_argument("--summary-output", type=Path, default=PROJECT_ROOT / "artifacts" / "pair_feature_ablations" / "summary.tsv")
    parser.add_argument("--paired-output", type=Path)
    parser.add_argument("--paired-summary-output", type=Path)
    parser.add_argument("--error-output", type=Path)
    parser.add_argument("--error-feature-set", default="baseline_plus_address")
    parser.add_argument("--error-seed", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    per_seed, aggregate, thresholds, summary = run_ablation(args)
    paired_per_seed, paired_summary = paired_differences(per_seed)
    output_frames = [
        (per_seed, args.per_seed_output),
        (aggregate, args.aggregate_output),
        (thresholds, args.threshold_output),
        (summary, args.summary_output),
    ]
    if args.paired_output is not None:
        output_frames.append((paired_per_seed, args.paired_output))
    if args.paired_summary_output is not None:
        output_frames.append((paired_summary, args.paired_summary_output))
    for frame, path in output_frames:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
    print("Per-seed results")
    print(per_seed.to_string(index=False))
    print("\nAggregated results")
    print(aggregate.to_string(index=False))
    print("\nPaired differences against baseline")
    print(paired_summary.to_string(index=False))
    print("\nCandidate summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
