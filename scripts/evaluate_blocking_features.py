#!/usr/bin/env python3
"""Evaluate blocked candidates through pair features and entity-level macro F0.5.

This script deliberately evaluates only candidates emitted by ``CandidateBlocker``.
It does not construct a Cartesian pair table, and the ground truth is used only
for diagnostics, labels, development threshold selection, and evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import resource
import sys
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import (  # noqa: E402
    CANDIDATE_ID,
    DEFAULT_RULE_NAMES,
    SOURCE1_ID,
    BlockingConfig,
    CandidateBlocker,
    evaluate_candidate_pairs,
)
from src.matching import LogisticMatcher, Matcher  # noqa: E402
from src.metrics import (  # noqa: E402
    macro_f05,
    predictions_from_scored_pairs,
)
from src.pair_features import (  # noqa: E402
    PairFeatureGenerator,
    attach_ground_truth_labels,
    build_candidate_pairs_from_ground_truth,
    split_ground_truth_by_s1,
)

DEFAULT_THRESHOLDS = tuple(np.linspace(0.0, 1.0, 21))


@dataclass(frozen=True)
class ConfigurationSpec:
    name: str
    config: BlockingConfig
    rules: tuple[str, ...]


class ConstantScoreModel:
    """Fallback model for empty or single-class training data."""

    class_weight = None
    solver = "constant"
    random_state = None
    max_iter = 0

    def __init__(self, score: float):
        self.score = float(score)

    def predict_scores(self, count: int) -> np.ndarray:
        return np.full(count, self.score, dtype=float)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _path_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    explicit = [args.source1, args.source2, args.source3]
    if any(path is not None for path in explicit):
        if not all(path is not None for path in explicit):
            parser.error("--source1, --source2, and --source3 must be supplied together")
        return tuple(path for path in explicit)  # type: ignore[return-value]
    return (
        args.processed_root / args.source1_name,
        args.processed_root / args.source2_name,
        args.processed_root / args.source3_name,
    )


def configuration_specs(base_config: BlockingConfig) -> tuple[ConfigurationSpec, ...]:
    """Return the source-only configurations used in this experiment."""
    exact = ("country_name_exact", "country_address_exact")
    return (
        ConfigurationSpec(
            "strict_overlap_2",
            BlockingConfig(max_exact_bucket_size=base_config.max_exact_bucket_size),
            exact + ("name_token_overlap_2", "address_token_overlap_2", "name_address_token"),
        ),
        ConfigurationSpec(
            "name_cap_25_address_cap_25",
            BlockingConfig(
                name_token_max_frequency=25,
                address_token_max_frequency=25,
                max_exact_bucket_size=base_config.max_exact_bucket_size,
            ),
            DEFAULT_RULE_NAMES,
        ),
        ConfigurationSpec(
            "address_cap_25",
            BlockingConfig(
                address_token_max_frequency=25,
                max_exact_bucket_size=base_config.max_exact_bucket_size,
            ),
            DEFAULT_RULE_NAMES,
        ),
        ConfigurationSpec(
            "name_cap_25",
            BlockingConfig(
                name_token_max_frequency=25,
                max_exact_bucket_size=base_config.max_exact_bucket_size,
            ),
            DEFAULT_RULE_NAMES,
        ),
        ConfigurationSpec("default_union", base_config, DEFAULT_RULE_NAMES),
    )


def _truth_pairs(ground_truth: pd.DataFrame) -> set[tuple[str, str]]:
    positive = build_candidate_pairs_from_ground_truth(ground_truth)
    if positive.empty:
        return set()
    return set(zip(positive[SOURCE1_ID], positive[CANDIDATE_ID]))


def _candidate_rows_for_s1(candidates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Keep only candidates belonging to the supplied S1 partition."""
    ids = set(truth[SOURCE1_ID].astype(str))
    return candidates.loc[candidates[SOURCE1_ID].isin(ids)].copy()


def _feature_rows(
    generator: PairFeatureGenerator,
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    """Transform blocked candidates and attach labels for one S1 partition."""
    features = generator.transform(candidates)
    labels = attach_ground_truth_labels(candidates, truth)
    if len(features) != len(labels):
        raise RuntimeError("Feature and label row counts differ")
    features["label"] = labels["label"].to_numpy(dtype=int)
    return features


def fit_matching_model(features: pd.DataFrame, feature_names: Sequence[str], seed: int):
    """Fit balanced logistic regression, with a deterministic single-class fallback."""
    labels = features["label"].to_numpy(dtype=int)
    if len(labels) == 0:
        return ConstantScoreModel(0.0)
    unique = np.unique(labels)
    if len(unique) == 1:
        return ConstantScoreModel(float(unique[0]))

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="liblinear",
    )
    model.fit(features[list(feature_names)].to_numpy(dtype=float), labels)
    return model


def score_features(model, features: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Return a copy of features with a positive-class score column."""
    scored = features.copy()
    if scored.empty:
        scored["score"] = pd.Series(dtype=float)
    elif isinstance(model, ConstantScoreModel):
        scored["score"] = model.predict_scores(len(scored))
    else:
        scored["score"] = model.predict_proba(
            scored[list(feature_names)].to_numpy(dtype=float)
        )[:, 1]
    return scored


def fit_matcher(
    features: pd.DataFrame,
    feature_names: Sequence[str],
    seed: int,
    matcher_factory=LogisticMatcher,
) -> Matcher:
    """Fit an injected matcher while preserving selected-column isolation."""
    try:
        matcher = matcher_factory(seed=seed)
    except TypeError:
        matcher = matcher_factory()
    return matcher.fit(features, feature_names)


def score_matcher(
    matcher: Matcher,
    features: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Score rows through the replaceable matcher contract."""
    scored = features.copy()
    scores = matcher.predict_proba(scored, feature_names)
    if len(scores) != len(scored):
        raise ValueError("Matcher returned a score count different from feature rows")
    scored["score"] = scores
    return scored


def _prediction_map(scored: pd.DataFrame, threshold: float, truth: pd.DataFrame) -> dict[str, list[str]]:
    predictions = predictions_from_scored_pairs(scored, threshold, score_col="score")
    for s1_id in truth[SOURCE1_ID].astype(str):
        predictions.setdefault(s1_id, [])
    return predictions


def link_counts(
    scored: pd.DataFrame,
    threshold: float,
    truth: pd.DataFrame,
) -> dict[str, float | int]:
    """Separate blocked-out truth links from model errors among candidates."""
    candidate_pairs = set(zip(scored[SOURCE1_ID], scored[CANDIDATE_ID]))
    truth_pairs = _truth_pairs(truth)
    selected = set(
        zip(
            scored.loc[scored["score"] >= threshold, SOURCE1_ID],
            scored.loc[scored["score"] >= threshold, CANDIDATE_ID],
        )
    )
    available_truth = candidate_pairs & truth_pairs
    blocked_out = truth_pairs - candidate_pairs
    true_positive = selected & truth_pairs
    false_positive = selected - truth_pairs
    model_false_negative = available_truth - selected
    total_false_negative = truth_pairs - true_positive
    precision = len(true_positive) / len(selected) if selected else 0.0
    recall = len(true_positive) / len(truth_pairs) if truth_pairs else 0.0
    return {
        "candidate_true_links": len(available_truth),
        "candidate_loss_links": len(blocked_out),
        "true_positive_links": len(true_positive),
        "false_positive_links": len(false_positive),
        "model_fn_among_candidates": len(model_false_negative),
        "false_negative_links": len(total_false_negative),
        "predicted_links": len(selected),
        "link_precision": precision,
        "link_recall": recall,
    }


def threshold_report(
    scored: pd.DataFrame,
    truth: pd.DataFrame,
    thresholds: Iterable[float],
    *,
    stage: str,
    configuration: str,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        counts = link_counts(scored, float(threshold), truth)
        predictions = _prediction_map(scored, float(threshold), truth)
        rows.append(
            {
                "configuration": configuration,
                "stage": stage,
                "threshold": float(threshold),
                "precision": counts["link_precision"],
                "recall": counts["link_recall"],
                "macro_f0_5": macro_f05(predictions, truth),
                **counts,
            }
        )
    return pd.DataFrame(rows)


def select_threshold(development_report: pd.DataFrame) -> float:
    """Select only from development results, preferring the higher threshold on ties."""
    if development_report.empty:
        return 0.5
    ranked = development_report.sort_values(
        ["macro_f0_5", "threshold"], ascending=[False, False], kind="stable"
    )
    return float(ranked.iloc[0]["threshold"])


def _summary_row(
    spec: ConfigurationSpec,
    blocker: CandidateBlocker,
    candidates: pd.DataFrame,
    blocking_metrics: dict[str, float | int],
    validation_scored: pd.DataFrame,
    validation_truth: pd.DataFrame,
    selected_threshold: float,
    development_report: pd.DataFrame,
    feature_names: Sequence[str],
    elapsed_seconds: float,
) -> dict[str, object]:
    counts = link_counts(validation_scored, selected_threshold, validation_truth)
    predictions = _prediction_map(validation_scored, selected_threshold, validation_truth)
    selected_development = development_report.loc[
        development_report["threshold"] == selected_threshold
    ]
    development_macro = (
        float(selected_development.iloc[0]["macro_f0_5"])
        if not selected_development.empty
        else 0.0
    )
    return {
        "configuration": spec.name,
        "candidates": blocking_metrics["candidate_count"],
        "avg/S1": blocking_metrics["mean_candidates_per_s1"],
        "median/S1": blocking_metrics["median_candidates_per_s1"],
        "p95/S1": blocking_metrics["p95_candidates_per_s1"],
        "p99/S1": blocking_metrics["p99_candidates_per_s1"],
        "max/S1": blocking_metrics["maximum_candidates_per_s1"],
        "S1 >100 candidates": blocking_metrics["s1_with_gt_100_candidates"],
        "positive_candidate_pairs": blocking_metrics["true_matches_retrieved"],
        "negative_candidate_pairs": blocking_metrics["candidate_count"]
        - blocking_metrics["true_matches_retrieved"],
        "blocking_recall": blocking_metrics["blocking_recall"],
        "S2_blocking_recall": blocking_metrics["s2_recall"],
        "S3_blocking_recall": blocking_metrics["s3_recall"],
        "blocking_missed_links": blocking_metrics["true_match_count"]
        - blocking_metrics["true_matches_retrieved"],
        "feature_rows": len(candidates),
        "feature_columns": len(feature_names),
        "feature_shape": f"({len(candidates)}, {len(feature_names)})",
        "threshold": selected_threshold,
        "development_macro_f0_5": development_macro,
        "precision": counts["link_precision"],
        "recall": counts["link_recall"],
        "macro_f0_5": macro_f05(predictions, validation_truth),
        "predicted_links": counts["predicted_links"],
        "true_positive_links": counts["true_positive_links"],
        "false_positive_links": counts["false_positive_links"],
        "model_fn_among_candidates": counts["model_fn_among_candidates"],
        "blocked_out_validation_links": counts["candidate_loss_links"],
        "false_negative_links": counts["false_negative_links"],
        "validation_true_links": len(_truth_pairs(validation_truth)),
        "validation_candidate_true_links": counts["candidate_true_links"],
        "runtime_seconds": elapsed_seconds,
    }


def evaluate_configuration(
    spec: ConfigurationSpec,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    ground_truth: pd.DataFrame,
    train_truth: pd.DataFrame,
    validation_truth: pd.DataFrame,
    model_truth: pd.DataFrame,
    development_truth: pd.DataFrame,
    generator: PairFeatureGenerator,
    *,
    seed: int,
    thresholds: Sequence[float],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    blocker = CandidateBlocker(spec.config).fit(source1, source2, source3)
    candidates = blocker.generate(spec.rules)
    blocking_metrics = evaluate_candidate_pairs(
        candidates, ground_truth, s1_ids=blocker.s1_ids
    )

    all_features = _feature_rows(generator, candidates, ground_truth)
    model_features = all_features.loc[
        all_features[SOURCE1_ID].isin(set(model_truth[SOURCE1_ID].astype(str)))
    ].copy()
    development_features = all_features.loc[
        all_features[SOURCE1_ID].isin(set(development_truth[SOURCE1_ID].astype(str)))
    ].copy()
    train_model = fit_matching_model(model_features, generator.feature_names, seed)
    development_scored = score_features(
        train_model, development_features, generator.feature_names
    )
    development_report = threshold_report(
        development_scored,
        development_truth,
        thresholds,
        stage="development",
        configuration=spec.name,
    )
    selected_threshold = select_threshold(development_report)

    # Refit on all outer-training S1s after threshold selection. Validation S1s
    # are never included in this fit.
    outer_train_features = all_features.loc[
        all_features[SOURCE1_ID].isin(set(train_truth[SOURCE1_ID].astype(str)))
    ].copy()
    final_model = fit_matching_model(outer_train_features, generator.feature_names, seed)
    validation_features = all_features.loc[
        all_features[SOURCE1_ID].isin(set(validation_truth[SOURCE1_ID].astype(str)))
    ].copy()
    validation_scored = score_features(
        final_model, validation_features, generator.feature_names
    )
    validation_report = threshold_report(
        validation_scored,
        validation_truth,
        thresholds,
        stage="validation",
        configuration=spec.name,
    )
    summary = _summary_row(
        spec,
        blocker,
        candidates,
        blocking_metrics,
        validation_scored,
        validation_truth,
        selected_threshold,
        development_report,
        generator.feature_names,
        time.perf_counter() - started,
    )
    # Keep this assertion close to the integration boundary: the feature frame
    # must have exactly one row per blocker identity, never a Cartesian table.
    if len(all_features) != len(candidates):
        raise RuntimeError("Feature matrix row count differs from candidate count")
    return summary, pd.concat([development_report, validation_report], ignore_index=True), candidates


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_paths = _path_args(argparse.ArgumentParser(), args)
    source1, source2, source3 = map(_read, source_paths)
    ground_truth = _read(args.ground_truth)
    train_truth, validation_truth = split_ground_truth_by_s1(
        ground_truth,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    model_truth, development_truth = split_ground_truth_by_s1(
        train_truth,
        validation_fraction=args.development_fraction,
        seed=args.seed + 1,
    )
    base_config = BlockingConfig(
        name_token_max_frequency=args.name_token_max_frequency,
        address_token_max_frequency=args.address_token_max_frequency,
        max_exact_bucket_size=args.max_exact_bucket_size,
    )
    specs = configuration_specs(base_config)
    generator = PairFeatureGenerator().fit(source1, source2, source3)
    summaries = []
    threshold_reports = []
    for spec in specs:
        summary, report, _ = evaluate_configuration(
            spec,
            source1,
            source2,
            source3,
            ground_truth,
            train_truth,
            validation_truth,
            model_truth,
            development_truth,
            generator,
            seed=args.seed,
            thresholds=DEFAULT_THRESHOLDS,
        )
        summaries.append(summary)
        threshold_reports.append(report)
    return pd.DataFrame(summaries), pd.concat(threshold_reports, ignore_index=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "processed" / "sample")
    parser.add_argument("--source1", type=Path)
    parser.add_argument("--source2", type=Path)
    parser.add_argument("--source3", type=Path)
    parser.add_argument("--source1-name", default="train_source1_sample.tsv")
    parser.add_argument("--source2-name", default="train_source2_sample.tsv")
    parser.add_argument("--source3-name", default="train_source3_sample.tsv")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "sample" / "train_ground_truth_sample.tsv",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--development-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name-token-max-frequency", type=int, default=50)
    parser.add_argument("--address-token-max-frequency", type=int, default=50)
    parser.add_argument("--max-exact-bucket-size", type=int, default=500)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--threshold-output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    started = time.perf_counter()
    summaries, thresholds = run_experiment(args)
    print("Blocking → pair features → balanced logistic regression → entity macro F0.5")
    print(
        summaries.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}" if isinstance(value, float) else str(value),
        )
    )
    print("\nThreshold results (development selects; validation is held out)")
    print(thresholds.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summaries.to_csv(args.summary_output, sep="\t", index=False)
    if args.threshold_output:
        args.threshold_output.parent.mkdir(parents=True, exist_ok=True)
        thresholds.to_csv(args.threshold_output, sep="\t", index=False)
    print(f"\nTotal runtime seconds: {time.perf_counter() - started:.3f}")
    print(f"Peak RSS MB: {_rss_mb():.1f}")


if __name__ == "__main__":
    main()
