#!/usr/bin/env python3
"""Evaluate blocking recall, per-S1 tails, and candidate-volume tradeoffs."""
from __future__ import annotations

import argparse
from pathlib import Path
import resource
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import (
    DEFAULT_RULE_NAMES,
    BlockingConfig,
    CandidateBlocker,
    candidate_pairs_by_s1,
    evaluate_candidate_pairs,
    evaluate_rule_contributions,
    summarize_candidate_distribution,
)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _configurations(base_config: BlockingConfig) -> list[tuple[str, BlockingConfig, tuple[str, ...] | None]]:
    """Return source-only configurations chosen before reading evaluation results."""
    default = tuple(DEFAULT_RULE_NAMES)
    exact = ("country_name_exact", "country_address_exact")
    tokens = ("name_token", "address_token", "name_address_token")
    return [
        ("default_union", base_config, default),
        ("tokens_only", BlockingConfig(), tokens),
        ("exact_only", BlockingConfig(), exact),
        ("no_name_token", BlockingConfig(), ("country_name_exact", "country_address_exact", "address_token", "name_address_token")),
        ("no_address_token", BlockingConfig(), ("country_name_exact", "country_address_exact", "name_token", "name_address_token")),
        ("name_cap_25", BlockingConfig(name_token_max_frequency=25), default),
        ("address_cap_25", BlockingConfig(address_token_max_frequency=25), default),
        ("name_cap_10", BlockingConfig(name_token_max_frequency=10), default),
        ("address_cap_10", BlockingConfig(address_token_max_frequency=10), default),
        ("name_cap_10_address_cap_25", BlockingConfig(name_token_max_frequency=10, address_token_max_frequency=25), default),
        ("name_cap_25_address_cap_10", BlockingConfig(name_token_max_frequency=25, address_token_max_frequency=10), default),
        ("name_cap_25_address_cap_25", BlockingConfig(name_token_max_frequency=25, address_token_max_frequency=25), default),
        ("strict_overlap_2", BlockingConfig(), exact + ("name_token_overlap_2", "address_token_overlap_2", "name_address_token")),
        ("strict_overlap_3", BlockingConfig(), exact + ("name_token_overlap_3", "address_token_overlap_3", "name_address_token")),
        ("address_overlap_2_only", BlockingConfig(), exact + ("name_token", "address_token_overlap_2", "name_address_token")),
        ("name_overlap_2_only", BlockingConfig(), exact + ("name_token_overlap_2", "address_token", "name_address_token")),
    ]


def _configuration_row(name, blocker, candidates, ground_truth, rules):
    metrics = evaluate_candidate_pairs(candidates, ground_truth, s1_ids=blocker.s1_ids)
    return {
        "configuration": name,
        "rules": "|".join(rules or blocker.config.default_rules),
        "candidates": metrics["candidate_count"],
        "avg/S1": metrics["mean_candidates_per_s1"],
        "median": metrics["median_candidates_per_s1"],
        "p90": metrics["p90_candidates_per_s1"],
        "p95": metrics["p95_candidates_per_s1"],
        "p99": metrics["p99_candidates_per_s1"],
        "max": metrics["maximum_candidates_per_s1"],
        "recall": metrics["blocking_recall"],
        "S2 recall": metrics["s2_recall"],
        "S3 recall": metrics["s3_recall"],
        "missed_true_links": metrics["true_match_count"] - metrics["true_matches_retrieved"],
    }


def _pareto_frontier(report: pd.DataFrame) -> pd.DataFrame:
    """Keep configurations not dominated on count, p95, p99, and recall."""
    frontier = []
    for i, row in report.iterrows():
        dominated = False
        for j, other in report.iterrows():
            if i == j:
                continue
            no_worse = (
                other["candidates"] <= row["candidates"]
                and other["p95"] <= row["p95"]
                and other["p99"] <= row["p99"]
                and other["recall"] >= row["recall"]
            )
            strictly_better = (
                other["candidates"] < row["candidates"]
                or other["p95"] < row["p95"]
                or other["p99"] < row["p99"]
                or other["recall"] > row["recall"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return pd.DataFrame(frontier).sort_values(
        ["candidates", "p95", "p99", "recall"],
        ascending=[True, True, True, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "processed" / "sample")
    parser.add_argument("--ground-truth", type=Path, default=PROJECT_ROOT / "dataset" / "sample" / "train_ground_truth_sample.tsv")
    parser.add_argument("--name-token-max-frequency", type=int, default=50)
    parser.add_argument("--address-token-max-frequency", type=int, default=50)
    parser.add_argument("--max-exact-bucket-size", type=int, default=500)
    args = parser.parse_args()

    started = time.perf_counter()
    root = args.processed_root
    source1 = _read(root / "train_source1_sample.tsv")
    source2 = _read(root / "train_source2_sample.tsv")
    source3 = _read(root / "train_source3_sample.tsv")
    ground_truth = _read(args.ground_truth)
    base_config = BlockingConfig(
        name_token_max_frequency=args.name_token_max_frequency,
        address_token_max_frequency=args.address_token_max_frequency,
        max_exact_bucket_size=args.max_exact_bucket_size,
    )
    base_blocker = CandidateBlocker(base_config).fit(source1, source2, source3)
    default_candidates = base_blocker.generate()

    print("Per-S1 distribution: default_union")
    default_per_s1 = candidate_pairs_by_s1(
        default_candidates, ground_truth, s1_ids=base_blocker.s1_ids
    )
    distribution_rows = []
    for label, frame in (
        ("all_s1", default_per_s1),
        ("s1_with_truth", default_per_s1.loc[default_per_s1["has_ground_truth_match"]]),
        ("s1_without_truth", default_per_s1.loc[~default_per_s1["has_ground_truth_match"]]),
    ):
        distribution_rows.append({"population": label, **summarize_candidate_distribution(frame)})
    print(pd.DataFrame(distribution_rows).to_string(index=False))
    print("\nRule contribution report: default_union")
    contribution = evaluate_rule_contributions(base_blocker, ground_truth)
    print(
        contribution[
            [
                "rule",
                "candidate_count",
                "unique_candidate_count",
                "overlap_candidate_count",
                "true_links_retrieved",
                "unique_true_links_retrieved",
                "mean_candidates_per_s1",
                "p95_candidates_per_s1",
                "p99_candidates_per_s1",
                "maximum_candidates_per_s1",
                "blocking_recall",
            ]
        ].to_string(index=False)
    )

    rows = []
    for name, config, rules in _configurations(base_config):
        if name == "default_union" and config == base_config:
            blocker = base_blocker
        else:
            blocker = CandidateBlocker(config).fit(source1, source2, source3)
        candidates = blocker.generate(rules)
        rows.append(_configuration_row(name, blocker, candidates, ground_truth, rules))
    report = pd.DataFrame(rows)
    print("\nConfiguration comparison")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nPareto frontier (candidates, p95, p99, recall)")
    print(_pareto_frontier(report).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nElapsed seconds: {time.perf_counter() - started:.3f}")
    print(f"Peak RSS MB: {_rss_mb():.1f}")


if __name__ == "__main__":
    main()
