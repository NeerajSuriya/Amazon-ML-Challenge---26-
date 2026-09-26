#!/usr/bin/env python3
"""Run the five-seed ADDRESS/CROSS_FIELD pair-feature robustness experiment."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_pair_feature_ablations import (
    PROJECT_ROOT,
    _parser,
    paired_differences,
    run_ablation,
)


DEFAULT_SEEDS = "42,123,2024,3407,7777"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "pair_feature_robustness"


def main() -> None:
    parser = _parser()
    parser.set_defaults(
        robustness=True,
        seeds=DEFAULT_SEEDS,
        per_seed_output=OUTPUT_ROOT / "per_seed.tsv",
        aggregate_output=OUTPUT_ROOT / "aggregate.tsv",
        threshold_output=OUTPUT_ROOT / "thresholds.tsv",
        summary_output=OUTPUT_ROOT / "summary.tsv",
        paired_output=OUTPUT_ROOT / "paired_per_seed.tsv",
        paired_summary_output=OUTPUT_ROOT / "paired_summary.tsv",
        error_output=OUTPUT_ROOT / "error_analysis.tsv",
        error_feature_set="baseline_plus_address",
    )
    args = parser.parse_args()
    per_seed, aggregate, thresholds, summary = run_ablation(args)
    paired_per_seed, paired_summary = paired_differences(per_seed)
    frames = [
        (per_seed, args.per_seed_output),
        (aggregate, args.aggregate_output),
        (thresholds, args.threshold_output),
        (summary, args.summary_output),
        (paired_per_seed, args.paired_output),
        (paired_summary, args.paired_summary_output),
    ]
    for frame, path in frames:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
    print("Five-seed robustness results")
    print(per_seed.to_string(index=False))
    print("\nAggregate results")
    print(aggregate.to_string(index=False))
    print("\nPaired differences against baseline")
    print(paired_summary.to_string(index=False))
    print("\nCandidate summary")
    print(summary.to_string(index=False))
    print(f"\nArtifacts: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
