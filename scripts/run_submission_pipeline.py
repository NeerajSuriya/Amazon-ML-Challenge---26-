#!/usr/bin/env python3
"""Run the raw-data-to-submission entity-resolution pipeline.

The command preprocesses raw challenge TSVs, generates the final blocked
candidate set, trains the frozen ProductionMatcher, thresholds pair
probabilities, and writes the two one-row-per-S1 submission files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_production_inference import (  # noqa: E402
    MATCH_THRESHOLD,
    read_tsv,
    run_pipeline,
)
from src.metrics import (  # noqa: E402
    macro_f05,
    predictions_from_scored_pairs,
    score_thresholds,
)
from src.preprocessing import preprocess_file  # noqa: E402
from src.submission import validate_submission_frames  # noqa: E402

RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def _read_raw(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing raw columns: {missing}")
    return frame


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    dataset_root = Path(args.dataset_root)
    train_dir = Path(args.train_dir) if args.train_dir else dataset_root / "train"
    test_dir = Path(args.test_dir) if args.test_dir else dataset_root / "test"
    ground_truth = (
        Path(args.ground_truth)
        if args.ground_truth
        else train_dir / args.train_ground_truth_name
    )
    return {
        "train_source1": train_dir / args.train_source1_name,
        "train_source2": train_dir / args.train_source2_name,
        "train_source3": train_dir / args.train_source3_name,
        "ground_truth": ground_truth,
        "test_source1": test_dir / args.test_source1_name,
        "test_source2": test_dir / args.test_source2_name,
        "test_source3": test_dir / args.test_source3_name,
    }


def run_submission(
    *,
    paths: dict[str, Path],
    output_dir: Path,
    threshold: float = MATCH_THRESHOLD,
    random_state: int = 42,
    inference_chunk_size: int | None = 100_000,
    evaluation_ground_truth: Path | None = None,
    evaluation_thresholds: list[float] | None = None,
    ngram_size: int = 3,
    preprocessing_chunk_size: int = 50_000,
) -> dict[str, object]:
    """Run the complete pipeline and return QA/evaluation information."""
    raw = {
        name: _read_raw(path, name.replace("_", " "))
        for name, path in paths.items()
        if name != "ground_truth"
    }
    ground_truth = read_tsv(paths["ground_truth"], "ground truth")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="entity-resolution-") as temp_dir:
        processed_dir = Path(temp_dir)
        processed = {}
        for name, frame in raw.items():
            source_path = paths[name]
            processed_path = processed_dir / f"{name}.tsv"
            # Read/validate the raw frame above before using the existing
            # chunked preprocessor, which is the production normalization path.
            del frame
            preprocess_file(
                source_path,
                processed_path,
                ngram_size=ngram_size,
                chunksize=preprocessing_chunk_size,
            )
            processed[name] = read_tsv(processed_path, name.replace("_", " "))

        scored_path = processed_dir / "scored_candidates.tsv"
        pair_matches_path = processed_dir / "thresholded_matches.tsv"
        summary = run_pipeline(
            train_source1=processed["train_source1"],
            train_source2=processed["train_source2"],
            train_source3=processed["train_source3"],
            ground_truth=ground_truth,
            inference_source1=processed["test_source1"],
            inference_source2=processed["test_source2"],
            inference_source3=processed["test_source3"],
            scored_output=scored_path,
            matches_output=pair_matches_path,
            threshold=threshold,
            random_state=random_state,
            candidate_submission_output=output_dir / "candidate_pairs.tsv",
            matching_submission_output=output_dir / "matching_results.tsv",
            inference_chunk_size=inference_chunk_size,
        )

        candidate_output = read_tsv(output_dir / "candidate_pairs.tsv", "candidate output")
        matching_output = read_tsv(output_dir / "matching_results.tsv", "matching output")
        validation = validate_submission_frames(
            candidate_output,
            matching_output,
            test_source1_ids=raw["test_source1"]["entity_id"].tolist(),
            test_source2_ids=raw["test_source2"]["entity_id"].tolist(),
            test_source3_ids=raw["test_source3"]["entity_id"].tolist(),
        )

        result: dict[str, object] = {**summary, **validation, "validation_status": "passed"}
        n_s1 = len(raw["test_source1"])
        cartesian = n_s1 * (len(raw["test_source2"]) + len(raw["test_source3"]))
        result["cartesian_pair_count"] = cartesian
        result["candidate_reduction_ratio"] = (
            1.0 - (validation["candidate_pair_count"] / cartesian) if cartesian else 0.0
        )

        if evaluation_ground_truth is not None:
            eval_truth = read_tsv(evaluation_ground_truth, "evaluation ground truth")
            scored = read_tsv(scored_path, "scored candidates")
            scored["match_probability"] = pd.to_numeric(
                scored["match_probability"], errors="raise"
            )
            predictions = predictions_from_scored_pairs(
                scored,
                threshold,
                score_col="match_probability",
            )
            result["macro_f0_5"] = macro_f05(predictions, eval_truth)
            if evaluation_thresholds:
                threshold_scores = score_thresholds(
                    scored,
                    eval_truth,
                    evaluation_thresholds,
                    score_col="match_probability",
                )
                result["threshold_scores"] = threshold_scores.to_dict("records")

    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--evaluation-ground-truth", type=Path)
    parser.add_argument(
        "--evaluate-thresholds",
        help="Optional comma-separated thresholds for local F0.5 analysis; does not change production threshold",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--train-source1-name", default="train_source1.tsv")
    parser.add_argument("--train-source2-name", default="train_source2.tsv")
    parser.add_argument("--train-source3-name", default="train_source3.tsv")
    parser.add_argument("--train-ground-truth-name", default="train_ground_truth.tsv")
    parser.add_argument("--test-source1-name", default="test_source1.tsv")
    parser.add_argument("--test-source2-name", default="test_source2.tsv")
    parser.add_argument("--test-source3-name", default="test_source3.tsv")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--inference-chunk-size", type=int, default=100_000)
    parser.add_argument("--preprocessing-chunk-size", type=int, default=50_000)
    parser.add_argument("--ngram-size", type=int, default=3, choices=[2, 3, 4, 5])
    return parser


def main() -> None:
    args = _parser().parse_args()
    paths = _resolve_paths(args)
    try:
        evaluation_thresholds = (
            [float(value.strip()) for value in args.evaluate_thresholds.split(",") if value.strip()]
            if args.evaluate_thresholds
            else None
        )
        if evaluation_thresholds and not all(0.0 <= value <= 1.0 for value in evaluation_thresholds):
            raise ValueError("all evaluation thresholds must be between 0 and 1")
        summary = run_submission(
            paths=paths,
            output_dir=args.output_dir,
            threshold=args.threshold,
            random_state=args.random_state,
            inference_chunk_size=args.inference_chunk_size,
            evaluation_ground_truth=args.evaluation_ground_truth,
            evaluation_thresholds=evaluation_thresholds,
            ngram_size=args.ngram_size,
            preprocessing_chunk_size=args.preprocessing_chunk_size,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"PIPELINE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("Entity-resolution submission QA")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
