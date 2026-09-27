#!/usr/bin/env python3
"""Train and run the frozen production matcher on blocked candidate pairs.

This is an explicit-path integration entry point.  It does not provide implicit
repository or competition-data paths and it does not claim an external
submission schema.  The command fits ``ProductionMatcher`` on labeled
candidates, scores a second (possibly identical) source set, and writes local
scored and thresholded TSV artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import (  # noqa: E402
    CANDIDATE_ID,
    SOURCE1_ID,
    BlockingConfig,
    CandidateBlocker,
)
from src.matcher import (  # noqa: E402
    FEATURE_COLUMNS,
    MATCH_THRESHOLD,
    PAIR_ID_COLUMNS,
    ProductionMatcher,
)
from src.pair_features import (  # noqa: E402
    PairFeatureGenerator,
    attach_ground_truth_labels,
)
from src.submission import (  # noqa: E402
    aggregate_candidate_pairs,
    aggregate_matches,
)

PROCESSED_REQUIRED_COLUMNS = (
    "entity_id",
    "name_norm",
    "address_norm",
    "country_norm",
    "name_tokens",
    "address_tokens",
    "name_char_ngrams",
    "address_char_ngrams",
)
GROUND_TRUTH_REQUIRED_COLUMNS = (SOURCE1_ID, "matched_entity_ids")
OUTPUT_COLUMNS = [SOURCE1_ID, CANDIDATE_ID, "match_probability"]


def read_tsv(path: Path, label: str) -> pd.DataFrame:
    """Read one required TSV input using repository conventions."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _validate_required_columns(
    frame: pd.DataFrame, required: Sequence[str], label: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _validate_processed_frame(frame: pd.DataFrame, label: str) -> None:
    _validate_required_columns(frame, PROCESSED_REQUIRED_COLUMNS, label)


def _validate_ground_truth(frame: pd.DataFrame) -> None:
    _validate_required_columns(frame, GROUND_TRUTH_REQUIRED_COLUMNS, "ground truth")
    s1_ids = frame[SOURCE1_ID]
    if s1_ids.isna().any() or s1_ids.astype(str).str.strip().eq("").any():
        raise ValueError("ground truth contains missing or empty source1_entity_id values")
    if not s1_ids.astype(str).str.startswith("S1-", na=False).all():
        raise ValueError("ground truth contains a source1 ID outside the S1- namespace")


def _validate_candidate_pairs(candidates: pd.DataFrame, label: str) -> None:
    """Validate the blocker identity contract without changing row order."""
    _validate_required_columns(candidates, PAIR_ID_COLUMNS, label)
    identity = candidates[PAIR_ID_COLUMNS]
    for column in PAIR_ID_COLUMNS:
        values = identity[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{label} contains missing or empty {column} values")
    if identity.duplicated().any():
        raise ValueError(f"{label} contains duplicate candidate identities")
    if not identity[SOURCE1_ID].astype(str).str.startswith("S1-", na=False).all():
        raise ValueError(f"{label} contains a source1 ID outside the S1- namespace")
    if not identity[CANDIDATE_ID].astype(str).str.startswith(("S2-", "S3-"), na=False).all():
        raise ValueError(f"{label} contains a candidate outside S2-/S3- namespaces")


def _identity_tuples(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(frame[PAIR_ID_COLUMNS].itertuples(index=False, name=None))


def _combine_records(*frames: pd.DataFrame) -> pd.DataFrame:
    """Combine train/inference records, allowing identical repeated inputs.

    PairFeatureGenerator indexes records by entity ID.  Passing the same
    processed files as both training and inference inputs is useful for local
    validation, so identical repeated records are collapsed.  Conflicting
    records with one entity ID are rejected rather than silently selected.
    """
    combined = pd.concat(frames, ignore_index=True, sort=False).copy()
    if not combined["entity_id"].duplicated().any():
        return combined

    compare_columns = [
        column
        for column in PROCESSED_REQUIRED_COLUMNS + ("source",)
        if column in combined.columns
    ]
    for entity_id, group in combined.groupby("entity_id", sort=False, dropna=False):
        if len(group) <= 1:
            continue
        baseline = group.iloc[0]
        for _, row in group.iloc[1:].iterrows():
            for column in compare_columns:
                if str(row[column]) != str(baseline[column]):
                    raise ValueError(
                        f"Conflicting processed records for duplicate entity_id: {entity_id}"
                    )
    return combined.drop_duplicates(subset=["entity_id"], keep="first").reset_index(drop=True)


def _validate_feature_frame(
    features: pd.DataFrame,
    expected_feature_names: Sequence[str],
    expected_candidates: pd.DataFrame,
    label: str,
) -> None:
    expected = list(FEATURE_COLUMNS)
    if list(expected_feature_names) != expected:
        raise ValueError(
            f"{label} feature schema mismatch: expected {expected}, "
            f"got {list(expected_feature_names)}"
        )
    _validate_required_columns(features, [*PAIR_ID_COLUMNS, *expected], label)
    if len(features) != len(expected_candidates):
        raise ValueError(
            f"{label} row count mismatch: {len(features)} features vs "
            f"{len(expected_candidates)} candidates"
        )
    if _identity_tuples(features) != _identity_tuples(expected_candidates):
        raise ValueError(f"{label} candidate identity row order changed")
    try:
        values = features[expected].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains non-numeric matcher features") from exc
    if values.size and not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite matcher features")


def _build_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    *,
    label: str,
) -> tuple[CandidateBlocker, pd.DataFrame]:
    blocker = CandidateBlocker(BlockingConfig()).fit(source1, source2, source3)
    candidates = blocker.generate()
    _validate_candidate_pairs(candidates, label)
    return blocker, candidates


def _write_output(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def run_pipeline(
    *,
    train_source1: pd.DataFrame,
    train_source2: pd.DataFrame,
    train_source3: pd.DataFrame,
    ground_truth: pd.DataFrame,
    inference_source1: pd.DataFrame,
    inference_source2: pd.DataFrame,
    inference_source3: pd.DataFrame,
    scored_output: Path,
    matches_output: Path,
    threshold: float = MATCH_THRESHOLD,
    random_state: int = 42,
    candidate_submission_output: Path | None = None,
    matching_submission_output: Path | None = None,
    inference_chunk_size: int | None = 100_000,
) -> dict[str, object]:
    """Fit the frozen matcher and write pair and optional submission outputs.

    The optional aggregate outputs are built from the exact inference candidate
    frame and thresholded matcher rows used in this call.  They are not
    regenerated from a separate candidate file.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    threshold = float(threshold)
    if inference_chunk_size is not None and int(inference_chunk_size) < 1:
        raise ValueError("inference_chunk_size must be positive or None")
    inference_chunk_size = (
        None if inference_chunk_size is None else int(inference_chunk_size)
    )

    training_frames = (train_source1, train_source2, train_source3)
    inference_frames = (inference_source1, inference_source2, inference_source3)
    for index, frame in enumerate(training_frames, start=1):
        _validate_processed_frame(frame, f"training source {index}")
    for index, frame in enumerate(inference_frames, start=1):
        _validate_processed_frame(frame, f"inference source {index}")
    _validate_ground_truth(ground_truth)

    _, training_candidates = _build_candidates(*training_frames, label="training candidates")
    _, inference_candidates = _build_candidates(
        *inference_frames, label="inference candidates"
    )

    # Fit one feature view over all records so TF-IDF coordinates and ID lookup
    # are available for both the training and inference candidate populations.
    records = _combine_records(*training_frames, *inference_frames)
    generator = PairFeatureGenerator(include_experimental=False).fit(records)
    if generator.feature_names != list(FEATURE_COLUMNS):
        raise ValueError(
            f"Pair-feature schema does not match matcher schema: "
            f"{generator.feature_names}"
        )

    training_features = generator.transform(training_candidates)
    _validate_feature_frame(
        training_features,
        generator.feature_names,
        training_candidates,
        "training features",
    )
    labeled_candidates = attach_ground_truth_labels(training_candidates, ground_truth)
    if len(training_features) != len(labeled_candidates):
        raise ValueError(
            f"training label row count mismatch: {len(labeled_candidates)} labels vs "
            f"{len(training_features)} features"
        )
    training_features = training_features.copy()
    training_features["label"] = labeled_candidates["label"].to_numpy(dtype=int)
    if training_features["label"].nunique() < 2:
        raise ValueError(
            "training candidate labels must contain both positive and negative classes"
        )

    matcher = ProductionMatcher(random_state=random_state)
    matcher.fit(training_features)

    scored_chunks = []
    chunk_size = inference_chunk_size or max(len(inference_candidates), 1)
    for start in range(0, len(inference_candidates), chunk_size):
        candidate_chunk = inference_candidates.iloc[start : start + chunk_size].reset_index(drop=True)
        feature_chunk = generator.transform(candidate_chunk)
        _validate_feature_frame(
            feature_chunk,
            generator.feature_names,
            candidate_chunk,
            "inference features",
        )
        if feature_chunk.empty:
            continue
        scored_chunk = matcher.predict_dataframe(feature_chunk)
        if list(scored_chunk.columns) != OUTPUT_COLUMNS:
            raise ValueError(
                f"matcher output schema mismatch: expected {OUTPUT_COLUMNS}, "
                f"got {list(scored_chunk.columns)}"
            )
        if len(scored_chunk) != len(candidate_chunk):
            raise ValueError(
                f"matcher output row count mismatch: {len(scored_chunk)} outputs vs "
                f"{len(candidate_chunk)} candidates"
            )
        if _identity_tuples(scored_chunk) != _identity_tuples(candidate_chunk):
            raise ValueError("matcher output candidate identity row order changed")
        probabilities = scored_chunk["match_probability"].to_numpy(dtype=float)
        if probabilities.size and (
            not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or (probabilities > 1.0).any()
        ):
            raise ValueError("matcher returned a probability outside the finite [0, 1] range")
        scored_chunks.append(scored_chunk.loc[:, OUTPUT_COLUMNS])

    if scored_chunks:
        scored = pd.concat(scored_chunks, ignore_index=True)
    else:
        scored = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if len(scored) != len(inference_candidates):
        raise ValueError(
            f"matcher output row count mismatch: {len(scored)} outputs vs "
            f"{len(inference_candidates)} candidates"
        )
    if _identity_tuples(scored) != _identity_tuples(inference_candidates):
        raise ValueError("matcher output candidate identity row order changed")

    matches = scored.loc[scored["match_probability"] >= threshold].copy().reset_index(drop=True)
    _write_output(scored, scored_output)
    _write_output(matches, matches_output)

    if candidate_submission_output is not None:
        candidate_output = aggregate_candidate_pairs(
            inference_candidates,
            inference_source1["entity_id"].tolist(),
            source2_ids=inference_source2["entity_id"].tolist(),
            source3_ids=inference_source3["entity_id"].tolist(),
        )
        _write_output(candidate_output, candidate_submission_output)
    if matching_submission_output is not None:
        matching_output = aggregate_matches(
            matches,
            inference_source1["entity_id"].tolist(),
            candidate_pairs=inference_candidates,
            source2_ids=inference_source2["entity_id"].tolist(),
            source3_ids=inference_source3["entity_id"].tolist(),
        )
        _write_output(matching_output, matching_submission_output)

    candidate_counts = (
        inference_candidates.groupby(SOURCE1_ID).size()
        if not inference_candidates.empty
        else pd.Series(dtype=int)
    )
    match_counts = matches.groupby(SOURCE1_ID).size() if not matches.empty else pd.Series(dtype=int)
    inference_s1_count = int(len(inference_source1))
    return {
        "training_candidate_count": int(len(training_candidates)),
        "training_feature_rows": int(len(training_features)),
        "training_feature_columns": int(len(FEATURE_COLUMNS)),
        "inference_s1_count": inference_s1_count,
        "inference_candidate_count": int(len(inference_candidates)),
        "inference_feature_rows": int(len(inference_candidates)),
        "inference_candidate_s1_zero_count": int(inference_s1_count - len(candidate_counts.index)),
        "matcher_output_count": int(len(scored)),
        "predicted_match_count": int(len(matches)),
        "predicted_s1_zero_count": int(inference_s1_count - len(match_counts.index)),
        "predicted_s1_one_count": int((match_counts == 1).sum()),
        "predicted_s1_multiple_count": int((match_counts > 1).sum()),
        "threshold": threshold,
        "inference_chunk_size": inference_chunk_size,
        "scored_schema": list(scored.columns),
        "matches_schema": list(matches.columns),
        "scored_output": str(scored_output),
        "matches_output": str(matches_output),
        "candidate_submission_output": (
            None if candidate_submission_output is None else str(candidate_submission_output)
        ),
        "matching_submission_output": (
            None if matching_submission_output is None else str(matching_submission_output)
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix, help_text in (
        ("train", "labeled matcher-training inputs"),
        ("inference", "candidate-scoring inputs"),
    ):
        group = parser.add_argument_group(help_text)
        group.add_argument(f"--{prefix}-source1", type=Path, required=True)
        group.add_argument(f"--{prefix}-source2", type=Path, required=True)
        group.add_argument(f"--{prefix}-source3", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scored-output", type=Path, required=True)
    parser.add_argument("--matches-output", type=Path, required=True)
    parser.add_argument(
        "--candidate-submission-output",
        type=Path,
        help="Optional aggregated candidate_pairs.tsv output",
    )
    parser.add_argument(
        "--matching-submission-output",
        type=Path,
        help="Optional aggregated matching_results.tsv output",
    )
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--inference-chunk-size", type=int, default=100_000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    train_sources = tuple(
        read_tsv(path, f"training source {index}")
        for index, path in enumerate(
            (args.train_source1, args.train_source2, args.train_source3), start=1
        )
    )
    inference_sources = tuple(
        read_tsv(path, f"inference source {index}")
        for index, path in enumerate(
            (args.inference_source1, args.inference_source2, args.inference_source3),
            start=1,
        )
    )
    ground_truth = read_tsv(args.ground_truth, "ground truth")
    summary = run_pipeline(
        train_source1=train_sources[0],
        train_source2=train_sources[1],
        train_source3=train_sources[2],
        ground_truth=ground_truth,
        inference_source1=inference_sources[0],
        inference_source2=inference_sources[1],
        inference_source3=inference_sources[2],
        scored_output=args.scored_output,
        matches_output=args.matches_output,
        threshold=args.threshold,
        random_state=args.random_state,
        candidate_submission_output=args.candidate_submission_output,
        matching_submission_output=args.matching_submission_output,
        inference_chunk_size=args.inference_chunk_size,
    )
    print("Production matcher integration QA")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
