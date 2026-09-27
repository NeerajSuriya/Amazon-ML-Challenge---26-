import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_submission_pipeline import run_submission  # noqa: E402
from src.submission import (  # noqa: E402
    CANDIDATE_OUTPUT_COLUMNS,
    MATCH_OUTPUT_COLUMNS,
    aggregate_candidate_pairs,
    aggregate_matches,
    validate_submission_frames,
)


def _raw(entity_id, name, address, country):
    return {
        "entity_id": entity_id,
        "business_name": name,
        "business_address": address,
        "country": country,
    }


def _raw_fixture(tmp_path):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    s1 = pd.DataFrame(
        [
            _raw("S1-1", "Alpha One", "1 Main Road", "India"),
            _raw("S1-2", "Beta Two", "2 Main Road", "India"),
            _raw("S1-3", "Zeta No Match", "999 Remote Lane", "France"),
        ]
    )
    s2 = pd.DataFrame(
        [
            _raw("S2-1", "Alpha One", "1 Main Road", "India"),
            _raw("S2-2", "Beta Two", "2 Main Road", "India"),
            _raw("S2-3", "Alpha Other", "1 Main Road", "India"),
        ]
    )
    s3 = pd.DataFrame(
        [
            _raw("S3-1", "Alpha One", "1 Main Road", "India"),
            _raw("S3-2", "Beta Two", "2 Main Road", "India"),
            _raw("S3-3", "Unrelated", "77 Remote Road", "India"),
        ]
    )
    truth = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-2,S3-2"},
            {"source1_entity_id": "S1-3", "matched_entity_ids": ""},
        ]
    )
    for directory, suffix in ((train_dir, ""), (test_dir, "test")):
        for frame, name in (
            (s1, "source1"),
            (s2, "source2"),
            (s3, "source3"),
        ):
            frame.to_csv(directory / f"{name}.tsv", sep="\t", index=False)
    truth.to_csv(train_dir / "ground_truth.tsv", sep="\t", index=False)
    return {
        "train_source1": train_dir / "source1.tsv",
        "train_source2": train_dir / "source2.tsv",
        "train_source3": train_dir / "source3.tsv",
        "ground_truth": train_dir / "ground_truth.tsv",
        "test_source1": test_dir / "source1.tsv",
        "test_source2": test_dir / "source2.tsv",
        "test_source3": test_dir / "source3.tsv",
    }, s1, s2, s3


def test_aggregation_includes_zero_and_preserves_zero_to_many():
    pairs = pd.DataFrame(
        [
            {"source1_entity_id": "S1-2", "candidate_entity_id": "S3-2"},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-1"},
        ]
    )
    candidates = aggregate_candidate_pairs(
        pairs,
        ["S1-1", "S1-2", "S1-3"],
        source2_ids=["S2-1"],
        source3_ids=["S3-1", "S3-2"],
    )
    assert list(candidates.columns) == CANDIDATE_OUTPUT_COLUMNS
    assert candidates.to_dict("records") == [
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S3-1"},
        {"source1_entity_id": "S1-2", "candidate_entity_ids": "S3-2"},
        {"source1_entity_id": "S1-3", "candidate_entity_ids": ""},
    ]
    matches = aggregate_matches(
        pairs.iloc[[0, 1]],
        ["S1-1", "S1-2", "S1-3"],
        candidate_pairs=pairs,
        source2_ids=["S2-1"],
        source3_ids=["S3-1", "S3-2"],
    )
    assert list(matches.columns) == MATCH_OUTPUT_COLUMNS
    assert matches.loc[matches.source1_entity_id == "S1-1", "matched_entity_ids"].iloc[0] == "S2-1"
    assert matches.loc[matches.source1_entity_id == "S1-3", "matched_entity_ids"].iloc[0] == ""


def test_aggregation_rejects_duplicate_identity_and_subset_violation():
    duplicate = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_candidate_pairs(duplicate, ["S1-1"], source2_ids=["S2-1"])

    candidates = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"}])
    matches = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S3-1"}])
    with pytest.raises(ValueError, match="subset"):
        aggregate_matches(matches, ["S1-1"], candidate_pairs=candidates)


def test_validator_rejects_invalid_namespace_and_internal_empty_id():
    candidate = pd.DataFrame(
        [{"source1_entity_id": "S1-1", "candidate_entity_ids": "S1-2"}]
    )
    matching = pd.DataFrame(
        [{"source1_entity_id": "S1-1", "matched_entity_ids": ""}]
    )
    with pytest.raises(ValueError, match="namespace|unknown"):
        validate_submission_frames(
            candidate,
            matching,
            test_source1_ids=["S1-1"],
            test_source2_ids=["S2-1"],
            test_source3_ids=[],
        )

    candidate["candidate_entity_ids"] = "S2-1,,S2-1"
    with pytest.raises(ValueError, match="duplicate|namespace"):
        validate_submission_frames(
            candidate,
            matching,
            test_source1_ids=["S1-1"],
            test_source2_ids=["S2-1"],
            test_source3_ids=[],
        )


def test_raw_to_submission_end_to_end_and_validation(tmp_path):
    paths, s1, s2, s3 = _raw_fixture(tmp_path)
    output_dir = tmp_path / "output"
    result = run_submission(
        paths=paths,
        output_dir=output_dir,
        threshold=0.85,
        inference_chunk_size=2,
        evaluation_ground_truth=paths["ground_truth"],
        evaluation_thresholds=[0.5, 0.85],
        preprocessing_chunk_size=2,
    )
    assert result["validation_status"] == "passed"
    assert result["s1_count"] == len(s1)
    assert result["inference_feature_rows"] == result["inference_candidate_count"]
    assert len(result["threshold_scores"]) == 2
    candidate = pd.read_csv(output_dir / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    matching = pd.read_csv(output_dir / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert list(candidate.columns) == CANDIDATE_OUTPUT_COLUMNS
    assert list(matching.columns) == MATCH_OUTPUT_COLUMNS
    assert len(candidate) == len(s1)
    assert len(matching) == len(s1)
    checked = validate_submission_frames(
        candidate,
        matching,
        test_source1_ids=s1.entity_id,
        test_source2_ids=s2.entity_id,
        test_source3_ids=s3.entity_id,
    )
    assert checked["candidate_pair_count"] == result["candidate_pair_count"]
    assert checked["predicted_match_count"] == result["predicted_match_count"]
