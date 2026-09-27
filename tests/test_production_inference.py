import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_production_inference import (  # noqa: E402
    OUTPUT_COLUMNS,
    read_tsv,
    run_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _record(entity_id, name, address, country="india"):
    return {
        "entity_id": entity_id,
        "name_norm": name,
        "address_norm": address,
        "country_norm": country,
        "name_tokens": json.dumps(name.split() if name else []),
        "address_tokens": json.dumps(address.split() if address else []),
        "name_char_ngrams": json.dumps(
            [name[i : i + 3] for i in range(max(0, len(name) - 2))]
        ),
        "address_char_ngrams": json.dumps(
            [address[i : i + 3] for i in range(max(0, len(address) - 2))]
        ),
        "source": entity_id[:2],
    }


def _small_inputs():
    source1 = pd.DataFrame(
        [
            _record("S1-1", "alpha one", "1 main road"),
            _record("S1-2", "beta two", "2 main road"),
        ]
    )
    source2 = pd.DataFrame(
        [
            _record("S2-1", "alpha one", "1 main road"),
            _record("S2-2", "alpha other", "99 other road"),
            _record("S2-3", "beta two", "2 main road"),
        ]
    )
    source3 = pd.DataFrame(
        [
            _record("S3-1", "alpha one", "1 main road"),
            _record("S3-2", "unrelated", "77 remote road"),
        ]
    )
    truth = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
        ]
    )
    return source1, source2, source3, truth


def _run_small(tmp_path, *, threshold=0.85, inference=None):
    source1, source2, source3, truth = _small_inputs()
    inference = inference or (source1, source2, source3)
    scored_path = tmp_path / "scored.tsv"
    matches_path = tmp_path / "matches.tsv"
    summary = run_pipeline(
        train_source1=source1,
        train_source2=source2,
        train_source3=source3,
        ground_truth=truth,
        inference_source1=inference[0],
        inference_source2=inference[1],
        inference_source3=inference[2],
        scored_output=scored_path,
        matches_output=matches_path,
        threshold=threshold,
    )
    return summary, pd.read_csv(scored_path, sep="\t"), pd.read_csv(matches_path, sep="\t")


def test_end_to_end_preserves_identity_order_and_probability_rows(tmp_path):
    summary, scored, matches = _run_small(tmp_path)

    assert list(scored.columns) == OUTPUT_COLUMNS
    assert list(matches.columns) == OUTPUT_COLUMNS
    assert len(scored) == summary["inference_candidate_count"]
    assert len(scored) == summary["matcher_output_count"]
    assert scored[["source1_entity_id", "candidate_entity_id"]].duplicated().sum() == 0
    assert scored["match_probability"].between(0.0, 1.0).all()
    assert len(matches) <= len(scored)
    assert (matches["match_probability"] >= 0.85).all()
    assert summary["training_feature_columns"] == 20
    assert summary["training_candidate_count"] == summary["training_feature_rows"]


def test_threshold_is_applied_without_reordering(tmp_path):
    low_summary, low_scored, low_matches = _run_small(tmp_path / "low", threshold=0.0)
    high_summary, high_scored, high_matches = _run_small(tmp_path / "high", threshold=1.0)

    assert low_summary["predicted_match_count"] == low_summary["matcher_output_count"]
    assert high_summary["predicted_match_count"] <= low_summary["predicted_match_count"]
    assert list(
        low_scored[["source1_entity_id", "candidate_entity_id"]].itertuples(
            index=False, name=None
        )
    ) == list(
        low_matches[["source1_entity_id", "candidate_entity_id"]].itertuples(
            index=False, name=None
        )
    )
    assert list(
        low_scored[["source1_entity_id", "candidate_entity_id"]].itertuples(
            index=False, name=None
        )
    ) == list(
        high_scored[["source1_entity_id", "candidate_entity_id"]].itertuples(
            index=False, name=None
        )
    )


def test_empty_inference_candidates_produce_empty_outputs(tmp_path):
    source1, source2, source3, _ = _small_inputs()
    empty_s1 = pd.DataFrame([_record("S1-new", "zzzz unique", "999 nowhere", "france")])
    empty_s2 = pd.DataFrame([_record("S2-new", "xxxx unrelated", "888 elsewhere", "india")])
    empty_s3 = pd.DataFrame([_record("S3-new", "yyyy unrelated", "777 elsewhere", "india")])
    truth = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
        ]
    )
    scored_path = tmp_path / "empty-scored.tsv"
    matches_path = tmp_path / "empty-matches.tsv"
    summary = run_pipeline(
        train_source1=source1,
        train_source2=source2,
        train_source3=source3,
        ground_truth=truth,
        inference_source1=empty_s1,
        inference_source2=empty_s2,
        inference_source3=empty_s3,
        scored_output=scored_path,
        matches_output=matches_path,
    )

    assert summary["inference_candidate_count"] == 0
    assert summary["matcher_output_count"] == 0
    assert summary["predicted_match_count"] == 0
    assert list(pd.read_csv(scored_path, sep="\t").columns) == OUTPUT_COLUMNS
    assert list(pd.read_csv(matches_path, sep="\t").columns) == OUTPUT_COLUMNS


def test_repository_sample_data_uses_existing_processed_and_truth_conventions(tmp_path):
    processed = PROJECT_ROOT / "processed" / "sample"
    truth_path = PROJECT_ROOT / "dataset" / "sample" / "train_ground_truth_sample.tsv"
    source_paths = [
        processed / "train_source1_sample.tsv",
        processed / "train_source2_sample.tsv",
        processed / "train_source3_sample.tsv",
    ]
    if not all(path.is_file() for path in [*source_paths, truth_path]):
        pytest.skip("repository sample inputs are unavailable")

    sources = [read_tsv(path, path.name) for path in source_paths]
    truth = read_tsv(truth_path, truth_path.name)
    summary = run_pipeline(
        train_source1=sources[0],
        train_source2=sources[1],
        train_source3=sources[2],
        ground_truth=truth,
        inference_source1=sources[0],
        inference_source2=sources[1],
        inference_source3=sources[2],
        scored_output=tmp_path / "sample-scored.tsv",
        matches_output=tmp_path / "sample-matches.tsv",
    )

    assert summary["inference_candidate_count"] == 161162
    assert summary["matcher_output_count"] == 161162
    assert summary["training_feature_columns"] == 20


def test_missing_column_and_missing_path_fail_clearly(tmp_path):
    source1, source2, source3, truth = _small_inputs()
    broken_source = source1.drop(columns=["name_char_ngrams"])
    with pytest.raises(ValueError, match="name_char_ngrams"):
        run_pipeline(
            train_source1=broken_source,
            train_source2=source2,
            train_source3=source3,
            ground_truth=truth,
            inference_source1=source1,
            inference_source2=source2,
            inference_source3=source3,
            scored_output=tmp_path / "scored.tsv",
            matches_output=tmp_path / "matches.tsv",
        )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_tsv(tmp_path / "missing.tsv", "training source 1")


def test_duplicate_ids_and_single_class_training_fail(tmp_path):
    source1, source2, source3, truth = _small_inputs()
    duplicate_source1 = pd.concat([source1, source1.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique entity_id"):
        run_pipeline(
            train_source1=duplicate_source1,
            train_source2=source2,
            train_source3=source3,
            ground_truth=truth,
            inference_source1=source1,
            inference_source2=source2,
            inference_source3=source3,
            scored_output=tmp_path / "duplicate-scored.tsv",
            matches_output=tmp_path / "duplicate-matches.tsv",
        )

    no_positive_truth = truth.copy()
    no_positive_truth["matched_entity_ids"] = ""
    with pytest.raises(ValueError, match="both positive and negative"):
        run_pipeline(
            train_source1=source1,
            train_source2=source2,
            train_source3=source3,
            ground_truth=no_positive_truth,
            inference_source1=source1,
            inference_source2=source2,
            inference_source3=source3,
            scored_output=tmp_path / "single-class-scored.tsv",
            matches_output=tmp_path / "single-class-matches.tsv",
        )


def test_empty_fields_are_valid_processed_records(tmp_path):
    empty_s1 = pd.DataFrame([_record("S1-empty", "", "", "")])
    empty_s2 = pd.DataFrame([_record("S2-empty", "", "", "")])
    empty_s3 = pd.DataFrame([_record("S3-empty", "other", "other road", "france")])
    truth = pd.DataFrame([{"source1_entity_id": "S1-empty", "matched_entity_ids": "S2-empty"}])
    # Include empty-field records in the training population.  Blocking correctly
    # emits no pair for empty blocking keys, while feature preparation remains valid.
    source1, source2, source3, base_truth = _small_inputs()
    train_truth = pd.concat([base_truth, truth], ignore_index=True)
    train_s1 = pd.concat([source1, empty_s1], ignore_index=True)
    train_s2 = pd.concat([source2, empty_s2], ignore_index=True)
    train_s3 = pd.concat([source3, empty_s3], ignore_index=True)
    scored_path = tmp_path / "empty-fields-scored.tsv"
    matches_path = tmp_path / "empty-fields-matches.tsv"
    summary = run_pipeline(
        train_source1=train_s1,
        train_source2=train_s2,
        train_source3=train_s3,
        ground_truth=train_truth,
        inference_source1=empty_s1,
        inference_source2=empty_s2,
        inference_source3=empty_s3,
        scored_output=scored_path,
        matches_output=matches_path,
    )
    assert summary["matcher_output_count"] == 0
    assert list(pd.read_csv(scored_path, sep="\t").columns) == OUTPUT_COLUMNS
