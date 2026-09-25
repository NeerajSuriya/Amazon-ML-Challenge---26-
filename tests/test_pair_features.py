import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import entity_f05, macro_f05
from src.pair_features import (
    PairFeatureGenerator,
    attach_ground_truth_labels,
    build_candidate_pairs_from_ground_truth,
    ground_truth_summary,
    parse_json_list,
    split_ground_truth_by_s1,
)


def _record(entity_id, name, address, country="India", source=None):
    source = source or entity_id[:2]
    return {
        "entity_id": entity_id,
        "name_norm": name,
        "address_norm": address,
        "country_norm": country,
        "name_tokens": json.dumps(name.split() if name else []),
        "address_tokens": json.dumps(address.split() if address else []),
        "name_char_ngrams": json.dumps([name[:3]] if name else []),
        "address_char_ngrams": json.dumps([address[:3]] if address else []),
        "source": source,
    }


def _frames():
    s1 = pd.DataFrame([_record("S1-1", "abc private limited", "1 main road")])
    s2 = pd.DataFrame(
        [
            _record("S2-1", "abc private limited", "1 main road"),
            _record("S2-2", "different company", "9 other street", "France"),
        ]
    )
    s3 = pd.DataFrame([_record("S3-1", "abc private ltd", "1 main road")])
    return s1, s2, s3


def test_json_parsing_and_malformed_values():
    assert parse_json_list('["abc", "limited"]') == ["abc", "limited"]
    assert parse_json_list([]) == []
    assert parse_json_list("") == []
    assert parse_json_list(float("nan")) == []
    with pytest.raises(ValueError):
        parse_json_list('{"not": "a list"}')


def test_exact_match_and_non_match_features():
    s1, s2, s3 = _frames()
    generator = PairFeatureGenerator().fit(s1, s2, s3)
    candidates = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-2"},
        ]
    )
    output = generator.transform(candidates)
    assert output.loc[0, "name_exact"] == 1
    assert output.loc[0, "address_exact"] == 1
    assert output.loc[0, "country_exact"] == 1
    assert output.loc[1, "name_exact"] == 0
    assert output.loc[1, "address_exact"] == 0
    assert output.loc[1, "country_exact"] == 0
    assert list(output["candidate_entity_id"]) == ["S2-1", "S2-2"]
    assert list(output["candidate_source"]) == ["S2", "S2"]


def test_empty_fields_and_empty_candidates_are_supported():
    s1 = pd.DataFrame([_record("S1-1", "", "", "")])
    s2 = pd.DataFrame([_record("S2-1", "", "", "")])
    generator = PairFeatureGenerator().fit(s1, s2)
    pair = generator.transform(
        pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"}])
    )
    assert pair.loc[0, "name_tfidf_cosine"] == 1
    assert pair.loc[0, "address_tfidf_cosine"] == 1
    empty = generator.transform(pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"]))
    assert empty.empty
    assert "name_ratio" in empty.columns


def test_zero_to_many_ground_truth_labels_and_duplicate_candidates():
    candidates = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-a", "block": "name"},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-a", "block": "address"},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-b", "block": "name"},
            {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-a", "block": "name"},
        ]
    )
    truth = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-a,S3-b"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
        ]
    )
    labeled = attach_ground_truth_labels(candidates, truth)
    assert list(labeled["label"]) == [1, 1, 1, 0]
    assert list(labeled["block"]) == ["name", "address", "name", "name"]
    positives = build_candidate_pairs_from_ground_truth(truth)
    assert set(zip(positives.source1_entity_id, positives.candidate_entity_id)) == {
        ("S1-1", "S2-a"),
        ("S1-1", "S3-b"),
    }


def test_split_is_s1_level_and_summary_counts_edges():
    truth = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-a,S3-a"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
            {"source1_entity_id": "S1-3", "matched_entity_ids": "S2-b"},
            {"source1_entity_id": "S1-4", "matched_entity_ids": "S3-c"},
            {"source1_entity_id": "S1-5", "matched_entity_ids": "S2-d"},
        ]
    )
    train, validation = split_ground_truth_by_s1(truth, validation_fraction=0.4, seed=42)
    assert set(train.source1_entity_id).isdisjoint(validation.source1_entity_id)
    assert set(train.source1_entity_id) | set(validation.source1_entity_id) == set(truth.source1_entity_id)
    assert ground_truth_summary(truth) == {
        "s1_entities": 5,
        "ground_truth_rows": 5,
        "empty_truth_entities": 1,
        "positive_links": 5,
        "s2_links": 3,
        "s3_links": 2,
    }


def test_macro_f05_handles_empty_and_multiple_sets():
    truth = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-a,S2-b"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
            {"source1_entity_id": "S1-3", "matched_entity_ids": "S3-c"},
        ]
    )
    predictions = {"S1-1": ["S2-a", "S2-b", "S2-extra"], "S1-2": []}
    assert entity_f05([], []) == 1.0
    assert entity_f05(["S2-extra"], []) == 0.0
    assert entity_f05(["S2-a"], ["S2-a", "S2-b"]) == pytest.approx(5 / 6)
    assert macro_f05(predictions, truth) == pytest.approx((5 / 7 + 1.0 + 0.0) / 3)


def test_missing_prediction_key_is_empty_and_duplicate_truth_rows_are_unioned():
    truth = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-a"},
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S3-a"},
        ]
    )
    assert macro_f05({"S1-1": ["S2-a", "S3-a"]}, truth) == 1.0
    assert macro_f05({}, truth) == 0.0
