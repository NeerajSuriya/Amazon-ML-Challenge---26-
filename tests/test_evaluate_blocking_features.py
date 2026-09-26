import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_blocking_features import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    _feature_rows,
    _prediction_map,
    configuration_specs,
    fit_matching_model,
    link_counts,
    score_features,
    select_threshold,
    threshold_report,
)
from src.blocking import CandidateBlocker, BlockingConfig  # noqa: E402
from src.pair_features import PairFeatureGenerator  # noqa: E402


def _record(entity_id, name, address, country="india"):
    return {
        "entity_id": entity_id,
        "name_norm": name,
        "address_norm": address,
        "country_norm": country,
        "name_tokens": json.dumps(name.split() if name else []),
        "address_tokens": json.dumps(address.split() if address else []),
        "name_char_ngrams": json.dumps([name[:3]] if name else []),
        "address_char_ngrams": json.dumps([address[:3]] if address else []),
        "source": entity_id[:2],
    }


def _small_frames():
    source1 = pd.DataFrame(
        [
            _record("S1-1", "alpha one", "1 main road"),
            _record("S1-2", "", "", ""),
            _record("S1-3", "gamma three", "3 main road", "us"),
        ]
    )
    source2 = pd.DataFrame(
        [
            _record("S2-1", "alpha one", "1 main road"),
            _record("S2-2", "gamma three", "3 main road", "us"),
            _record("S2-3", "unrelated", "99 other road", "us"),
        ]
    )
    source3 = pd.DataFrame(
        [
            _record("S3-1", "alpha one limited", "1 main road"),
            _record("S3-2", "gamma three ltd", "3 main road", "us"),
        ]
    )
    truth = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
            {"source1_entity_id": "S1-3", "matched_entity_ids": "S2-2,S3-2"},
        ]
    )
    return source1, source2, source3, truth


def test_requested_configurations_are_source_only_and_exactly_scoped():
    specs = configuration_specs(BlockingConfig())
    assert [spec.name for spec in specs] == [
        "strict_overlap_2",
        "name_cap_25_address_cap_25",
        "address_cap_25",
        "name_cap_25",
        "default_union",
    ]
    assert specs[0].rules[-3:] == (
        "name_token_overlap_2",
        "address_token_overlap_2",
        "name_address_token",
    )
    assert specs[1].config.name_token_max_frequency == 25
    assert specs[1].config.address_token_max_frequency == 25
    assert specs[-1].rules == BlockingConfig().default_rules


def test_blocked_candidates_only_are_transformed_and_s2_s3_are_retained():
    source1, source2, source3, truth = _small_frames()
    candidates = CandidateBlocker().fit(source1, source2, source3).generate(
        ["country_name_exact", "country_address_exact"]
    )
    assert set(candidates["candidate_source"]) == {"S2", "S3"}
    generator = PairFeatureGenerator().fit(source1, source2, source3)
    features = _feature_rows(generator, candidates, truth)
    assert len(features) == len(candidates)
    assert set(zip(features.source1_entity_id, features.candidate_entity_id)) == set(
        zip(candidates.source1_entity_id, candidates.candidate_entity_id)
    )
    assert {"S2-1", "S3-1"}.issubset(set(features.candidate_entity_id))
    assert len(generator.feature_names) == 20


def test_balanced_logistic_model_uses_only_feature_columns():
    features = pd.DataFrame(
        [
            {"label": 0, "f1": 0.0, "f2": 1.0},
            {"label": 0, "f1": 0.1, "f2": 0.8},
            {"label": 1, "f1": 1.0, "f2": 0.0},
        ]
    )
    model = fit_matching_model(features, ["f1", "f2"], seed=42)
    assert model.class_weight == "balanced"
    assert model.random_state == 42
    assert model.max_iter == 1000


def test_candidate_loss_is_separate_from_model_false_negatives():
    scored = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-a", "score": 0.4},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-negative", "score": 0.9},
            {"source1_entity_id": "S1-2", "candidate_entity_id": "S3-x", "score": 0.9},
        ]
    )
    truth = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-a,S2-blocked"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": "S3-x"},
        ]
    )
    counts = link_counts(scored, 0.5, truth)
    assert counts["candidate_true_links"] == 2
    assert counts["candidate_loss_links"] == 1
    assert counts["true_positive_links"] == 1
    assert counts["false_positive_links"] == 1
    assert counts["model_fn_among_candidates"] == 1
    assert counts["false_negative_links"] == 2
    assert counts["link_precision"] == pytest.approx(0.5)
    assert counts["link_recall"] == pytest.approx(1 / 3)


def test_prediction_map_includes_s1_entities_without_candidates_and_threshold_is_development_only():
    scored = pd.DataFrame(
        [{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-a", "score": 0.8}]
    )
    truth = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-a"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
        ]
    )
    predictions = _prediction_map(scored, 0.5, truth)
    assert predictions == {"S1-1": ["S2-a"], "S1-2": []}

    development = threshold_report(
        scored,
        truth,
        DEFAULT_THRESHOLDS,
        stage="development",
        configuration="test",
    )
    assert select_threshold(development) in DEFAULT_THRESHOLDS
    assert set(development["stage"]) == {"development"}


def test_cli_runs_all_configurations_and_writes_deterministic_summary(tmp_path):
    source1, source2, source3, truth = _small_frames()
    processed = tmp_path / "processed"
    processed.mkdir()
    for name, frame in (
        ("train_source1_sample.tsv", source1),
        ("train_source2_sample.tsv", source2),
        ("train_source3_sample.tsv", source3),
    ):
        frame.to_csv(processed / name, sep="\t", index=False)
    truth_path = tmp_path / "truth.tsv"
    truth.to_csv(truth_path, sep="\t", index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_blocking_features.py"
    summary_a = tmp_path / "summary_a.tsv"
    summary_b = tmp_path / "summary_b.tsv"

    command = [
        sys.executable,
        str(script),
        "--processed-root",
        str(processed),
        "--ground-truth",
        str(truth_path),
        "--validation-fraction",
        "0.34",
        "--development-fraction",
        "0.5",
        "--summary-output",
        str(summary_a),
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    command[-1] = str(summary_b)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    for name in (
        "strict_overlap_2",
        "name_cap_25_address_cap_25",
        "address_cap_25",
        "name_cap_25",
        "default_union",
    ):
        assert name in first.stdout
        assert name in second.stdout
    first_report = pd.read_csv(summary_a, sep="\t").drop(columns=["runtime_seconds"])
    second_report = pd.read_csv(summary_b, sep="\t").drop(columns=["runtime_seconds"])
    pd.testing.assert_frame_equal(first_report, second_report)
    report = pd.read_csv(summary_a, sep="\t")
    assert len(report) == 5
    assert {
        "feature_shape",
        "positive_candidate_pairs",
        "negative_candidate_pairs",
        "blocking_missed_links",
        "macro_f0_5",
        "false_negative_links",
    }.issubset(report.columns)


def test_empty_candidate_scores_preserve_empty_predictions():
    empty_features = pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "label", "f1"]
    )
    model = fit_matching_model(empty_features, ["f1"], seed=42)
    scored = score_features(model, empty_features, ["f1"])
    truth = pd.DataFrame(
        [{"source1_entity_id": "S1-1", "matched_entity_ids": ""}]
    )
    assert scored.empty
    assert link_counts(scored, 0.5, truth)["predicted_links"] == 0
    assert _prediction_map(scored, 0.5, truth) == {"S1-1": []}
