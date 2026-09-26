import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_blocking_features import fit_matcher, score_matcher  # noqa: E402
from scripts.evaluate_pair_feature_ablations import run_ablation  # noqa: E402
from src.feature_experiments import (  # noqa: E402
    aggregate_results,
    default_feature_sets,
    diagnose_features,
    paired_differences,
    resolve_feature_set,
)
from src.matching import LogisticMatcher  # noqa: E402
from src.pair_features import (  # noqa: E402
    BASELINE_FEATURE_NAMES,
    EXPERIMENTAL_FEATURE_NAMES,
    PairFeatureGenerator,
)


def _record(entity_id, name, address, country="india"):
    return {
        "entity_id": entity_id,
        "name_norm": name,
        "address_norm": address,
        "country_norm": country,
        "name_tokens": json.dumps(name.split() if name else []),
        "address_tokens": json.dumps(address.split() if address else []),
        "name_char_ngrams": json.dumps([name[i : i + 3] for i in range(max(0, len(name) - 2))]),
        "address_char_ngrams": json.dumps([address[i : i + 3] for i in range(max(0, len(address) - 2))]),
        "source": entity_id[:2],
    }


def test_experimental_features_are_ordered_and_finite():
    source1 = pd.DataFrame([_record("S1-1", "alpha road", "12 main road")])
    source2 = pd.DataFrame([_record("S2-1", "alpha road limited", "12 main road, block 4")])
    generator = PairFeatureGenerator(include_experimental=True).fit(source1, source2)
    pair = generator.transform(
        pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"}])
    )

    assert generator.feature_names == list(BASELINE_FEATURE_NAMES) + list(EXPERIMENTAL_FEATURE_NAMES)
    assert pair.loc[0, "name_token_containment_s1"] == pytest.approx(1.0)
    assert pair.loc[0, "address_numeric_jaccard"] == pytest.approx(0.5)
    assert np.isfinite(pair[generator.feature_names].to_numpy(dtype=float)).all()


def test_experimental_empty_and_missing_country_values_are_finite():
    source1 = pd.DataFrame([_record("S1-1", "", "", "")])
    source2 = pd.DataFrame([_record("S2-1", "", "", "")])
    generator = PairFeatureGenerator(include_experimental=True).fit(source1, source2)
    pair = generator.transform(
        pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"}])
    )

    assert pair.loc[0, "country_missing_either"] == 1
    assert pair.loc[0, "country_both_missing"] == 1
    assert np.isfinite(pair[generator.feature_names].to_numpy(dtype=float)).all()


def test_feature_set_resolution_rejects_invalid_and_handles_cross_field_group():
    assert resolve_feature_set("baseline").count == 20
    assert resolve_feature_set("baseline_plus_cross_field").count == 24
    assert resolve_feature_set("baseline_plus_address_cross_field").count == 29
    assert resolve_feature_set("baseline_plus_name_address_cross").count == 32
    assert len(default_feature_sets()) == 6
    with pytest.raises(ValueError):
        resolve_feature_set("baseline_plus_not_a_group")
    with pytest.raises(ValueError):
        resolve_feature_set([])
    with pytest.raises(ValueError):
        resolve_feature_set([BASELINE_FEATURE_NAMES[0], BASELINE_FEATURE_NAMES[0]])


def test_diagnostics_and_aggregation_are_deterministic():
    frame = pd.DataFrame(
        [
            {"label": 0, "feature": 0.0},
            {"label": 1, "feature": 1.0},
            {"label": 1, "feature": 0.8},
        ]
    )
    diagnostics = diagnose_features(frame, ["feature"])
    assert diagnostics.loc[0, "positive_count"] == 2
    assert diagnostics.loc[0, "negative_count"] == 1
    assert diagnostics.loc[0, "univariate_auc"] == 1.0

    per_seed = pd.DataFrame(
        [
            {"feature_set": "baseline", "validation_macro_f0_5": 0.8, "validation_precision": 0.9, "validation_recall": 0.7, "threshold": 0.9},
            {"feature_set": "baseline", "validation_macro_f0_5": 0.9, "validation_precision": 0.8, "validation_recall": 0.8, "threshold": 0.95},
        ]
    )
    aggregate = aggregate_results(per_seed)
    assert aggregate.loc[0, "validation_macro_f0_5_mean"] == pytest.approx(0.85)
    assert aggregate.loc[0, "validation_macro_f0_5_std"] == pytest.approx(0.05)
    assert aggregate.loc[0, "threshold_max"] == pytest.approx(0.95)

    paired = pd.DataFrame(
        [
            {"feature_set": "baseline", "seed": 42, "validation_macro_f0_5": 0.80},
            {"feature_set": "baseline_plus_address", "seed": 42, "validation_macro_f0_5": 0.84},
            {"feature_set": "baseline", "seed": 123, "validation_macro_f0_5": 0.82},
            {"feature_set": "baseline_plus_address", "seed": 123, "validation_macro_f0_5": 0.81},
        ]
    )
    paired_rows, paired_summary = paired_differences(paired)
    assert list(paired_rows["paired_improvement"]) == pytest.approx([0.04, -0.01])
    assert paired_summary.loc[0, "mean"] == pytest.approx(0.015)
    assert paired_summary.loc[0, "min"] == pytest.approx(-0.01)


class _FakeMatcher:
    def __init__(self):
        self.seen_columns = None

    def fit(self, features, feature_names):
        self.seen_columns = tuple(feature_names)
        return self

    def predict_proba(self, features, feature_names):
        assert tuple(feature_names) == self.seen_columns
        return np.linspace(0.1, 0.9, len(features))


def test_replaceable_matcher_preserves_row_order_and_selected_columns():
    features = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "label": 0, "f1": 0.0, "ignored": 99.0},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-1", "label": 1, "f1": 1.0, "ignored": -1.0},
        ]
    )
    matcher = fit_matcher(features, ["f1"], 42, matcher_factory=_FakeMatcher)
    scored = score_matcher(matcher, features, ["f1"])
    assert list(scored["candidate_entity_id"]) == ["S2-1", "S3-1"]
    assert list(scored["score"]) == pytest.approx([0.1, 0.9])


def test_default_logistic_matcher_integrates_through_replaceable_boundary():
    features = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "label": 0, "f1": 0.0, "ignored": 12.0},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-1", "label": 1, "f1": 1.0, "ignored": -12.0},
        ]
    )
    matcher = fit_matcher(features, ["f1"], 42)
    assert isinstance(matcher, LogisticMatcher)
    assert matcher.class_weight == "balanced"
    scored = score_matcher(matcher, features, ["f1"])
    assert list(scored["candidate_entity_id"]) == ["S2-1", "S3-1"]
    assert scored["score"].between(0.0, 1.0).all()


def test_logistic_matcher_empty_and_single_class_fallbacks():
    empty = pd.DataFrame(columns=["label", "f1"])
    matcher = LogisticMatcher(seed=42).fit(empty, ["f1"])
    assert matcher.predict_proba(empty, ["f1"]).size == 0

    one_class = pd.DataFrame([{"label": 1, "f1": 0.5}])
    matcher = LogisticMatcher(seed=42).fit(one_class, ["f1"])
    assert matcher.predict_proba(pd.DataFrame([{"f1": 0.0}, {"f1": 1.0}]), ["f1"]).tolist() == [1.0, 1.0]


def test_small_ablation_runner_schema_and_seed_metadata(tmp_path):
    source1 = pd.DataFrame([
        _record("S1-1", "alpha one", "1 main road"),
        _record("S1-2", "beta two", "2 main road"),
        _record("S1-3", "gamma three", "3 main road"),
    ])
    source2 = pd.DataFrame([
        _record("S2-1", "alpha one", "1 main road"),
        _record("S2-2", "beta two", "2 main road"),
        _record("S2-3", "unrelated", "99 other road"),
    ])
    source3 = pd.DataFrame([_record("S3-1", "gamma three", "3 main road")])
    processed = tmp_path / "processed"
    processed.mkdir()
    for name, frame in (("train_source1_sample.tsv", source1), ("train_source2_sample.tsv", source2), ("train_source3_sample.tsv", source3)):
        frame.to_csv(processed / name, sep="\t", index=False)
    truth_path = tmp_path / "truth.tsv"
    pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-2"},
        {"source1_entity_id": "S1-3", "matched_entity_ids": "S3-1"},
    ]).to_csv(truth_path, sep="\t", index=False)
    args = SimpleNamespace(
        processed_root=processed,
        source1_name="train_source1_sample.tsv",
        source2_name="train_source2_sample.tsv",
        source3_name="train_source3_sample.tsv",
        ground_truth=truth_path,
        validation_fraction=0.34,
        development_fraction=0.5,
        seeds="42",
    )
    per_seed, aggregate, thresholds, summary = run_ablation(args)
    assert len(per_seed) == 6
    assert set(per_seed["seed"]) == {42}
    assert set(per_seed["selection_uses_validation_labels"]) == {False}
    assert len(aggregate) == 6
    assert "validation_macro_f0_5_mean" in aggregate.columns
    assert {"development", "validation"}.issubset(set(thresholds["stage"]))
    assert not bool(summary.loc[0, "selection_uses_validation_labels"])
