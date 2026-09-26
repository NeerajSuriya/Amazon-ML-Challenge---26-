import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import (
    BLOCKING_RULES,
    CANDIDATE_ID,
    CANDIDATE_SOURCE,
    SOURCE1_ID,
    BlockingConfig,
    CandidateBlocker,
    candidate_pairs_by_s1,
    evaluate_blocking_rules,
    evaluate_candidate_pairs,
    evaluate_rule_contributions,
    summarize_candidate_distribution,
)


def _record(entity_id, name, address, country="india", source=None):
    source = source or entity_id[:2]
    return {
        "entity_id": entity_id,
        "name_norm": name,
        "address_norm": address,
        "country_norm": country,
        "name_tokens": json.dumps(name.split() if name else []),
        "address_tokens": json.dumps(address.split() if address else []),
        "source": source,
    }


def _frames():
    source1 = pd.DataFrame(
        [
            _record("S1-1", "acme trading", "10 main road", "india"),
            _record("S1-2", "", "", "india"),
            _record("S1-3", "duplicate name", "3 third street", "india"),
        ]
    )
    source2 = pd.DataFrame(
        [
            _record("S2-1", "acme trading", "10 main road", "india"),
            _record("S2-2", "acme trading", "9 other road", "us"),
            _record("S2-3", "different company", "10 main road", "india"),
            _record("S2-4", "duplicate name", "4 fourth street", "india"),
        ]
    )
    source3 = pd.DataFrame(
        [
            _record("S3-1", "acme holdings", "10 main street", "india"),
            _record("S3-2", "duplicate name", "5 fifth street", "india"),
        ]
    )
    return source1, source2, source3


def test_exact_name_country_blocking_excludes_incompatible_country():
    source1, source2, source3 = _frames()
    blocker = CandidateBlocker().fit(source1, source2, source3)
    candidates = blocker.generate(["country_name_exact"])
    pairs = set(zip(candidates[SOURCE1_ID], candidates[CANDIDATE_ID]))
    assert ("S1-1", "S2-1") in pairs
    assert ("S1-1", "S2-2") not in pairs


def test_token_and_address_blocking_find_plausible_non_exact_pairs():
    source1, source2, source3 = _frames()
    blocker = CandidateBlocker().fit(source1, source2, source3)
    name_token = blocker.generate(["name_token"])
    address_token = blocker.generate(["address_token"])
    assert ("S1-1", "S3-1") in set(zip(name_token[SOURCE1_ID], name_token[CANDIDATE_ID]))
    assert ("S1-1", "S2-3") in set(zip(address_token[SOURCE1_ID], address_token[CANDIDATE_ID]))


def test_combined_token_rule_and_union_provenance_are_deduplicated():
    source1, source2, source3 = _frames()
    blocker = CandidateBlocker().fit(source1, source2, source3)
    candidates = blocker.generate(["country_name_exact", "country_address_exact", "name_address_token"])
    pair_rows = candidates[
        (candidates[SOURCE1_ID] == "S1-1") & (candidates[CANDIDATE_ID] == "S2-1")
    ]
    assert len(pair_rows) == 1
    assert set(pair_rows.iloc[0][BLOCKING_RULES].split("|")) == {
        "country_name_exact",
        "country_address_exact",
        "name_address_token",
    }
    assert candidates.duplicated([SOURCE1_ID, CANDIDATE_ID]).sum() == 0


def test_empty_values_do_not_form_a_global_block():
    source1, source2, source3 = _frames()
    blocker = CandidateBlocker().fit(source1, source2, source3)
    candidates = blocker.generate()
    empty_s1_pairs = candidates[candidates[SOURCE1_ID] == "S1-2"]
    assert empty_s1_pairs.empty


def test_missing_country_is_compatible_but_known_mismatch_is_not():
    source1 = pd.DataFrame([_record("S1-1", "same name", "1 road", "")])
    source2 = pd.DataFrame(
        [
            _record("S2-1", "same name", "2 road", "india"),
            _record("S2-2", "same name", "2 road", "us"),
        ]
    )
    source3 = pd.DataFrame([_record("S3-1", "other", "3 road", "india")])
    candidates = CandidateBlocker().fit(source1, source2, source3).generate(
        ["country_name_exact"]
    )
    assert set(candidates[CANDIDATE_ID]) == {"S2-1", "S2-2"}


def test_duplicate_normalized_names_and_both_candidate_sources_are_retained():
    source1, source2, source3 = _frames()
    blocker = CandidateBlocker().fit(source1, source2, source3)
    candidates = blocker.generate(["country_name_exact"])
    duplicate_pairs = set(
        zip(
            candidates.loc[candidates[SOURCE1_ID] == "S1-3", SOURCE1_ID],
            candidates.loc[candidates[SOURCE1_ID] == "S1-3", CANDIDATE_ID],
        )
    )
    assert duplicate_pairs == {("S1-3", "S2-4"), ("S1-3", "S3-2")}
    assert set(candidates[CANDIDATE_SOURCE]) == {"S2", "S3"}


def test_recall_evaluation_handles_multiple_and_zero_true_matches():
    candidates = pd.DataFrame(
        [
            {SOURCE1_ID: "S1-1", CANDIDATE_ID: "S2-1"},
            {SOURCE1_ID: "S1-1", CANDIDATE_ID: "S3-1"},
            {SOURCE1_ID: "S1-1", CANDIDATE_ID: "S2-extra"},
        ]
    )
    truth = pd.DataFrame(
        [
            {SOURCE1_ID: "S1-1", "matched_entity_ids": "S2-1,S3-1"},
            {SOURCE1_ID: "S1-2", "matched_entity_ids": ""},
        ]
    )
    report = evaluate_candidate_pairs(candidates, truth, s1_ids=["S1-1", "S1-2"])
    assert report["number_s1_records"] == 2
    assert report["candidate_count"] == 3
    assert report["true_match_count"] == 2
    assert report["true_matches_retrieved"] == 2
    assert report["blocking_recall"] == 1.0
    assert report["s2_recall"] == 1.0
    assert report["s3_recall"] == 1.0
    assert report["average_candidates_per_s1"] == pytest.approx(1.5)
    assert report["median_candidates_per_s1"] == pytest.approx(1.5)
    assert report["maximum_candidates_per_s1"] == 3


def test_rule_evaluation_reports_each_rule_and_union():
    source1, source2, source3 = _frames()
    truth = pd.DataFrame(
        [
            {SOURCE1_ID: "S1-1", "matched_entity_ids": "S2-1,S3-1"},
            {SOURCE1_ID: "S1-2", "matched_entity_ids": ""},
            {SOURCE1_ID: "S1-3", "matched_entity_ids": "S2-4,S3-2"},
        ]
    )
    blocker = CandidateBlocker(
        BlockingConfig(name_token_max_frequency=10, address_token_max_frequency=10)
    ).fit(source1, source2, source3)
    report = evaluate_blocking_rules(blocker, truth)
    assert set(report.configuration) == set(blocker.available_rules) | {"union_default"}
    assert set(report.columns) >= {
        "candidate_count",
        "blocking_recall",
        "s2_recall",
        "s3_recall",
        "p95_candidates_per_s1",
        "p99_candidates_per_s1",
    }


def test_per_s1_diagnostics_include_zero_candidates_and_zero_truth_entities():
    candidates = pd.DataFrame(
        [
            {SOURCE1_ID: "S1-1", CANDIDATE_ID: "S2-1"},
            {SOURCE1_ID: "S1-3", CANDIDATE_ID: "S2-4"},
        ]
    )
    truth = pd.DataFrame(
        [
            {SOURCE1_ID: "S1-1", "matched_entity_ids": "S2-1,S3-1"},
            {SOURCE1_ID: "S1-2", "matched_entity_ids": ""},
            {SOURCE1_ID: "S1-3", "matched_entity_ids": "S2-4"},
        ]
    )
    per_s1 = candidate_pairs_by_s1(candidates, truth, s1_ids=["S1-1", "S1-2", "S1-3"])
    assert per_s1.set_index(SOURCE1_ID).loc["S1-2", "candidate_count"] == 0
    assert not per_s1.set_index(SOURCE1_ID).loc["S1-2", "has_ground_truth_match"]
    assert per_s1.set_index(SOURCE1_ID).loc["S1-1", "per_s1_blocking_recall"] == 0.5
    assert summarize_candidate_distribution(per_s1)["s1_with_0_candidates"] == 1
    metrics = evaluate_candidate_pairs(candidates, truth, s1_ids=["S1-1", "S1-2", "S1-3"])
    assert metrics["p95_candidates_per_s1"] == pytest.approx(1.0)
    assert metrics["matched_s1_count"] == 2
    assert metrics["zero_match_s1_count"] == 1
    assert metrics["zero_match_s1_with_0_candidates"] == 1


def test_multi_token_overlap_rule_requires_stronger_evidence():
    source1, source2, source3 = _frames()
    blocker = CandidateBlocker().fit(source1, source2, source3)
    candidates = blocker.generate(["name_token_overlap_2"])
    pairs = set(zip(candidates[SOURCE1_ID], candidates[CANDIDATE_ID]))
    assert ("S1-1", "S2-1") in pairs
    assert ("S1-1", "S3-1") not in pairs
    assert candidates.duplicated([SOURCE1_ID, CANDIDATE_ID]).sum() == 0


def test_rule_contribution_report_separates_unique_and_overlapping_pairs():
    source1, source2, source3 = _frames()
    truth = pd.DataFrame(
        [
            {SOURCE1_ID: "S1-1", "matched_entity_ids": "S2-1,S3-1"},
            {SOURCE1_ID: "S1-2", "matched_entity_ids": ""},
            {SOURCE1_ID: "S1-3", "matched_entity_ids": "S2-4,S3-2"},
        ]
    )
    blocker = CandidateBlocker().fit(source1, source2, source3)
    report = evaluate_rule_contributions(blocker, truth)
    assert set(report["rule"]) == set(blocker.available_rules)
    name_row = report.set_index("rule").loc["country_name_exact"]
    assert name_row["candidate_count"] >= name_row["unique_candidate_count"]
    assert name_row["true_links_retrieved"] >= name_row["unique_true_links_retrieved"]
    assert report["overlap_candidate_count"].ge(0).all()


def test_candidate_writer_emits_identity_columns_without_duplicates(tmp_path):
    import subprocess

    source1, source2, source3 = _frames()
    paths = []
    for name, frame in zip(("s1.tsv", "s2.tsv", "s3.tsv"), (source1, source2, source3)):
        path = tmp_path / name
        frame.to_csv(path, sep="\t", index=False)
        paths.append(path)
    output = tmp_path / "candidate_pairs.tsv"
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "generate_candidate_pairs.py"),
            "--source1",
            str(paths[0]),
            "--source2",
            str(paths[1]),
            "--source3",
            str(paths[2]),
            "--output",
            str(output),
        ],
        check=True,
        cwd=project_root,
    )
    written = pd.read_csv(output, sep="\t", dtype=str)
    assert list(written.columns) == [SOURCE1_ID, CANDIDATE_ID]
    assert written.duplicated([SOURCE1_ID, CANDIDATE_ID]).sum() == 0
    assert written[CANDIDATE_ID].str.startswith(("S2-", "S3-"), na=False).all()
