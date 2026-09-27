"""Submission aggregation and validation for zero-to-many entity matching.

The blocker and matcher operate on one row per candidate identity.  The
challenge output is one row per Source-1 entity, so this module is the single
boundary between those internal pair tables and the final TSV contract.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

SOURCE1_ID = "source1_entity_id"
CANDIDATE_ID = "candidate_entity_id"
CANDIDATE_LIST = "candidate_entity_ids"
MATCHED_LIST = "matched_entity_ids"
CANDIDATE_OUTPUT_COLUMNS = [SOURCE1_ID, CANDIDATE_LIST]
MATCH_OUTPUT_COLUMNS = [SOURCE1_ID, MATCHED_LIST]
PAIR_COLUMNS = [SOURCE1_ID, CANDIDATE_ID]


def _text(value) -> str:
    if value is None:
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return ""
    return str(value).strip()


def parse_id_list(value) -> list[str]:
    """Parse a comma-separated output cell, retaining duplicates for checks."""
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        text = _text(value)
        if not text:
            return []
        values = text.split(",")
    return [_text(item) for item in values]


def _unique_ids(values: Iterable[str], *, label: str) -> list[str]:
    result = []
    seen = set()
    for value in values:
        entity_id = _text(value)
        if not entity_id:
            raise ValueError(f"{label} contains an empty entity ID")
        if entity_id in seen:
            raise ValueError(f"{label} contains duplicate entity ID: {entity_id}")
        seen.add(entity_id)
        result.append(entity_id)
    return result


def _source1_ids(values: Iterable[str]) -> list[str]:
    ids = _unique_ids(values, label="source1 IDs")
    if not all(value.startswith("S1-") for value in ids):
        raise ValueError("source1 IDs must use the S1- namespace")
    return ids


def _validate_pair_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = set(PAIR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
    out = frame.loc[:, PAIR_COLUMNS].copy()
    out[SOURCE1_ID] = out[SOURCE1_ID].map(_text)
    out[CANDIDATE_ID] = out[CANDIDATE_ID].map(_text)
    if out[PAIR_COLUMNS].eq("").any().any():
        raise ValueError(f"{label} contains empty pair IDs")
    if not out[SOURCE1_ID].str.startswith("S1-").all():
        raise ValueError(f"{label} contains a source1 ID outside S1-")
    if not out[CANDIDATE_ID].str.startswith(("S2-", "S3-"), na=False).all():
        raise ValueError(f"{label} contains a candidate outside S2-/S3-")
    if out.duplicated(PAIR_COLUMNS).any():
        raise ValueError(f"{label} contains duplicate candidate identities")
    return out


def _known_candidate_ids(
    source2_ids: Iterable[str] | None, source3_ids: Iterable[str] | None
) -> set[str] | None:
    if source2_ids is None and source3_ids is None:
        return None
    ids = _unique_ids(
        [*([] if source2_ids is None else source2_ids), *([] if source3_ids is None else source3_ids)],
        label="candidate source IDs",
    )
    if not all(value.startswith(("S2-", "S3-")) for value in ids):
        raise ValueError("candidate source IDs must use S2-/S3- namespaces")
    return set(ids)


def aggregate_candidate_pairs(
    candidates: pd.DataFrame,
    source1_ids: Iterable[str],
    *,
    source2_ids: Iterable[str] | None = None,
    source3_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aggregate final one-row-per-pair candidates into the challenge schema."""
    s1_ids = _source1_ids(source1_ids)
    pairs = _validate_pair_frame(candidates, "candidate pairs")
    s1_set = set(s1_ids)
    unknown_s1 = set(pairs[SOURCE1_ID]) - s1_set
    if unknown_s1:
        raise ValueError(f"candidate pairs contain unknown source1 IDs: {sorted(unknown_s1)}")
    known_candidates = _known_candidate_ids(source2_ids, source3_ids)
    if known_candidates is not None:
        unknown = set(pairs[CANDIDATE_ID]) - known_candidates
        if unknown:
            raise ValueError(f"candidate pairs contain unknown candidate IDs: {sorted(unknown)}")

    grouped = pairs.groupby(SOURCE1_ID, sort=False)[CANDIDATE_ID].agg(list).to_dict()
    rows = [
        {
            SOURCE1_ID: s1_id,
            CANDIDATE_LIST: ",".join(sorted(grouped.get(s1_id, []))),
        }
        for s1_id in sorted(s1_ids)
    ]
    return pd.DataFrame(rows, columns=CANDIDATE_OUTPUT_COLUMNS)


def aggregate_matches(
    matches: pd.DataFrame,
    source1_ids: Iterable[str],
    *,
    candidate_pairs: pd.DataFrame | None = None,
    source2_ids: Iterable[str] | None = None,
    source3_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aggregate thresholded pair matches into one row per Source-1 entity."""
    s1_ids = _source1_ids(source1_ids)
    pair_matches = _validate_pair_frame(matches, "matched pairs")
    s1_set = set(s1_ids)
    unknown_s1 = set(pair_matches[SOURCE1_ID]) - s1_set
    if unknown_s1:
        raise ValueError(f"matched pairs contain unknown source1 IDs: {sorted(unknown_s1)}")
    known_candidates = _known_candidate_ids(source2_ids, source3_ids)
    if known_candidates is not None:
        unknown = set(pair_matches[CANDIDATE_ID]) - known_candidates
        if unknown:
            raise ValueError(f"matched pairs contain unknown candidate IDs: {sorted(unknown)}")
    if candidate_pairs is not None:
        pairs = _validate_pair_frame(candidate_pairs, "candidate pairs")
        allowed = set(pairs.itertuples(index=False, name=None))
        selected = set(pair_matches.itertuples(index=False, name=None))
        if not selected <= allowed:
            raise ValueError("matched pairs are not a subset of candidate pairs")

    grouped = pair_matches.groupby(SOURCE1_ID, sort=False)[CANDIDATE_ID].agg(list).to_dict()
    rows = [
        {
            SOURCE1_ID: s1_id,
            MATCHED_LIST: ",".join(sorted(grouped.get(s1_id, []))),
        }
        for s1_id in sorted(s1_ids)
    ]
    return pd.DataFrame(rows, columns=MATCH_OUTPUT_COLUMNS)


def validate_submission_frames(
    candidate_output: pd.DataFrame,
    matching_output: pd.DataFrame,
    *,
    test_source1_ids: Iterable[str],
    test_source2_ids: Iterable[str],
    test_source3_ids: Iterable[str],
) -> dict[str, int]:
    """Validate final aggregate outputs against the test entity universe."""
    expected_s1 = _source1_ids(test_source1_ids)
    expected_s1_set = set(expected_s1)
    valid_candidates = _known_candidate_ids(test_source2_ids, test_source3_ids) or set()

    for frame, expected, label in (
        (candidate_output, CANDIDATE_OUTPUT_COLUMNS, "candidate output"),
        (matching_output, MATCH_OUTPUT_COLUMNS, "matching output"),
    ):
        if list(frame.columns) != expected:
            raise ValueError(f"{label} columns must be exactly {expected}")
        if len(frame) != len(expected_s1):
            raise ValueError(
                f"{label} must contain exactly one row per test S1: "
                f"expected {len(expected_s1)}, got {len(frame)}"
            )
        ids = frame[SOURCE1_ID].map(_text)
        if ids.duplicated().any():
            raise ValueError(f"{label} contains duplicate source1 rows")
        if set(ids) != expected_s1_set:
            raise ValueError(f"{label} does not contain exactly the test S1 IDs")

    candidate_by_s1: dict[str, set[str]] = {}
    match_by_s1: dict[str, set[str]] = {}
    for row in candidate_output.itertuples(index=False):
        s1_id = _text(row[0])
        ids = parse_id_list(row[1])
        if len(ids) != len(set(ids)):
            raise ValueError(f"candidate output contains duplicate IDs for {s1_id}")
        if not all(entity_id.startswith(("S2-", "S3-")) for entity_id in ids):
            raise ValueError(f"candidate output contains an invalid namespace for {s1_id}")
        unknown = set(ids) - valid_candidates
        if unknown:
            raise ValueError(f"candidate output contains unknown candidate IDs: {sorted(unknown)}")
        candidate_by_s1[s1_id] = set(ids)

    for row in matching_output.itertuples(index=False):
        s1_id = _text(row[0])
        ids = parse_id_list(row[1])
        if len(ids) != len(set(ids)):
            raise ValueError(f"matching output contains duplicate IDs for {s1_id}")
        if not all(entity_id.startswith(("S2-", "S3-")) for entity_id in ids):
            raise ValueError(f"matching output contains an invalid namespace for {s1_id}")
        unknown = set(ids) - valid_candidates
        if unknown:
            raise ValueError(f"matching output contains unknown candidate IDs: {sorted(unknown)}")
        match_by_s1[s1_id] = set(ids)

    for s1_id, matches in match_by_s1.items():
        if not matches <= candidate_by_s1[s1_id]:
            raise ValueError(f"matching IDs for {s1_id} are not a subset of candidates")

    candidate_count = sum(len(ids) for ids in candidate_by_s1.values())
    match_count = sum(len(ids) for ids in match_by_s1.values())
    return {
        "s1_count": len(expected_s1),
        "candidate_pair_count": candidate_count,
        "predicted_match_count": match_count,
        "zero_candidate_s1_count": sum(not ids for ids in candidate_by_s1.values()),
        "zero_match_s1_count": sum(not ids for ids in match_by_s1.values()),
    }


__all__ = [
    "CANDIDATE_ID",
    "CANDIDATE_LIST",
    "CANDIDATE_OUTPUT_COLUMNS",
    "MATCHED_LIST",
    "MATCH_OUTPUT_COLUMNS",
    "PAIR_COLUMNS",
    "SOURCE1_ID",
    "aggregate_candidate_pairs",
    "aggregate_matches",
    "parse_id_list",
    "validate_submission_frames",
]
