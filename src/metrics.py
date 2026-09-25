"""Entity-level evaluation helpers for the business entity-resolution task.

The challenge score is macro F0.5 over Source-1 entities. Each S1 is scored
against sets of predicted and true S2/S3 IDs, then those entity scores are
averaged; pair-level global F0.5 is not used here.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

SOURCE1_ID = "source1_entity_id"
CANDIDATE_ID = "candidate_entity_id"
REQUIRED_TRUTH_COLUMNS = {SOURCE1_ID, "matched_entity_ids"}


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _as_text(value) -> str:
    return "" if _is_missing(value) else str(value).strip()


def _parse_ids(value) -> list[str]:
    """Parse a comma-separated or iterable ID collection, dropping blanks."""
    if isinstance(value, (list, tuple, set)):
        values = value
    elif _is_missing(value):
        return []
    else:
        values = str(value).split(",")
    return [_as_text(item) for item in values if _as_text(item)]


def _validate_truth(ground_truth: pd.DataFrame) -> None:
    missing = REQUIRED_TRUTH_COLUMNS - set(ground_truth.columns)
    if missing:
        raise ValueError(f"Missing ground-truth columns: {sorted(missing)}")


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F-beta from precision and recall."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    return (1 + beta_squared) * precision * recall / denominator if denominator else 0.0


def entity_f05(predicted: Iterable[str], truth: Iterable[str]) -> float:
    """Compute one entity's set-based F0.5 score."""
    predicted_ids = {_as_text(value) for value in predicted if _as_text(value)}
    truth_ids = {_as_text(value) for value in truth if _as_text(value)}
    if not predicted_ids and not truth_ids:
        return 1.0
    if not predicted_ids or not truth_ids:
        return 0.0
    true_positives = len(predicted_ids & truth_ids)
    precision = true_positives / len(predicted_ids)
    recall = true_positives / len(truth_ids)
    return fbeta(precision, recall, beta=0.5)


def macro_f05(
    predictions: Mapping[str, Iterable[str]],
    ground_truth: pd.DataFrame,
) -> float:
    """Macro-average entity-level F0.5 over every ground-truth S1.

    Missing prediction keys represent empty predictions. If malformed data has
    repeated S1 rows, their truth IDs are unioned so that an S1 is still scored
    exactly once rather than receiving extra weight.
    """
    _validate_truth(ground_truth)
    truth_by_s1: dict[str, set[str]] = {}
    for s1_id, matched in ground_truth[[SOURCE1_ID, "matched_entity_ids"]].itertuples(
        index=False, name=None
    ):
        key = _as_text(s1_id)
        truth_by_s1.setdefault(key, set()).update(_parse_ids(matched))

    if not truth_by_s1:
        return 0.0
    scores = [
        entity_f05(predictions.get(s1_id, []), truth_ids)
        for s1_id, truth_ids in truth_by_s1.items()
    ]
    return sum(scores) / len(scores)


def predictions_from_scored_pairs(
    scored_pairs: pd.DataFrame,
    threshold: float,
    score_col: str = "score",
    s1_col: str = SOURCE1_ID,
    candidate_col: str = CANDIDATE_ID,
) -> dict[str, list[str]]:
    """Threshold pair scores into an S1 -> candidate-ID mapping."""
    required = {s1_col, candidate_col, score_col}
    missing = required - set(scored_pairs.columns)
    if missing:
        raise ValueError(f"Missing scored-pair columns: {sorted(missing)}")
    predictions: dict[str, list[str]] = {}
    for s1_id, group in scored_pairs.groupby(s1_col, sort=False, dropna=False):
        selected = group.loc[group[score_col] >= threshold, candidate_col]
        ids = []
        seen = set()
        for candidate_id in selected:
            candidate_id = _as_text(candidate_id)
            if candidate_id and candidate_id not in seen:
                ids.append(candidate_id)
                seen.add(candidate_id)
        predictions[_as_text(s1_id)] = ids
    return predictions


def score_thresholds(
    scored_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    thresholds: Iterable[float],
    score_col: str = "score",
) -> pd.DataFrame:
    """Evaluate a documented collection of score thresholds."""
    rows = []
    for threshold in thresholds:
        predictions = predictions_from_scored_pairs(scored_pairs, threshold, score_col)
        rows.append(
            {
                "threshold": float(threshold),
                "macro_f0_5": macro_f05(predictions, ground_truth),
            }
        )
    return pd.DataFrame(rows)
