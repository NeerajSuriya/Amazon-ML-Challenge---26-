"""Blocking-agnostic pair features for business entity resolution.

The module consumes externally generated candidate pairs and processed entity
records. It never creates candidates itself:

    candidate pairs -> fit records -> pair feature matrix -> labels

Processed TSV list columns are JSON strings; they are parsed at the boundary so
all comparisons operate on Python lists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

ID_COL = "entity_id"
SOURCE1_ID = "source1_entity_id"
CANDIDATE_ID = "candidate_entity_id"
SOURCE1_SOURCE = "source1_source"
CANDIDATE_SOURCE = "candidate_source"

JSON_LIST_COLUMNS = {
    "name_tokens",
    "address_tokens",
    "name_char_ngrams",
    "address_char_ngrams",
}
REQUIRED_RECORD_COLUMNS = {ID_COL, "name_norm", "address_norm", "country_norm"}
REQUIRED_CANDIDATE_COLUMNS = {SOURCE1_ID, CANDIDATE_ID}
REQUIRED_TRUTH_COLUMNS = {SOURCE1_ID, "matched_entity_ids"}


def _is_missing(value) -> bool:
    """Return whether a scalar value is missing without treating lists as scalars."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _as_text(value) -> str:
    """Convert a scalar identifier/text value to a stable non-NA string."""
    return "" if _is_missing(value) else str(value).strip()


def parse_json_list(value) -> list[str]:
    """Parse one preprocessing JSON-list value into a list of strings.

    In-memory lists are accepted for convenience in tests and notebooks, while
    persisted TSV values are parsed with :func:`json.loads`.
    """
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if not _is_missing(x)]
    if _is_missing(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got {type(parsed).__name__}: {text[:80]}")
    return [str(x) for x in parsed if not _is_missing(x)]


def _parse_matched_ids(value) -> list[str]:
    """Parse a comma-separated zero-to-many ground-truth cell."""
    if isinstance(value, (list, tuple, set)):
        values = value
    elif _is_missing(value):
        return []
    else:
        values = str(value).split(",")

    result = []
    seen = set()
    for value in values:
        entity_id = _as_text(value)
        if entity_id and entity_id not in seen:
            result.append(entity_id)
            seen.add(entity_id)
    return result


def _set_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _overlap_coefficient(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def _token_set_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def _token_sort_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def _length_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))


def _ensure_parsed(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy a processed record frame, parsing JSON list columns."""
    missing = REQUIRED_RECORD_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing record columns: {sorted(missing)}")

    out = df.copy()
    out[ID_COL] = out[ID_COL].map(_as_text)
    if out[ID_COL].eq("").any():
        raise ValueError("Record entity_id values must be non-empty")
    for col in ("name_norm", "address_norm", "country_norm"):
        out[col] = out[col].map(_as_text)
    for col in JSON_LIST_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Missing processed JSON list column: {col}")
        out[col] = out[col].map(parse_json_list)
    if "source" not in out.columns:
        out["source"] = out[ID_COL].str.extract(r"^(S[123])-", expand=False).fillna("")
    else:
        out["source"] = out["source"].map(_as_text)
    return out


def _validate_ground_truth(ground_truth: pd.DataFrame) -> None:
    missing = REQUIRED_TRUTH_COLUMNS - set(ground_truth.columns)
    if missing:
        raise ValueError(f"Missing ground-truth columns: {sorted(missing)}")


def _validate_candidates(candidates: pd.DataFrame) -> None:
    missing = REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")


@dataclass
class TfidfViews:
    """TF-IDF matrices aligned to a combined record table."""

    name: object
    address: object


class PairFeatureGenerator:
    """Generate pairwise name, address, and country comparison features."""

    def __init__(self, *, analyzer: str = "char", ngram_range=(3, 5), min_df=1):
        self.vectorizer_name = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=False,
            norm="l2",
        )
        self.vectorizer_address = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=False,
            norm="l2",
        )
        self._records = None
        self._name_matrix = None
        self._address_matrix = None
        self._row: dict[str, int] = {}

    @staticmethod
    def _fit_vectorizer(vectorizer, values: list[str]):
        """Fit TF-IDF, returning None when the field has no usable vocabulary."""
        try:
            return vectorizer.fit_transform(values)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
            return None

    def fit(self, *frames: pd.DataFrame) -> "PairFeatureGenerator":
        if not frames:
            raise ValueError("At least one processed record frame is required")
        records = pd.concat([_ensure_parsed(frame) for frame in frames], ignore_index=True)
        if records[ID_COL].duplicated().any():
            duplicate = records.loc[records[ID_COL].duplicated(), ID_COL].iloc[0]
            raise ValueError(f"Duplicate entity_id: {duplicate}")

        self._records = records
        self._row = {entity_id: i for i, entity_id in enumerate(records[ID_COL])}
        names = records["name_norm"].tolist()
        addresses = records["address_norm"].tolist()
        self._name_matrix = self._fit_vectorizer(self.vectorizer_name, names)
        self._address_matrix = self._fit_vectorizer(self.vectorizer_address, addresses)
        return self

    @property
    def feature_names(self) -> list[str]:
        return [
            "name_exact",
            "name_ratio",
            "name_token_set_ratio",
            "name_token_sort_ratio",
            "name_jaccard",
            "name_overlap",
            "name_char_ngram_jaccard",
            "name_tfidf_cosine",
            "address_exact",
            "address_ratio",
            "address_token_set_ratio",
            "address_token_sort_ratio",
            "address_jaccard",
            "address_overlap",
            "address_char_ngram_jaccard",
            "address_tfidf_cosine",
            "country_exact",
            "country_nonempty_both",
            "name_length_ratio",
            "address_length_ratio",
        ]

    def _tfidf_sim(
        self,
        matrix,
        a_id: str,
        b_id: str,
        a_text: str,
        b_text: str,
    ) -> float:
        if matrix is None:
            # Cosine is undefined without a vocabulary; use the same neutral
            # empty/non-empty convention as the other text similarities.
            return _ratio(a_text, b_text)
        ia, ib = self._row.get(a_id), self._row.get(b_id)
        if ia is None or ib is None:
            raise KeyError(f"Unknown entity ID in pair: {a_id}, {b_id}")
        return float(matrix[ia].multiply(matrix[ib]).sum())

    def pair_features(self, a: pd.Series, b: pd.Series) -> dict[str, float]:
        name_a, name_b = _as_text(a["name_norm"]), _as_text(b["name_norm"])
        address_a, address_b = _as_text(a["address_norm"]), _as_text(b["address_norm"])
        country_a, country_b = _as_text(a["country_norm"]), _as_text(b["country_norm"])
        name_tokens_a, name_tokens_b = a["name_tokens"], b["name_tokens"]
        address_tokens_a, address_tokens_b = a["address_tokens"], b["address_tokens"]
        name_ngrams_a, name_ngrams_b = a["name_char_ngrams"], b["name_char_ngrams"]
        address_ngrams_a, address_ngrams_b = a["address_char_ngrams"], b["address_char_ngrams"]
        a_id, b_id = _as_text(a[ID_COL]), _as_text(b[ID_COL])

        return {
            "name_exact": float(bool(name_a) and name_a == name_b),
            "name_ratio": _ratio(name_a, name_b),
            "name_token_set_ratio": _token_set_ratio(name_a, name_b),
            "name_token_sort_ratio": _token_sort_ratio(name_a, name_b),
            "name_jaccard": _set_jaccard(name_tokens_a, name_tokens_b),
            "name_overlap": _overlap_coefficient(name_tokens_a, name_tokens_b),
            "name_char_ngram_jaccard": _set_jaccard(name_ngrams_a, name_ngrams_b),
            "name_tfidf_cosine": self._tfidf_sim(
                self._name_matrix, a_id, b_id, name_a, name_b
            ),
            "address_exact": float(bool(address_a) and address_a == address_b),
            "address_ratio": _ratio(address_a, address_b),
            "address_token_set_ratio": _token_set_ratio(address_a, address_b),
            "address_token_sort_ratio": _token_sort_ratio(address_a, address_b),
            "address_jaccard": _set_jaccard(address_tokens_a, address_tokens_b),
            "address_overlap": _overlap_coefficient(address_tokens_a, address_tokens_b),
            "address_char_ngram_jaccard": _set_jaccard(
                address_ngrams_a, address_ngrams_b
            ),
            "address_tfidf_cosine": self._tfidf_sim(
                self._address_matrix, a_id, b_id, address_a, address_b
            ),
            "country_exact": float(bool(country_a) and country_a == country_b),
            "country_nonempty_both": float(bool(country_a) and bool(country_b)),
            "name_length_ratio": _length_ratio(name_a, name_b),
            "address_length_ratio": _length_ratio(address_a, address_b),
        }

    def transform(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Transform externally generated ``source1_entity_id,candidate_entity_id`` pairs."""
        if self._records is None:
            raise RuntimeError("Call fit() before transform().")
        _validate_candidates(candidates)

        output_columns = [
            SOURCE1_ID,
            CANDIDATE_ID,
            SOURCE1_SOURCE,
            CANDIDATE_SOURCE,
            *self.feature_names,
        ]
        if candidates.empty:
            return pd.DataFrame(columns=output_columns)

        by_id = self._records.set_index(ID_COL, drop=False)
        rows = []
        for s1_id, candidate_id in candidates[[SOURCE1_ID, CANDIDATE_ID]].itertuples(
            index=False, name=None
        ):
            s1_id, candidate_id = _as_text(s1_id), _as_text(candidate_id)
            if s1_id not in by_id.index or candidate_id not in by_id.index:
                raise KeyError(f"Candidate contains unknown ID: {s1_id}, {candidate_id}")
            s1 = by_id.loc[s1_id]
            candidate = by_id.loc[candidate_id]
            if _as_text(s1["source"]) != "S1" or _as_text(candidate["source"]) not in {
                "S2",
                "S3",
            }:
                raise ValueError(f"Invalid cross-source pair: {s1_id}, {candidate_id}")
            features = self.pair_features(s1, candidate)
            rows.append(
                {
                    SOURCE1_ID: s1_id,
                    CANDIDATE_ID: candidate_id,
                    SOURCE1_SOURCE: _as_text(s1["source"]),
                    CANDIDATE_SOURCE: _as_text(candidate["source"]),
                    **features,
                }
            )
        return pd.DataFrame(rows, columns=output_columns)


def build_candidate_pairs_from_ground_truth(
    ground_truth: pd.DataFrame,
    *,
    include_positive_only: bool = True,
) -> pd.DataFrame:
    """Expand one-row-per-S1 ground truth into one row per positive pair.

    The ground truth cannot provide negative candidates without a candidate
    universe, so ``include_positive_only`` is retained for API compatibility
    and positive rows are always the only rows emitted.
    """
    _validate_ground_truth(ground_truth)
    rows = []
    for s1_id, matched in ground_truth[[SOURCE1_ID, "matched_entity_ids"]].itertuples(
        index=False, name=None
    ):
        for candidate_id in _parse_matched_ids(matched):
            rows.append(
                {
                    SOURCE1_ID: _as_text(s1_id),
                    CANDIDATE_ID: candidate_id,
                    "label": 1,
                }
            )
    return pd.DataFrame(rows, columns=[SOURCE1_ID, CANDIDATE_ID, "label"])


def attach_ground_truth_labels(
    candidates: pd.DataFrame, ground_truth: pd.DataFrame
) -> pd.DataFrame:
    """Label arbitrary blocking candidates against zero-to-many ground truth."""
    _validate_candidates(candidates)
    _validate_ground_truth(ground_truth)

    positives = set()
    for s1_id, matched in ground_truth[[SOURCE1_ID, "matched_entity_ids"]].itertuples(
        index=False, name=None
    ):
        positives.update((_as_text(s1_id), candidate_id) for candidate_id in _parse_matched_ids(matched))

    out = candidates.copy()
    out[SOURCE1_ID] = out[SOURCE1_ID].map(_as_text)
    out[CANDIDATE_ID] = out[CANDIDATE_ID].map(_as_text)
    out["label"] = [
        int((s1_id, candidate_id) in positives)
        for s1_id, candidate_id in out[[SOURCE1_ID, CANDIDATE_ID]].itertuples(
            index=False, name=None
        )
    ]
    return out


def make_labeled_features(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Fit feature views and return features plus aligned labels."""
    generator = PairFeatureGenerator().fit(source1, source2, source3)
    features = generator.transform(candidates)
    labels = attach_ground_truth_labels(candidates, ground_truth)
    if len(features) != len(labels):
        raise RuntimeError("Feature and label row counts differ")
    features["label"] = labels["label"].to_numpy()
    return features


def split_ground_truth_by_s1(
    ground_truth: pd.DataFrame,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically split ground-truth rows without splitting an S1 entity."""
    _validate_ground_truth(ground_truth)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    s1_ids = ground_truth[SOURCE1_ID].map(_as_text).drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(s1_ids)
    n_validation = max(1, int(round(len(s1_ids) * validation_fraction))) if len(s1_ids) else 0
    validation_ids = set(s1_ids[:n_validation])
    normalized_ids = ground_truth[SOURCE1_ID].map(_as_text)
    train = ground_truth.loc[~normalized_ids.isin(validation_ids)].copy()
    validation = ground_truth.loc[normalized_ids.isin(validation_ids)].copy()
    return train, validation


def ground_truth_summary(ground_truth: pd.DataFrame) -> dict[str, int]:
    """Return counts useful for split documentation and experiment logs."""
    _validate_ground_truth(ground_truth)
    s1_ids = ground_truth[SOURCE1_ID].map(_as_text)
    link_lists = [_parse_matched_ids(value) for value in ground_truth["matched_entity_ids"]]
    return {
        "s1_entities": int(s1_ids.nunique()),
        "ground_truth_rows": int(len(ground_truth)),
        "empty_truth_entities": int(sum(not ids for ids in link_lists)),
        "positive_links": int(sum(len(ids) for ids in link_lists)),
        "s2_links": int(sum(entity_id.startswith("S2-") for ids in link_lists for entity_id in ids)),
        "s3_links": int(sum(entity_id.startswith("S3-") for ids in link_lists for entity_id in ids)),
    }
