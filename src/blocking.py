"""Candidate generation for business entity resolution.

Blocking reduces the S1 x (S2 + S3) comparison space. It deliberately does
not calculate pair similarity features or decide whether a pair is a match.
Each output row has one unique ``(source1_entity_id, candidate_entity_id)``
identity and records the independent rules that generated it.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .pair_features import parse_json_list

SOURCE1_ID = "source1_entity_id"
CANDIDATE_ID = "candidate_entity_id"
CANDIDATE_SOURCE = "candidate_source"
BLOCKING_RULES = "blocking_rules"

REQUIRED_RECORD_COLUMNS = {
    "entity_id",
    "name_norm",
    "address_norm",
    "country_norm",
    "name_tokens",
    "address_tokens",
}
REQUIRED_TRUTH_COLUMNS = {SOURCE1_ID, "matched_entity_ids"}
OUTPUT_COLUMNS = [SOURCE1_ID, CANDIDATE_ID, CANDIDATE_SOURCE, BLOCKING_RULES]
DEFAULT_RULE_NAMES = (
    "country_name_exact",
    "country_address_exact",
    "name_token",
    "address_token",
    "name_address_token",
)
EXPERIMENTAL_RULE_NAMES = (
    "name_token_overlap_2",
    "name_token_overlap_3",
    "address_token_overlap_2",
    "address_token_overlap_3",
)
RULE_NAMES = DEFAULT_RULE_NAMES + EXPERIMENTAL_RULE_NAMES


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


def _parse_truth_ids(value) -> list[str]:
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


def _validate_truth(ground_truth: pd.DataFrame) -> None:
    missing = REQUIRED_TRUTH_COLUMNS - set(ground_truth.columns)
    if missing:
        raise ValueError(f"Missing ground-truth columns: {sorted(missing)}")


def _prepare_records(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_RECORD_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing processed record columns: {sorted(missing)}")

    out = frame.copy()
    out["entity_id"] = out["entity_id"].map(_as_text)
    if out["entity_id"].eq("").any():
        raise ValueError("Processed entity_id values must be non-empty")
    for column in ("name_norm", "address_norm", "country_norm"):
        out[column] = out[column].map(_as_text)
    for column in ("name_tokens", "address_tokens"):
        out[column] = out[column].map(parse_json_list)
    if "source" not in out.columns:
        out["source"] = out["entity_id"].str.extract(r"^(S[123])-", expand=False).fillna("")
    else:
        out["source"] = out["source"].map(_as_text)
    return out


def _country_compatible(left: str, right: str) -> bool:
    """Allow unknown countries, rejecting only known unequal countries."""
    return not left or not right or left == right


@dataclass(frozen=True)
class BlockingConfig:
    """Deterministic limits used to keep generic token blocks bounded."""

    name_token_max_frequency: int = 50
    address_token_max_frequency: int = 50
    max_exact_bucket_size: int = 500
    default_rules: tuple[str, ...] = field(default=DEFAULT_RULE_NAMES)

    def __post_init__(self) -> None:
        if self.name_token_max_frequency < 1:
            raise ValueError("name_token_max_frequency must be at least 1")
        if self.address_token_max_frequency < 1:
            raise ValueError("address_token_max_frequency must be at least 1")
        if self.max_exact_bucket_size < 1:
            raise ValueError("max_exact_bucket_size must be at least 1")
        unknown = set(self.default_rules) - set(RULE_NAMES)
        if unknown:
            raise ValueError(f"Unknown default blocking rules: {sorted(unknown)}")


class CandidateBlocker:
    """Fit blocking indexes and generate country-compatible candidate pairs."""

    def __init__(self, config: BlockingConfig | None = None):
        self.config = config or BlockingConfig()
        self._source1: pd.DataFrame | None = None
        self._candidates: pd.DataFrame | None = None
        self._candidate_by_id: dict[str, dict] = {}
        self._name_index: dict[str, list[str]] = {}
        self._address_index: dict[str, list[str]] = {}
        self._name_token_index: dict[str, list[str]] = {}
        self._address_token_index: dict[str, list[str]] = {}
        self.name_token_frequency: Counter[str] = Counter()
        self.address_token_frequency: Counter[str] = Counter()

    @property
    def s1_ids(self) -> tuple[str, ...]:
        self._require_fit()
        return tuple(self._source1["entity_id"])

    @property
    def available_rules(self) -> tuple[str, ...]:
        """Return the default rules used in rule-by-rule reports.

        Experimental rules remain opt-in through ``generate(rule_names=...)`` so
        adding diagnostics does not change the default candidate union.
        """
        return DEFAULT_RULE_NAMES

    def _require_fit(self) -> None:
        if self._source1 is None or self._candidates is None:
            raise RuntimeError("Call fit() before generating candidates")

    @staticmethod
    def _build_index(records: pd.DataFrame, column: str) -> dict[str, list[str]]:
        index: defaultdict[str, list[str]] = defaultdict(list)
        for row in records[["entity_id", column]].itertuples(index=False, name=None):
            entity_id, key = row
            if key:
                index[key].append(entity_id)
        return dict(index)

    @staticmethod
    def _build_token_index(
        records: pd.DataFrame, column: str, frequencies: Counter[str], max_frequency: int
    ) -> dict[str, list[str]]:
        index: defaultdict[str, list[str]] = defaultdict(list)
        for entity_id, tokens in records[["entity_id", column]].itertuples(
            index=False, name=None
        ):
            for token in set(tokens):
                if token and frequencies[token] <= max_frequency:
                    index[token].append(entity_id)
        return dict(index)

    def fit(
        self,
        source1: pd.DataFrame,
        source2: pd.DataFrame,
        source3: pd.DataFrame,
    ) -> "CandidateBlocker":
        """Build indexes from processed S1, S2, and S3 frames."""
        prepared = [_prepare_records(frame) for frame in (source1, source2, source3)]
        if any(frame["entity_id"].duplicated().any() for frame in prepared):
            raise ValueError("Each source frame must have unique entity_id values")
        all_ids = pd.concat([frame["entity_id"] for frame in prepared], ignore_index=True)
        if all_ids.duplicated().any():
            duplicate = all_ids[all_ids.duplicated()].iloc[0]
            raise ValueError(f"Duplicate entity_id across source frames: {duplicate}")

        if not set(prepared[0]["source"]) <= {"S1"}:
            raise ValueError("source1 contains records without source S1")
        if not set(prepared[1]["source"]) <= {"S2"}:
            raise ValueError("source2 contains records without source S2")
        if not set(prepared[2]["source"]) <= {"S3"}:
            raise ValueError("source3 contains records without source S3")

        candidates = pd.concat(prepared[1:], ignore_index=True)
        self._source1 = prepared[0].reset_index(drop=True)
        self._candidates = candidates.reset_index(drop=True)
        self._candidate_by_id = candidates.set_index("entity_id", drop=False).to_dict("index")
        self._name_index = self._build_index(candidates, "name_norm")
        self._address_index = self._build_index(candidates, "address_norm")
        self.name_token_frequency = Counter(
            token for tokens in candidates["name_tokens"] for token in set(tokens) if token
        )
        self.address_token_frequency = Counter(
            token for tokens in candidates["address_tokens"] for token in set(tokens) if token
        )
        self._name_token_index = self._build_token_index(
            candidates,
            "name_tokens",
            self.name_token_frequency,
            self.config.name_token_max_frequency,
        )
        self._address_token_index = self._build_token_index(
            candidates,
            "address_tokens",
            self.address_token_frequency,
            self.config.address_token_max_frequency,
        )
        return self

    def _compatible_candidate_ids(self, s1: dict, candidate_ids: Iterable[str]) -> set[str]:
        country = _as_text(s1["country_norm"])
        return {
            candidate_id
            for candidate_id in candidate_ids
            if _country_compatible(country, _as_text(self._candidate_by_id[candidate_id]["country_norm"]))
        }

    def _exact_rule(self, column: str, index: Mapping[str, list[str]]) -> set[tuple[str, str]]:
        pairs = set()
        for s1 in self._source1.to_dict("records"):
            key = _as_text(s1[column])
            if not key:
                continue
            bucket = index.get(key, [])
            if len(bucket) > self.config.max_exact_bucket_size:
                continue
            for candidate_id in self._compatible_candidate_ids(s1, bucket):
                pairs.add((s1["entity_id"], candidate_id))
        return pairs

    def _token_rule(
        self,
        column: str,
        index: Mapping[str, list[str]],
        frequencies: Counter[str],
        max_frequency: int,
    ) -> set[tuple[str, str]]:
        pairs = set()
        for s1 in self._source1.to_dict("records"):
            tokens = {
                token
                for token in s1[column]
                if token and frequencies[token] <= max_frequency
            }
            candidate_ids = {
                candidate_id
                for token in tokens
                for candidate_id in index.get(token, [])
            }
            for candidate_id in self._compatible_candidate_ids(s1, candidate_ids):
                pairs.add((s1["entity_id"], candidate_id))
        return pairs

    def _token_overlap_rule(
        self,
        column: str,
        index: Mapping[str, list[str]],
        frequencies: Counter[str],
        max_frequency: int,
        minimum_overlap: int,
    ) -> set[tuple[str, str]]:
        """Return pairs sharing at least ``minimum_overlap`` indexed tokens.

        The inverted index keeps this bounded by the sum of the participating
        token buckets; it never constructs the S1 x candidate Cartesian table.
        """
        pairs = set()
        for s1 in self._source1.to_dict("records"):
            tokens = {
                token
                for token in s1[column]
                if token and frequencies[token] <= max_frequency
            }
            overlap_counts = Counter(
                candidate_id
                for token in tokens
                for candidate_id in index.get(token, [])
            )
            candidate_ids = {
                candidate_id
                for candidate_id, overlap in overlap_counts.items()
                if overlap >= minimum_overlap
            }
            for candidate_id in self._compatible_candidate_ids(s1, candidate_ids):
                pairs.add((s1["entity_id"], candidate_id))
        return pairs

    def _name_address_token_rule(self) -> set[tuple[str, str]]:
        pairs = set()
        for s1 in self._source1.to_dict("records"):
            name_tokens = {
                token
                for token in s1["name_tokens"]
                if token
                and self.name_token_frequency[token]
                <= self.config.name_token_max_frequency
            }
            address_tokens = {
                token
                for token in s1["address_tokens"]
                if token
                and self.address_token_frequency[token]
                <= self.config.address_token_max_frequency
            }
            name_candidates = {
                candidate_id
                for token in name_tokens
                for candidate_id in self._name_token_index.get(token, [])
            }
            address_candidates = {
                candidate_id
                for token in address_tokens
                for candidate_id in self._address_token_index.get(token, [])
            }
            for candidate_id in self._compatible_candidate_ids(
                s1, name_candidates & address_candidates
            ):
                pairs.add((s1["entity_id"], candidate_id))
        return pairs

    def _rule_pairs(self, rule_name: str) -> set[tuple[str, str]]:
        self._require_fit()
        if rule_name == "country_name_exact":
            return self._exact_rule("name_norm", self._name_index)
        if rule_name == "country_address_exact":
            return self._exact_rule("address_norm", self._address_index)
        if rule_name == "name_token":
            return self._token_rule(
                "name_tokens",
                self._name_token_index,
                self.name_token_frequency,
                self.config.name_token_max_frequency,
            )
        if rule_name == "address_token":
            return self._token_rule(
                "address_tokens",
                self._address_token_index,
                self.address_token_frequency,
                self.config.address_token_max_frequency,
            )
        if rule_name == "name_address_token":
            return self._name_address_token_rule()
        if rule_name in {"name_token_overlap_2", "name_token_overlap_3"}:
            return self._token_overlap_rule(
                "name_tokens",
                self._name_token_index,
                self.name_token_frequency,
                self.config.name_token_max_frequency,
                int(rule_name.rsplit("_", 1)[1]),
            )
        if rule_name in {"address_token_overlap_2", "address_token_overlap_3"}:
            return self._token_overlap_rule(
                "address_tokens",
                self._address_token_index,
                self.address_token_frequency,
                self.config.address_token_max_frequency,
                int(rule_name.rsplit("_", 1)[1]),
            )
        raise ValueError(f"Unknown blocking rule: {rule_name}")

    def generate(self, rule_names: Sequence[str] | None = None) -> pd.DataFrame:
        """Generate the deduplicated union of the requested blocking rules."""
        self._require_fit()
        requested = tuple(rule_names or self.config.default_rules)
        if not requested:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        unknown = set(requested) - set(RULE_NAMES)
        if unknown:
            raise ValueError(f"Unknown blocking rules: {sorted(unknown)}")

        provenance: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        for rule_name in dict.fromkeys(requested):
            for pair in self._rule_pairs(rule_name):
                provenance[pair].add(rule_name)

        rows = []
        for (s1_id, candidate_id), rules in provenance.items():
            rows.append(
                {
                    SOURCE1_ID: s1_id,
                    CANDIDATE_ID: candidate_id,
                    CANDIDATE_SOURCE: self._candidate_by_id[candidate_id]["source"],
                    BLOCKING_RULES: "|".join(sorted(rules, key=RULE_NAMES.index)),
                }
            )
        if not rows:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        return (
            pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
            .sort_values([SOURCE1_ID, CANDIDATE_ID], kind="stable")
            .reset_index(drop=True)
        )


def _truth_pairs(ground_truth: pd.DataFrame) -> tuple[set[tuple[str, str]], set[str]]:
    _validate_truth(ground_truth)
    pairs = set()
    s1_ids = set()
    for s1_id, matched in ground_truth[[SOURCE1_ID, "matched_entity_ids"]].itertuples(
        index=False, name=None
    ):
        s1_id = _as_text(s1_id)
        s1_ids.add(s1_id)
        pairs.update((s1_id, candidate_id) for candidate_id in _parse_truth_ids(matched))
    return pairs, s1_ids


def _candidate_pairs(candidates: pd.DataFrame) -> set[tuple[str, str]]:
    required = {SOURCE1_ID, CANDIDATE_ID}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")
    return {
        (_as_text(s1_id), _as_text(candidate_id))
        for s1_id, candidate_id in candidates[[SOURCE1_ID, CANDIDATE_ID]].itertuples(
            index=False, name=None
        )
        if _as_text(s1_id) and _as_text(candidate_id)
    }


def candidate_pairs_by_s1(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    s1_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return one evaluation row for every S1 entity, including zero candidates."""
    candidate_pairs = _candidate_pairs(candidates)
    truth_pairs, truth_s1_ids = _truth_pairs(ground_truth)
    if s1_ids is None:
        evaluation_ids = set(truth_s1_ids)
    else:
        evaluation_ids = {_as_text(value) for value in s1_ids if _as_text(value)}
    evaluation_ids.update(s1_id for s1_id, _ in candidate_pairs)

    truth_by_s1: defaultdict[str, set[str]] = defaultdict(set)
    retrieved_by_s1: defaultdict[str, set[str]] = defaultdict(set)
    for s1_id, candidate_id in truth_pairs:
        truth_by_s1[s1_id].add(candidate_id)
    for s1_id, candidate_id in candidate_pairs & truth_pairs:
        retrieved_by_s1[s1_id].add(candidate_id)
    counts = Counter(s1_id for s1_id, _ in candidate_pairs)

    rows = []
    for s1_id in sorted(evaluation_ids):
        truth_ids = truth_by_s1.get(s1_id, set())
        retrieved_ids = retrieved_by_s1.get(s1_id, set())
        rows.append(
            {
                SOURCE1_ID: s1_id,
                "candidate_count": counts.get(s1_id, 0),
                "has_ground_truth_match": bool(truth_ids),
                "ground_truth_match_count": len(truth_ids),
                "retrieved_ground_truth_match_count": len(retrieved_ids),
                "per_s1_blocking_recall": (
                    len(retrieved_ids) / len(truth_ids) if truth_ids else np.nan
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            SOURCE1_ID,
            "candidate_count",
            "has_ground_truth_match",
            "ground_truth_match_count",
            "retrieved_ground_truth_match_count",
            "per_s1_blocking_recall",
        ],
    )


def summarize_candidate_distribution(
    per_s1: pd.DataFrame,
    *,
    prefix: str = "",
) -> dict[str, float | int]:
    """Summarize candidate-count tails and requested threshold buckets."""
    if "candidate_count" not in per_s1.columns:
        raise ValueError("per_s1 must contain candidate_count")
    values = per_s1["candidate_count"].astype(float).to_numpy()
    if len(values):
        percentiles = np.percentile(values, [75, 90, 95, 99])
        summary: dict[str, float | int] = {
            "s1_count": int(len(values)),
            "mean_candidates_per_s1": float(np.mean(values)),
            "median_candidates_per_s1": float(np.median(values)),
            "p75_candidates_per_s1": float(percentiles[0]),
            "p90_candidates_per_s1": float(percentiles[1]),
            "p95_candidates_per_s1": float(percentiles[2]),
            "p99_candidates_per_s1": float(percentiles[3]),
            "maximum_candidates_per_s1": int(np.max(values)),
            "s1_with_0_candidates": int(np.count_nonzero(values == 0)),
            "s1_with_1_candidate": int(np.count_nonzero(values == 1)),
            "s1_with_le_5_candidates": int(np.count_nonzero(values <= 5)),
            "s1_with_le_10_candidates": int(np.count_nonzero(values <= 10)),
            "s1_with_le_20_candidates": int(np.count_nonzero(values <= 20)),
            "s1_with_le_50_candidates": int(np.count_nonzero(values <= 50)),
            "s1_with_le_100_candidates": int(np.count_nonzero(values <= 100)),
            "s1_with_gt_100_candidates": int(np.count_nonzero(values > 100)),
        }
    else:
        summary = {
            "s1_count": 0,
            "mean_candidates_per_s1": 0.0,
            "median_candidates_per_s1": 0.0,
            "p75_candidates_per_s1": 0.0,
            "p90_candidates_per_s1": 0.0,
            "p95_candidates_per_s1": 0.0,
            "p99_candidates_per_s1": 0.0,
            "maximum_candidates_per_s1": 0,
            "s1_with_0_candidates": 0,
            "s1_with_1_candidate": 0,
            "s1_with_le_5_candidates": 0,
            "s1_with_le_10_candidates": 0,
            "s1_with_le_20_candidates": 0,
            "s1_with_le_50_candidates": 0,
            "s1_with_le_100_candidates": 0,
            "s1_with_gt_100_candidates": 0,
        }
    return {f"{prefix}{key}": value for key, value in summary.items()}


def evaluate_candidate_pairs(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    s1_ids: Iterable[str] | None = None,
) -> dict[str, float | int]:
    """Evaluate recall and the full per-S1 candidate-count distribution."""
    per_s1 = candidate_pairs_by_s1(candidates, ground_truth, s1_ids=s1_ids)
    candidate_pairs = _candidate_pairs(candidates)
    truth, _ = _truth_pairs(ground_truth)
    retrieved = candidate_pairs & truth
    truth_s2 = {pair for pair in truth if pair[1].startswith("S2-")}
    truth_s3 = {pair for pair in truth if pair[1].startswith("S3-")}
    retrieved_s2 = retrieved & truth_s2
    retrieved_s3 = retrieved & truth_s3

    def recall(found: int, total: int) -> float:
        return found / total if total else 0.0

    result: dict[str, float | int] = {
        "number_s1_records": int(len(per_s1)),
        "candidate_count": len(candidate_pairs),
        "true_match_count": len(truth),
        "true_matches_retrieved": len(retrieved),
        "blocking_recall": recall(len(retrieved), len(truth)),
        "s2_true_match_count": len(truth_s2),
        "s2_true_matches_retrieved": len(retrieved_s2),
        "s2_recall": recall(len(retrieved_s2), len(truth_s2)),
        "s3_true_match_count": len(truth_s3),
        "s3_true_matches_retrieved": len(retrieved_s3),
        "s3_recall": recall(len(retrieved_s3), len(truth_s3)),
    }
    result.update(summarize_candidate_distribution(per_s1))
    matched = per_s1.loc[per_s1["has_ground_truth_match"]]
    zero_match = per_s1.loc[~per_s1["has_ground_truth_match"]]
    result.update(summarize_candidate_distribution(matched, prefix="matched_"))
    result.update(summarize_candidate_distribution(zero_match, prefix="zero_match_"))
    # Preserve the original public metric names.
    result["average_candidates_per_s1"] = result["mean_candidates_per_s1"]
    result["median_candidates_per_s1"] = result["median_candidates_per_s1"]
    return result


def evaluate_rule_contributions(
    blocker: CandidateBlocker,
    ground_truth: pd.DataFrame,
    rule_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Attribute candidates and true links uniquely to each requested rule."""
    names = tuple(rule_names or blocker.available_rules)
    pairs_by_rule = {name: set(blocker._rule_pairs(name)) for name in names}
    truth, _ = _truth_pairs(ground_truth)
    rows = []
    for name in names:
        pairs = pairs_by_rule[name]
        other_pairs = set().union(*(pairs_by_rule[other] for other in names if other != name))
        metrics = evaluate_candidate_pairs(
            blocker.generate((name,)), ground_truth, s1_ids=blocker.s1_ids
        )
        rows.append(
            {
                "rule": name,
                "candidate_count": len(pairs),
                "unique_candidate_count": len(pairs - other_pairs),
                "overlap_candidate_count": len(pairs & other_pairs),
                "true_links_retrieved": len(pairs & truth),
                "unique_true_links_retrieved": len((pairs & truth) - (other_pairs & truth)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def evaluate_blocking_rules(
    blocker: CandidateBlocker,
    ground_truth: pd.DataFrame,
    rule_sets: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Evaluate each rule and a union configuration in a comparable table."""
    if rule_sets is None:
        rule_sets = {rule: (rule,) for rule in blocker.available_rules}
        rule_sets = dict(rule_sets)
        rule_sets["union_default"] = blocker.config.default_rules

    rows = []
    for configuration, rules in rule_sets.items():
        candidates = blocker.generate(rules)
        metrics = evaluate_candidate_pairs(candidates, ground_truth, s1_ids=blocker.s1_ids)
        rows.append({"configuration": configuration, "rules": "|".join(rules), **metrics})
    return pd.DataFrame(rows)


__all__ = [
    "BLOCKING_RULES",
    "CANDIDATE_ID",
    "CANDIDATE_SOURCE",
    "CandidateBlocker",
    "BlockingConfig",
    "DEFAULT_RULE_NAMES",
    "EXPERIMENTAL_RULE_NAMES",
    "OUTPUT_COLUMNS",
    "SOURCE1_ID",
    "candidate_pairs_by_s1",
    "evaluate_blocking_rules",
    "evaluate_candidate_pairs",
    "evaluate_rule_contributions",
    "summarize_candidate_distribution",
]
