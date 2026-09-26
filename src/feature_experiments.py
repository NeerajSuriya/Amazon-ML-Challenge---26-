"""Reusable feature-set, diagnostic, and aggregation helpers.

This module only operates on externally generated candidate rows. It does not
construct candidates and it never reads ground truth while computing features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .pair_features import (
    BASELINE_FEATURE_NAMES,
    EXPERIMENTAL_FEATURE_NAMES,
)

BASELINE_FEATURES = tuple(BASELINE_FEATURE_NAMES)
EXPERIMENTAL_GROUPS: dict[str, tuple[str, ...]] = {
    "name": tuple(EXPERIMENTAL_FEATURE_NAMES[0:3]),
    "address": tuple(EXPERIMENTAL_FEATURE_NAMES[3:8]),
    "country": tuple(EXPERIMENTAL_FEATURE_NAMES[8:10]),
    "cross_field": tuple(EXPERIMENTAL_FEATURE_NAMES[10:]),
}
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "baseline": BASELINE_FEATURES,
    **{f"experimental_{name}": values for name, values in EXPERIMENTAL_GROUPS.items()},
}


@dataclass(frozen=True)
class FeatureSet:
    """A deterministic, named set of columns passed to a matcher."""

    name: str
    columns: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.columns)


def _unique_ordered(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result:
        raise ValueError("Feature set cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError("Feature set contains duplicate feature names")
    return result


def resolve_feature_set(
    specification: str | Sequence[str],
    available: Sequence[str] = BASELINE_FEATURES + tuple(EXPERIMENTAL_FEATURE_NAMES),
) -> FeatureSet:
    """Resolve a named or explicit feature set with stable ordering."""
    available = tuple(available)
    if isinstance(specification, str):
        name = specification
        if name == "all":
            columns = available
        elif name in FEATURE_GROUPS:
            columns = FEATURE_GROUPS[name]
        elif name.startswith("baseline_plus_"):
            suffix = name.removeprefix("baseline_plus_")
            if suffix in EXPERIMENTAL_GROUPS:
                group_names = [suffix]
            elif suffix == "address_cross_field":
                group_names = ["address", "cross_field"]
            elif suffix == "name_address_cross":
                group_names = ["name", "address", "cross_field"]
            else:
                group_names = suffix.split("_")
            unknown = set(group_names) - set(EXPERIMENTAL_GROUPS)
            if unknown:
                raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
            columns = BASELINE_FEATURES + tuple(
                feature
                for group in group_names
                for feature in EXPERIMENTAL_GROUPS[group]
            )
        else:
            raise ValueError(f"Unknown feature set: {specification}")
    else:
        name = "explicit"
        columns = _unique_ordered(specification)

    columns = _unique_ordered(columns)
    unknown = set(columns) - set(available)
    if unknown:
        raise ValueError(f"Unknown feature names: {sorted(unknown)}")
    return FeatureSet(name=name, columns=columns)


def default_feature_sets() -> tuple[FeatureSet, ...]:
    """Return the predeclared ablation sets used by the experiment runner."""
    names = (
        "baseline",
        "baseline_plus_name",
        "baseline_plus_address",
        "baseline_plus_country",
        "baseline_plus_cross_field",
        "baseline_plus_name_address_cross",
    )
    return tuple(resolve_feature_set(name) for name in names)


def _summary(values: pd.Series) -> tuple[float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return float("nan"), float("nan"), float("nan")
    return float(numeric.mean()), float(numeric.std(ddof=0)), float(numeric.median())


def diagnose_features(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    label_col: str = "label",
) -> pd.DataFrame:
    """Describe feature health and univariate positive/negative separation."""
    missing = set(feature_names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    labels = frame[label_col].astype(int) if label_col in frame else None
    rows = []
    for feature in feature_names:
        values = pd.to_numeric(frame[feature], errors="coerce")
        unique_count = int(values.nunique(dropna=True))
        value_counts = values.value_counts(dropna=False)
        dominant_share = float(value_counts.iloc[0] / len(values)) if len(values) else 0.0
        row = {
            "feature": feature,
            "dtype": str(frame[feature].dtype),
            "missing_rate": float(values.isna().mean()) if len(values) else 0.0,
            "zero_rate": float((values == 0).mean()) if len(values) else 0.0,
            "unique_values": unique_count,
            "constant": bool(unique_count <= 1),
            "nearly_constant": bool(unique_count <= 2 or dominant_share >= 0.99),
            "overall_mean": float(values.mean()) if len(values) else float("nan"),
            "overall_std": float(values.std(ddof=0)) if len(values) else float("nan"),
        }
        if labels is None:
            positive = values.iloc[0:0]
            negative = values.iloc[0:0]
        else:
            positive = values[labels == 1]
            negative = values[labels == 0]
        p_mean, p_std, p_median = _summary(positive)
        n_mean, n_std, n_median = _summary(negative)
        pooled = np.sqrt((p_std**2 + n_std**2) / 2) if np.isfinite(p_std + n_std) else np.nan
        row.update(
            {
                "positive_count": int(len(positive)),
                "negative_count": int(len(negative)),
                "positive_mean": p_mean,
                "positive_std": p_std,
                "positive_median": p_median,
                "negative_mean": n_mean,
                "negative_std": n_std,
                "negative_median": n_median,
                "mean_difference": p_mean - n_mean if np.isfinite(p_mean + n_mean) else np.nan,
                "standardized_mean_difference": (
                    (p_mean - n_mean) / pooled if np.isfinite(pooled) and pooled else np.nan
                ),
                "univariate_auc": np.nan,
            }
        )
        if labels is not None and labels.nunique() == 2:
            valid = values.notna()
            try:
                row["univariate_auc"] = float(roc_auc_score(labels[valid], values[valid]))
            except ValueError:
                row["univariate_auc"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def feature_correlations(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Return a long-form Pearson correlation table for numeric features."""
    values = frame[list(feature_names)].apply(pd.to_numeric, errors="coerce")
    matrix = values.corr(method="pearson")
    rows = []
    for i, left in enumerate(feature_names):
        for right in feature_names[i + 1 :]:
            rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "correlation": float(matrix.loc[left, right]),
                }
            )
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "correlation"])


def coefficient_diagnostics(model, feature_names: Sequence[str]) -> pd.DataFrame:
    """Extract deterministic coefficient diagnostics from a fitted linear matcher."""
    coefficients = getattr(model, "coef_", None)
    if coefficients is None and getattr(model, "model", None) is not None:
        coefficients = getattr(model.model, "coef_", None)
    if coefficients is None:
        return pd.DataFrame(columns=["feature", "coefficient", "absolute_coefficient"])
    values = np.asarray(coefficients)[0]
    return pd.DataFrame(
        {
            "feature": list(feature_names),
            "coefficient": values.astype(float),
            "absolute_coefficient": np.abs(values.astype(float)),
        }
    ).sort_values("absolute_coefficient", ascending=False, kind="stable").reset_index(drop=True)


def paired_differences(
    per_seed: pd.DataFrame,
    *,
    baseline: str = "baseline",
    metric: str = "validation_macro_f0_5",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate paired per-seed differences against one baseline set."""
    required = {"feature_set", "seed", metric}
    missing = required - set(per_seed.columns)
    if missing:
        raise ValueError(f"Missing paired-difference columns: {sorted(missing)}")
    baseline_rows = per_seed.loc[per_seed["feature_set"] == baseline, ["seed", metric]]
    if baseline_rows["seed"].duplicated().any():
        raise ValueError(f"Baseline has duplicate rows for seed: {baseline}")
    baseline_by_seed = baseline_rows.set_index("seed")[metric].to_dict()
    rows = []
    for row in per_seed.itertuples(index=False):
        if row.feature_set == baseline:
            continue
        if row.seed not in baseline_by_seed:
            raise ValueError(f"No baseline row for seed {row.seed}")
        value = float(getattr(row, metric))
        baseline_value = float(baseline_by_seed[row.seed])
        rows.append(
            {
                "feature_set": row.feature_set,
                "seed": row.seed,
                "baseline_macro_f0_5": baseline_value,
                "feature_macro_f0_5": value,
                "paired_improvement": value - baseline_value,
            }
        )
    per_seed_output = pd.DataFrame(
        rows,
        columns=[
            "feature_set",
            "seed",
            "baseline_macro_f0_5",
            "feature_macro_f0_5",
            "paired_improvement",
        ],
    )
    if per_seed_output.empty:
        return per_seed_output, pd.DataFrame()
    summary = (
        per_seed_output.groupby("feature_set", sort=False)["paired_improvement"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "seeds", "std": "paired_improvement_std"})
    )
    summary["paired_improvement_std"] = summary["paired_improvement_std"].fillna(0.0)
    return per_seed_output, summary


def aggregate_results(per_seed: pd.DataFrame, group_columns=("feature_set",)) -> pd.DataFrame:
    """Aggregate numeric experiment results without pooling predictions."""
    if per_seed.empty:
        return pd.DataFrame()
    metric_names = [
        name
        for name in ("macro_f0_5", "precision", "recall", "threshold")
        if name in per_seed.columns
    ]
    metric_names.extend(
        name
        for name in ("validation_macro_f0_5", "validation_precision", "validation_recall")
        if name in per_seed.columns
    )
    metric_names.extend(
        name
        for name in ("candidate_count", "blocking_recall")
        if name in per_seed.columns
    )
    grouped = per_seed.groupby(list(group_columns), dropna=False, sort=False)
    rows = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["seeds"] = len(group)
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "BASELINE_FEATURES",
    "EXPERIMENTAL_GROUPS",
    "FEATURE_GROUPS",
    "FeatureSet",
    "aggregate_results",
    "coefficient_diagnostics",
    "default_feature_sets",
    "diagnose_features",
    "feature_correlations",
    "paired_differences",
    "resolve_feature_set",
]
