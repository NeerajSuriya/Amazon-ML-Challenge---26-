"""Replaceable candidate-level matching interfaces.

``LogisticMatcher`` is an evaluation/reference matcher only; it is not the
team's production matching model. It exists so blocking and pair-feature
experiments can be evaluated before Neeraj's production matcher is available.
Neeraj's future matcher should implement or adapt to the same narrow contract:
``fit(features, feature_names)`` and
``predict_proba(features, feature_names)``. Every implementation scores each
candidate row independently and does not perform one-to-one assignment unless
that behavior is explicitly introduced by a future matcher adapter.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class Matcher(Protocol):
    """Minimal fit/score contract consumed by the evaluator."""

    def fit(self, features: pd.DataFrame, feature_names: Sequence[str]) -> "Matcher":
        ...

    def predict_proba(
        self, features: pd.DataFrame, feature_names: Sequence[str]
    ) -> np.ndarray:
        ...


class LogisticMatcher:
    """Adapter for the existing balanced LogisticRegression matcher."""

    def __init__(self, *, seed: int = 42):
        self.seed = int(seed)
        self.model = None
        self.constant_score: float | None = None

    def fit(self, features: pd.DataFrame, feature_names: Sequence[str]) -> "LogisticMatcher":
        missing = set(feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        labels = features["label"].to_numpy(dtype=int)
        self.model = None
        self.constant_score = None
        if len(labels) == 0:
            self.constant_score = 0.0
            return self
        unique = np.unique(labels)
        if len(unique) == 1:
            self.constant_score = float(unique[0])
            return self
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.seed,
            solver="liblinear",
        )
        self.model.fit(features[list(feature_names)].to_numpy(dtype=float), labels)
        return self

    def predict_proba(
        self, features: pd.DataFrame, feature_names: Sequence[str]
    ) -> np.ndarray:
        missing = set(feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        if features.empty:
            return np.empty(0, dtype=float)
        if self.model is None:
            return np.full(len(features), self.constant_score or 0.0, dtype=float)
        return self.model.predict_proba(
            features[list(feature_names)].to_numpy(dtype=float)
        )[:, 1]

    @property
    def coef_(self):
        return None if self.model is None else self.model.coef_

    @property
    def class_weight(self):
        return None if self.model is None else self.model.class_weight

    @property
    def random_state(self):
        return None if self.model is None else self.model.random_state

    @property
    def max_iter(self):
        return None if self.model is None else self.model.max_iter


__all__ = ["LogisticMatcher", "Matcher"]
