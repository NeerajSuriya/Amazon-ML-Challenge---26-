"""Production matching model for business entity resolution."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


FEATURE_COLUMNS = [
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

PAIR_ID_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
]

MATCH_THRESHOLD = 0.85


class ProductionMatcher:
    """Train and score candidate entity pairs."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state,
        )
        self.is_fitted = False

    @staticmethod
    def _validate_features(features: pd.DataFrame) -> None:
        missing = [
            column
            for column in FEATURE_COLUMNS
            if column not in features.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required matcher features: {missing}"
            )

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series | None = None,
    ) -> "ProductionMatcher":
        """Fit the matcher using the existing pair-feature matrix."""

        self._validate_features(features)

        if labels is None:
            if "label" not in features.columns:
                raise ValueError(
                    "No labels supplied and 'label' column is missing."
                )
            labels = features["label"]

        X = features[FEATURE_COLUMNS].astype(float)
        y = pd.Series(labels).astype(int)

        if len(X) != len(y):
            raise ValueError(
                f"Feature/label length mismatch: {len(X)} vs {len(y)}"
            )

        if y.nunique() < 2:
            raise ValueError(
                "Training data must contain both positive and negative labels."
            )

        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return P(match) for every candidate row, preserving row order."""

        if not self.is_fitted:
            raise RuntimeError(
                "Matcher has not been fitted. Call fit() first."
            )

        self._validate_features(features)

        X = features[FEATURE_COLUMNS].astype(float)

        # Column 1 is the probability of label=1, i.e. P(match).
        return self.model.predict_proba(X)[:, 1]

    def predict_dataframe(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return candidate identities plus P(match), preserving row order."""

        probabilities = self.predict_proba(features)

        output = features[PAIR_ID_COLUMNS].copy()
        output["match_probability"] = probabilities

        return output