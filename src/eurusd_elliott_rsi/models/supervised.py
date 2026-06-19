"""Supervised learning helpers for directional prediction."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_direction_classifier(random_state: int = 42) -> Pipeline:
    """Build a baseline classifier pipeline for next-bar direction labels."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=random_state)),
        ]
    )
