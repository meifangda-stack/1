"""Unsupervised market-regime helpers."""

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_regime_clusterer(n_clusters: int = 3, random_state: int = 42) -> Pipeline:
    """Build a baseline clustering pipeline for market-regime discovery."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("cluster", KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)),
        ]
    )
