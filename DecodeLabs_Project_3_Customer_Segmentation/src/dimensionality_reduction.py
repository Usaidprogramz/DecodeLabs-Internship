"""
dimensionality_reduction.py
------------------------------
Phase 1 (Scale) + Phase 2 (Compress) of the brief's IPO architecture:
StandardScaler, then PCA down to the number of components that
explain >= 95% cumulative variance (capped at 3 for the notebook's
3D visualization, per the brief's "2 or 3 dimensions" requirement).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


@dataclass
class PCAResult:
    components: np.ndarray          # (n_customers, n_components_kept)
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    n_components_95pct: int          # components needed for >=95% variance (uncapped)
    n_components_kept: int           # actually kept for clustering (capped at max_components)
    scaler: StandardScaler
    pca: PCA
    feature_names: list


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def reduce_dimensions(X: pd.DataFrame, max_components: int = 3) -> PCAResult:
    """
    Scale X, then fit PCA on all components to find how many are needed
    for 95% cumulative explained variance -- then keep min(that, max_components)
    for the actual clustering/visualization, per the brief's "2 or 3
    dimensions" requirement.
    """
    X_scaled, scaler = scale_features(X)

    full_pca = PCA(random_state=RANDOM_STATE)
    full_pca.fit(X_scaled)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    n_components_95pct = int(np.searchsorted(cumulative, 0.95) + 1)

    n_keep = min(n_components_95pct, max_components)
    n_keep = max(n_keep, 2)  # always keep at least 2 for visualization

    pca = PCA(n_components=n_keep, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_scaled)

    return PCAResult(
        components=components,
        explained_variance_ratio=pca.explained_variance_ratio_,
        cumulative_variance=np.cumsum(pca.explained_variance_ratio_),
        n_components_95pct=n_components_95pct,
        n_components_kept=n_keep,
        scaler=scaler,
        pca=pca,
        feature_names=list(X.columns),
    )
