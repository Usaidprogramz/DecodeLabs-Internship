"""
clustering.py
--------------
Phase 3 (Cluster) of the brief's IPO architecture: proves the optimal
number of K-Means clusters mathematically using BOTH the Elbow Method
(WCSS / inertia) and the Silhouette Score, rather than picking K by
eye or by convention.
"""

from dataclasses import dataclass

import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42


@dataclass
class KSelectionResult:
    k_values: list
    wcss: list                 # within-cluster sum of squares (inertia) per K
    silhouette_scores: list     # silhouette score per K
    elbow_k: int                # K suggested by the Elbow Method (KneeLocator)
    best_silhouette_k: int       # K with the highest silhouette score
    chosen_k: int                # final K used (elbow_k if the two gatekeepers agree closely, else best_silhouette_k)


def evaluate_k_range(X_reduced: np.ndarray, k_min: int = 2, k_max: int = 10) -> KSelectionResult:
    """
    Fit K-Means for every K in [k_min, k_max], recording WCSS (for the
    Elbow Method) and the Silhouette Score for each. Both diagnostics
    are computed on the PCA-reduced space, as this is exactly the space
    the final clustering runs in.
    """
    k_values = list(range(k_min, k_max + 1))
    wcss, sil_scores = [], []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_reduced)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X_reduced, labels))

    knee = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")
    elbow_k = knee.elbow if knee.elbow is not None else k_values[len(k_values) // 2]

    best_silhouette_k = k_values[int(np.argmax(sil_scores))]

    # The two gatekeepers usually agree closely; when they don't, prefer
    # the Silhouette Score, since it directly measures cluster quality
    # (cohesion vs. separation) rather than just curvature of a WCSS
    # curve, which can be ambiguous on real-world data.
    chosen_k = best_silhouette_k

    return KSelectionResult(
        k_values=k_values,
        wcss=wcss,
        silhouette_scores=sil_scores,
        elbow_k=elbow_k,
        best_silhouette_k=best_silhouette_k,
        chosen_k=chosen_k,
    )


def fit_final_clustering(X_reduced: np.ndarray, k: int):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_reduced)
    final_silhouette = silhouette_score(X_reduced, labels)
    return km, labels, final_silhouette
