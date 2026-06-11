"""
Cluster-based recommender: given a target feature vector + mask
(see clustering/onboarding_map.py), find the nearest cluster and rank
titles within it.
"""

import numpy as np
import pandas as pd

from app.clustering.features import FEATURE_DIMS


def nearest_cluster(vector: np.ndarray, mask: np.ndarray, centroids: list, profiles: dict) -> int:
    """
    Returns the cluster_id whose centroid is closest to `vector` on the
    masked dims (Euclidean distance, unmasked dims contribute 0).

    If `mask` has no True entries (the user answered "doesn't matter" to
    everything), falls back to the cluster with the highest
    avg_binge_fit_score, per the design spec's edge-case notes.
    """
    if not mask.any():
        return int(max(profiles, key=lambda cid: profiles[cid]["avg_binge_fit_score"]))

    centroids_arr = np.asarray(centroids, dtype=np.float32)
    diffs = (centroids_arr - vector) * mask
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    return int(np.argmin(distances))


def recommend_from_cluster(
    catalog_with_features: pd.DataFrame,
    cluster_id: int,
    vector: np.ndarray,
    mask: np.ndarray,
    top_n: int = 3,
    exclude_titles: list[str] | None = None,
) -> pd.DataFrame:
    """
    catalog_with_features: catalog rows joined with cluster_id + FEATURE_DIMS
                            columns (see app/state.py, Task 15).
    Returns the top_n rows (all original columns, reset index) from the
    given cluster, ranked by masked Euclidean distance to `vector` (closer
    first). If `mask` has no True entries, ranks by binge_fit_score
    (descending) instead.
    """
    pool = catalog_with_features[catalog_with_features["cluster_id"] == cluster_id]
    if exclude_titles:
        pool = pool[~pool["title"].isin(exclude_titles)]

    if pool.empty:
        return pool.head(0)

    if not mask.any():
        ranked = pool.sort_values("binge_fit_score", ascending=False)
    else:
        feature_matrix = pool[FEATURE_DIMS].values.astype(np.float32)
        diffs = (feature_matrix - vector) * mask
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        ranked = pool.assign(_distance=distances).sort_values("_distance").drop(columns="_distance")

    return ranked.head(top_n).reset_index(drop=True)
