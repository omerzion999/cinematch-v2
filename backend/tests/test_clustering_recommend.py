import numpy as np
import pandas as pd

from app.clustering.features import FEATURE_DIMS
from app.clustering.recommend import nearest_cluster, recommend_from_cluster


def _zeros_vector():
    return np.zeros(len(FEATURE_DIMS), dtype=np.float32)


def _zeros_mask():
    return np.zeros(len(FEATURE_DIMS), dtype=bool)


def test_nearest_cluster_all_any_picks_highest_binge_fit_score():
    vector = _zeros_vector()
    mask = _zeros_mask()
    centroids = [[0.0] * 14, [0.0] * 14, [0.0] * 14]
    profiles = {
        "0": {"avg_binge_fit_score": 4.0},
        "1": {"avg_binge_fit_score": 6.5},
        "2": {"avg_binge_fit_score": 5.0},
    }
    assert nearest_cluster(vector, mask, centroids, profiles) == 1


def test_nearest_cluster_picks_closest_centroid_on_masked_dims():
    vector = _zeros_vector()
    mask = _zeros_mask()
    drama_idx = FEATURE_DIMS.index("genre:Drama")
    mask[drama_idx] = True
    vector[drama_idx] = 1.0

    centroids = [[0.0] * 14, [0.0] * 14]
    centroids[0][drama_idx] = 0.1  # far from 1.0
    centroids[1][drama_idx] = 0.9  # close to 1.0
    profiles = {"0": {"avg_binge_fit_score": 10.0}, "1": {"avg_binge_fit_score": 0.0}}

    assert nearest_cluster(vector, mask, centroids, profiles) == 1


def _make_pool() -> pd.DataFrame:
    rows = []
    for i, (drama_val, binge) in enumerate([(1.0, 3.0), (0.9, 7.0), (0.0, 9.0)]):
        row = {"title": f"Show {i}", "cluster_id": 0, "binge_fit_score": binge}
        for dim in FEATURE_DIMS:
            row[dim] = 0.0
        row["genre:Drama"] = drama_val
        rows.append(row)
    # a title that belongs to a different cluster
    other = {"title": "Other Cluster Show", "cluster_id": 1, "binge_fit_score": 100.0}
    for dim in FEATURE_DIMS:
        other[dim] = 0.0
    rows.append(other)
    return pd.DataFrame(rows)


def test_recommend_from_cluster_ranks_by_distance_when_mask_set():
    pool = _make_pool()
    vector = _zeros_vector()
    mask = _zeros_mask()
    drama_idx = FEATURE_DIMS.index("genre:Drama")
    vector[drama_idx] = 1.0
    mask[drama_idx] = True

    result = recommend_from_cluster(pool, cluster_id=0, vector=vector, mask=mask, top_n=2)
    assert list(result["title"]) == ["Show 0", "Show 1"]


def test_recommend_from_cluster_ranks_by_binge_fit_score_when_mask_empty():
    pool = _make_pool()
    vector = _zeros_vector()
    mask = _zeros_mask()

    result = recommend_from_cluster(pool, cluster_id=0, vector=vector, mask=mask, top_n=2)
    assert list(result["title"]) == ["Show 2", "Show 1"]


def test_recommend_from_cluster_excludes_titles_and_other_clusters():
    pool = _make_pool()
    vector = _zeros_vector()
    mask = _zeros_mask()

    result = recommend_from_cluster(
        pool, cluster_id=0, vector=vector, mask=mask, top_n=3, exclude_titles=["Show 2"]
    )
    assert "Other Cluster Show" not in list(result["title"])
    assert "Show 2" not in list(result["title"])
    assert len(result) == 2
