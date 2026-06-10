import numpy as np
import pandas as pd

from app.clustering.features import FEATURE_DIMS
from app.clustering.train_clusters import build_cluster_profiles, choose_k


def test_choose_k_returns_k_in_range_and_scores():
    rng = np.random.default_rng(42)
    # Three well-separated synthetic clusters in 14-dim space
    cluster_a = rng.normal(loc=0.0, scale=0.1, size=(50, 14))
    cluster_b = rng.normal(loc=5.0, scale=0.1, size=(50, 14))
    cluster_c = rng.normal(loc=-5.0, scale=0.1, size=(50, 14))
    X = np.vstack([cluster_a, cluster_b, cluster_c]).astype(np.float32)

    best_k, scores = choose_k(X, k_range=range(2, 5))
    assert best_k in range(2, 5)
    assert set(scores.keys()) == set(range(2, 5))
    assert all(-1.0 <= v <= 1.0 for v in scores.values())


def test_build_cluster_profiles_structure():
    titles = [f"Show {i}" for i in range(6)]
    features = pd.DataFrame({"title": titles})
    for dim in FEATURE_DIMS:
        features[dim] = 0.0
    # 3 shows are Drama-heavy, 3 are Comedy-heavy
    features.loc[0:2, "genre:Drama"] = 1.0
    features.loc[3:5, "genre:Comedy"] = 1.0

    catalog = pd.DataFrame({
        "start_year": [2018, 2019, 2020, 2010, 2011, 2012],
        "rating": [8.0, 8.2, 8.4, 7.0, 7.2, 7.4],
        "binge_fit_score": [5.0, 5.2, 5.4, 4.0, 4.2, 4.4],
    })
    labels = np.array([0, 0, 0, 1, 1, 1])

    profiles = build_cluster_profiles(features, catalog, labels)
    assert set(profiles.keys()) == {"0", "1"}
    assert profiles["0"]["top_genres"][0] == "Drama"
    assert profiles["1"]["top_genres"][0] == "Comedy"
    assert profiles["0"]["size"] == 3
    assert profiles["1"]["size"] == 3
    assert "label_he" in profiles["0"] and "label_en" in profiles["0"]
    assert profiles["0"]["avg_rating"] == 8.2
