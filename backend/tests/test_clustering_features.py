import numpy as np

from app.clustering.features import FEATURE_DIMS, GENRE_DIMS, NUMERIC_DIMS, build_cluster_features


def test_feature_dims_has_14_entries():
    assert len(GENRE_DIMS) == 10
    assert len(NUMERIC_DIMS) == 4
    assert len(FEATURE_DIMS) == 14
    assert FEATURE_DIMS[:10] == [f"genre:{g}" for g in GENRE_DIMS]
    assert FEATURE_DIMS[10:] == NUMERIC_DIMS


def test_build_cluster_features_shape_and_columns(catalog):
    features = build_cluster_features(catalog)
    assert len(features) == len(catalog)
    assert list(features.columns) == ["title"] + FEATURE_DIMS


def test_genre_dims_are_binary(catalog):
    features = build_cluster_features(catalog)
    for genre in GENRE_DIMS:
        col = features[f"genre:{genre}"]
        assert set(col.unique()).issubset({0.0, 1.0})


def test_game_of_thrones_genre_dims(catalog):
    features = build_cluster_features(catalog)
    row = features[features["title"] == "Game of Thrones"].iloc[0]
    # genres = "Action, Adventure, Drama"
    assert row["genre:Drama"] == 1.0
    assert row["genre:Action & Adventure"] == 0.0
    assert row["genre:Comedy"] == 0.0


def test_num_seasons_z_has_no_nan_and_is_roughly_standardized(catalog):
    features = build_cluster_features(catalog)
    z = features["num_seasons_z"]
    assert not z.isna().any()
    assert abs(z.mean()) < 1e-3
    assert abs(z.std() - 1.0) < 1e-3
