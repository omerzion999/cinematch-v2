from app.state import load_state


def test_load_state_returns_expected_keys_and_shapes():
    state = load_state()

    assert len(state["catalog"]) == 11013
    assert state["embeddings"].shape == (11013, 384)
    assert state["numeric_matrix"].shape == (11013, 11013)

    assert "cluster_id" in state["catalog_with_features"].columns
    assert len(state["catalog_with_features"]) == len(state["catalog"])

    assert len(state["feature_dims"]) == 14
    assert len(state["cluster_centroids"]) == len(state["cluster_profiles"])
    for centroid in state["cluster_centroids"]:
        assert len(centroid) == 14

    assert isinstance(state["anomaly_threshold"], float)
