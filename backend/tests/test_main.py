def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_app_state_loaded_on_startup(client):
    state = client.app.state.cinematch
    assert "catalog" in state
    assert "catalog_with_features" in state
    assert "anomaly_threshold" in state
