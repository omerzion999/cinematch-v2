def test_recommend_returns_three_picks_with_intro_and_outro(client):
    response = client.post(
        "/api/recommend",
        json={
            "answers": {
                "genre": "drama",
                "length": "long",
                "era": "recent",
                "tone": "serious_drama",
                "popularity": "well_known",
            },
            "lang": "en",
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["intro"]
    assert body["outro"]
    assert isinstance(body["cluster_id"], int)
    assert 1 <= len(body["recommendations"]) <= 3

    for rec in body["recommendations"]:
        assert rec["title"]
        assert rec["genres"]
        assert isinstance(rec["rating"], float)
        assert rec["explanation"]


def test_recommend_all_any_returns_three_picks(client):
    response = client.post("/api/recommend", json={"answers": {}, "lang": "he"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 3


def test_recommend_hebrew_uses_hebrew_strings(client):
    response = client.post(
        "/api/recommend", json={"answers": {"genre": "comedy"}, "lang": "he"}
    )
    body = response.json()
    assert "תתחבר" in body["intro"]
    assert "אהבת" in body["outro"]


def test_recommend_no_picks_in_a_cluster_returns_no_recommendations_message(client, monkeypatch):
    import app.routers.recommend as recommend_module

    monkeypatch.setattr(
        recommend_module,
        "recommend_from_cluster",
        lambda *args, **kwargs: recommend_module.pd.DataFrame(),
    )

    response = client.post("/api/recommend", json={"answers": {}, "lang": "en"})
    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] == []
    assert body["intro"]
