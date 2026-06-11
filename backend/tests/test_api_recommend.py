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


def test_recommend_invalid_answer_value_returns_422(client):
    response = client.post(
        "/api/recommend", json={"answers": {"genre": "not_a_real_genre"}, "lang": "en"}
    )
    assert response.status_code == 422


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


def test_clean_poster_path_normalizes_nan_and_empty_strings():
    import math

    from app.routers.recommend import _clean_poster_path

    assert _clean_poster_path(math.nan) is None
    assert _clean_poster_path("") is None
    assert _clean_poster_path("/abc123.jpg") == "/abc123.jpg"


def test_resolve_poster_path_prefers_catalog_value(monkeypatch):
    import app.routers.recommend as recommend_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("search_tv_show should not be called when catalog has a poster_path")

    monkeypatch.setattr(recommend_module, "search_tv_show", fail_if_called)

    pick = {"title": "Breaking Bad", "poster_path": "/catalog.jpg", "start_year": 2008}
    assert recommend_module._resolve_poster_path(pick) == "/catalog.jpg"


def test_resolve_poster_path_falls_back_to_tmdb_when_catalog_value_missing(monkeypatch):
    import app.routers.recommend as recommend_module

    monkeypatch.setattr(
        recommend_module,
        "search_tv_show",
        lambda title, year=None: {"poster_path": "/from-tmdb.jpg"},
    )

    pick = {"title": "Dead to Me", "poster_path": "", "start_year": 2019}
    assert recommend_module._resolve_poster_path(pick) == "/from-tmdb.jpg"


def test_resolve_poster_path_returns_none_when_tmdb_has_no_match(monkeypatch):
    import app.routers.recommend as recommend_module

    monkeypatch.setattr(recommend_module, "search_tv_show", lambda title, year=None: None)

    pick = {"title": "Dead to Me", "poster_path": "", "start_year": 2019}
    assert recommend_module._resolve_poster_path(pick) is None
