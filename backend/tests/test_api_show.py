import app.routers.show as show_module


def test_show_not_found_returns_404_with_i18n_message(client):
    response = client.get("/api/show/NoSuchShowXYZ123?lang=en")
    assert response.status_code == 404
    assert response.json()["detail"] == "I couldn't find that show in our catalog."

    response_he = client.get("/api/show/NoSuchShowXYZ123?lang=he")
    assert response_he.status_code == 404
    assert response_he.json()["detail"] == "לא מצאתי את הסדרה הזו במאגר שלנו."


def test_show_found_returns_catalog_data_without_tmdb(client, monkeypatch):
    monkeypatch.setattr(show_module, "search_tv_show", lambda title, year=None: None)

    catalog = client.app.state.cinematch["catalog"]
    title = catalog.iloc[0]["title"]

    response = client.get(f"/api/show/{title}")
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == title
    assert body["trailer_url"] is None
    assert body["cast"] == []
    assert body["watch_providers"] == []


def test_show_found_with_tmdb_populates_optional_fields(client, monkeypatch):
    catalog = client.app.state.cinematch["catalog"]
    title = catalog.iloc[1]["title"]

    monkeypatch.setattr(
        show_module, "search_tv_show", lambda t, year=None: {"id": 999}
    )
    monkeypatch.setattr(
        show_module,
        "get_tv_show_details",
        lambda tmdb_id: {
            "videos": {"results": [{"site": "YouTube", "type": "Trailer", "key": "abc123"}]},
            "credits": {"cast": [{"name": "Actor One"}, {"name": "Actor Two"}]},
            "watch/providers": {"results": {"US": {"flatrate": [{"provider_name": "Netflix"}]}}},
        },
    )

    response = client.get(f"/api/show/{title}")
    assert response.status_code == 200
    body = response.json()
    assert body["trailer_url"] == "https://www.youtube.com/watch?v=abc123"
    assert body["cast"] == ["Actor One", "Actor Two"]
    assert body["watch_providers"] == ["Netflix"]
