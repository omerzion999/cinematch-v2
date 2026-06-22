"""Tests for the seed-picker: curated seeds, /api/seeds, and multi-seed similarity."""

import json
import os

from app.catalog_lookup import find_catalog_index
from app.engine.hybrid import recommend_from_seeds

GENRES = ["drama", "comedy", "action_adventure", "scifi_fantasy", "crime", "animation"]
SEEDS_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "data", "genre_seeds.json")


def _seed_titles() -> dict:
    with open(SEEDS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def test_every_curated_seed_resolves_in_catalog(catalog):
    """Every title in genre_seeds.json must resolve to a real catalog row."""
    seeds = _seed_titles()
    assert set(seeds) == set(GENRES)
    for genre, titles in seeds.items():
        assert len(titles) >= 6, f"{genre} has too few seeds"
        for title in titles:
            assert find_catalog_index(catalog, title) is not None, (
                f"seed {title!r} ({genre}) not found in catalog"
            )


def test_multi_seed_similarity_crime_drama_excludes_seeds(catalog, numeric_matrix, embeddings):
    picks = recommend_from_seeds(
        ["Breaking Bad", "Better Call Saul"], catalog, numeric_matrix, embeddings,
        top_n=3, query_lang="en",
    )
    assert 1 <= len(picks) <= 3
    titles = set(picks["title"])
    assert "Breaking Bad" not in titles and "Better Call Saul" not in titles
    for genres in picks["genres"]:
        assert "Crime" in genres or "Drama" in genres


def test_multi_seed_era_filter_prefers_recent_but_never_empty(catalog, numeric_matrix, embeddings):
    picks = recommend_from_seeds(
        ["Stranger Things"], catalog, numeric_matrix, embeddings,
        top_n=3, filters={"era_pref": "recent"}, query_lang="en",
    )
    # The seed has matches, so the relax-on-empty fallback guarantees >= 1.
    assert 1 <= len(picks) <= 3


def test_multi_seed_unknown_seed_returns_empty(catalog, numeric_matrix, embeddings):
    picks = recommend_from_seeds(
        ["zzz not a real show 123"], catalog, numeric_matrix, embeddings, top_n=3,
    )
    assert picks.empty


def test_api_seeds_returns_valid_cards_per_genre(client, catalog):
    for genre in GENRES:
        resp = client.get(f"/api/seeds?genre={genre}&lang=en")
        assert resp.status_code == 200
        body = resp.json()
        assert body["genre"] == genre
        assert len(body["seeds"]) >= 6
        for card in body["seeds"]:
            assert find_catalog_index(catalog, card["title"]) is not None


def test_api_seeds_rejects_unknown_genre(client):
    resp = client.get("/api/seeds?genre=not_a_genre&lang=en")
    assert resp.status_code == 422


def test_api_recommend_with_seeds_excludes_seeds(client):
    resp = client.post(
        "/api/recommend",
        json={
            "answers": {"genre": "crime", "era": "any", "popularity": "any"},
            "seeds": ["Breaking Bad", "Better Call Saul"],
            "lang": "en",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 1 <= len(body["recommendations"]) <= 3
    titles = {r["title"] for r in body["recommendations"]}
    assert "Breaking Bad" not in titles and "Better Call Saul" not in titles


def test_api_recommend_unknown_seed_is_graceful(client):
    """Failure case (rubric section 7): a seed that does not exist in the catalog
    yields no matches, and the API returns a graceful message with an empty list,
    never a crash or an invented title."""
    resp = client.post(
        "/api/recommend",
        json={
            "answers": {"genre": "crime", "era": "any", "popularity": "any"},
            "seeds": ["a show that absolutely does not exist 99999"],
            "lang": "en",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommendations"] == []
    assert body["intro"]  # a graceful no-match message, not empty


def test_api_recommend_without_seeds_uses_preference_ranker(client):
    resp = client.post(
        "/api/recommend",
        json={"answers": {"genre": "comedy", "era": "any", "popularity": "any"}, "lang": "en"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 1 <= len(body["recommendations"]) <= 3
    for rec in body["recommendations"]:
        assert "Comedy" in rec["genres"]
