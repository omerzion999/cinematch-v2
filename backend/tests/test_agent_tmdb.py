import requests

from app.agent import tmdb


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json_data


def test_search_tv_show_no_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    assert tmdb.search_tv_show("Breaking Bad") is None


def test_search_tv_show_returns_first_result(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    def fake_get(url, params=None, timeout=None):
        assert url == f"{tmdb.TMDB_BASE_URL}/search/tv"
        assert params["query"] == "Breaking Bad"
        assert params["api_key"] == "test-key"
        return _FakeResponse({"results": [{"id": 1396, "name": "Breaking Bad"}]})

    monkeypatch.setattr(tmdb.requests, "get", fake_get)
    result = tmdb.search_tv_show("Breaking Bad")
    assert result == {"id": 1396, "name": "Breaking Bad"}


def test_search_tv_show_passes_year_param(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured.update(params)
        return _FakeResponse({"results": [{"id": 1, "name": "X"}]})

    monkeypatch.setattr(tmdb.requests, "get", fake_get)
    tmdb.search_tv_show("X", year=2008)
    assert captured["first_air_date_year"] == 2008


def test_search_tv_show_no_results_returns_none(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")
    monkeypatch.setattr(tmdb.requests, "get", lambda *a, **k: _FakeResponse({"results": []}))
    assert tmdb.search_tv_show("Nonexistent Show 12345") is None


def test_search_tv_show_request_exception_returns_none(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    def fake_get(*a, **k):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(tmdb.requests, "get", fake_get)
    assert tmdb.search_tv_show("Breaking Bad") is None


def test_get_tv_show_details_no_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    assert tmdb.get_tv_show_details(1396) is None


def test_get_tv_show_details_returns_json(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    def fake_get(url, params=None, timeout=None):
        assert url == f"{tmdb.TMDB_BASE_URL}/tv/1396"
        assert params["append_to_response"] == "videos,credits,watch/providers"
        return _FakeResponse({"id": 1396, "name": "Breaking Bad", "videos": {"results": []}})

    monkeypatch.setattr(tmdb.requests, "get", fake_get)
    result = tmdb.get_tv_show_details(1396)
    assert result["name"] == "Breaking Bad"


def test_get_tv_show_details_request_exception_returns_none(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    def fake_get(*a, **k):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(tmdb.requests, "get", fake_get)
    assert tmdb.get_tv_show_details(1396) is None
