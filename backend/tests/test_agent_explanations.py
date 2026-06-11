import json

from app.agent import explanations


def _sample_picks():
    return [
        {"title": "Show A", "genres": "Drama, Crime", "decade_str": "2010s", "rating": 8.5, "overview": "A gritty drama."},
        {"title": "Show B", "genres": "Comedy", "decade_str": "2000s", "rating": 7.2, "overview": "A funny sitcom."},
    ]


def _sample_profile():
    return {"label_he": "דרמות פשע מהשנים האחרונות", "label_en": "Crime Dramas & recent years"}


def test_explain_picks_empty_list_returns_empty():
    assert explanations.explain_picks({}, _sample_profile(), [], "en") == []


def test_explain_picks_no_provider_uses_fallback(monkeypatch):
    monkeypatch.setattr(explanations, "_get_client", lambda: None)
    picks = _sample_picks()
    result = explanations.explain_picks({}, _sample_profile(), picks, "en")
    assert len(result) == 2
    assert "8.5" in result[0]
    assert "Crime Dramas" in result[0]


def test_explain_picks_no_provider_hebrew_fallback(monkeypatch):
    monkeypatch.setattr(explanations, "_get_client", lambda: None)
    picks = _sample_picks()
    result = explanations.explain_picks({}, _sample_profile(), picks, "he")
    assert "דרמות פשע" in result[0]


def test_explain_picks_with_provider_returns_llm_array(monkeypatch):
    monkeypatch.setattr(explanations, "_get_client", lambda: True)
    monkeypatch.setattr(
        explanations, "_call_llm",
        lambda system, user, max_tokens=500: json.dumps(["Great gritty drama!", "Light and fun sitcom."])
    )
    picks = _sample_picks()
    result = explanations.explain_picks({"genre": "drama"}, _sample_profile(), picks, "en")
    assert result == ["Great gritty drama!", "Light and fun sitcom."]


def test_explain_picks_with_provider_invalid_response_falls_back(monkeypatch):
    monkeypatch.setattr(explanations, "_get_client", lambda: True)
    monkeypatch.setattr(explanations, "_call_llm", lambda system, user, max_tokens=500: "not json")
    picks = _sample_picks()
    result = explanations.explain_picks({}, _sample_profile(), picks, "en")
    assert len(result) == 2
    assert "8.5" in result[0]


def test_explain_picks_with_provider_wrong_length_falls_back(monkeypatch):
    monkeypatch.setattr(explanations, "_get_client", lambda: True)
    monkeypatch.setattr(
        explanations, "_call_llm",
        lambda system, user, max_tokens=500: json.dumps(["Only one explanation"])
    )
    picks = _sample_picks()
    result = explanations.explain_picks({}, _sample_profile(), picks, "en")
    assert len(result) == 2
