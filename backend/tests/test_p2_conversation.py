"""P2 conversational-quality behaviors: off-topic bridge, fuzzy title lookup,
no-hiccup fallback, and the Hebrew explainer going through the LLM."""

import app.agent.llm as llm
from app.catalog_lookup import find_catalog_index


# ── Off-topic bridge ──────────────────────────────────────────────────────────

def test_bridge_detects_cooking():
    out = llm._detect_bridge("i want to make pasta")
    assert out is not None
    topic, keywords = out
    assert topic == "cooking"
    assert "chef" in keywords


def test_bridge_detects_hebrew_sports():
    out = llm._detect_bridge("בא לי לשחק כדורגל")
    assert out is not None
    assert out[0] == "sports"


def test_bridge_ignores_plain_tv_request():
    assert llm._detect_bridge("recommend a dark thriller") is None
    assert llm._detect_bridge("something funny") is None


def test_chat_turn_bridges_offtopic_to_search(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: None)
    monkeypatch.setattr(llm, "_provider", None)
    result = llm.chat_turn([{"role": "user", "content": "i want to make pasta"}], lang="en")
    assert result["action"] == "search"
    assert result["intent"]["keywords"]
    assert result["reply"]  # a warm bridge one-liner


def test_chat_endpoint_bridge_returns_cooking_shows(client):
    resp = client.post("/api/chat", json={
        "conversation": [{"role": "user", "content": "how do I cook pasta"}],
        "prev_recs": None, "lang": "en",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommendations"]  # real catalog shows, not a hard decline


# ── Fuzzy title lookup ────────────────────────────────────────────────────────

def test_fuzzy_lookup_exact(catalog):
    # Breaking Bad is a well-known catalog title; exact match resolves.
    idx = find_catalog_index(catalog, "Breaking Bad")
    assert idx is not None
    assert catalog.loc[idx, "title"] == "Breaking Bad"


def test_fuzzy_lookup_typo(catalog):
    idx = find_catalog_index(catalog, "breakin bad")
    assert idx is not None
    assert "Breaking Bad" == catalog.loc[idx, "title"]


def test_fuzzy_lookup_unknown_returns_none(catalog):
    assert find_catalog_index(catalog, "zzzz nonexistent show qqqq") is None


# ── No-hiccup fallback ────────────────────────────────────────────────────────

def test_chat_turn_llm_failure_falls_back_to_search(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: True)
    monkeypatch.setattr(llm, "_provider", "groq")
    monkeypatch.setattr(llm, "_call_llm", lambda *a, **k: None)  # simulate failure
    result = llm.chat_turn([{"role": "user", "content": "hey there"}], lang="en")
    assert result["action"] == "search"  # never the old "brain hiccup" chat message
    assert "hiccup" not in result["reply"].lower()


# ── Hebrew explainer goes through the LLM ─────────────────────────────────────

def test_hebrew_explanation_uses_llm(monkeypatch):
    calls = {}

    def fake_call(system, user, *a, **k):
        calls["used"] = True
        return "הסבר בעברית שנכתב על ידי המודל."

    monkeypatch.setattr(llm, "_get_client", lambda: True)
    monkeypatch.setattr(llm, "_provider", "groq")
    monkeypatch.setattr(llm, "_call_llm", fake_call)

    recs = [{"title": "X", "genres": "Drama", "rating": 8.0, "decade_str": "2010s"}]
    out = llm.explain_recommendations({"mood": ["emotional"], "free_text": "משהו מרגש"}, recs, lang="he")
    assert calls.get("used") is True
    assert out == "הסבר בעברית שנכתב על ידי המודל."
