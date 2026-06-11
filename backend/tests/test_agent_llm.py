import pytest

from app.agent import llm


@pytest.fixture(autouse=True)
def reset_llm_client_state():
    """Reset llm module global client state before and after each test."""
    llm._provider = None
    llm._groq_client = None
    llm._anthropic_client = None
    yield
    llm._provider = None
    llm._groq_client = None
    llm._anthropic_client = None


def test_read_secret_reads_from_environ(monkeypatch):
    monkeypatch.setenv("SOME_TEST_KEY", "test-value-123")
    assert llm._read_secret("SOME_TEST_KEY") == "test-value-123"


def test_read_secret_missing_returns_none(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_KEY", raising=False)
    assert llm._read_secret("SOME_UNSET_KEY") is None


def test_regex_parse_detects_dark_mood_and_hebrew():
    # "ומותחן" (and-thriller) must include the trailing "ן" so this string
    # contains "מותחן" — the exact "thrilling" keyword in _regex_parse's
    # mood_map (llm.py). Without that final letter the substring match
    # fails against the verbatim-ported v1 regex.
    result = llm._regex_parse("אני רוצה משהו אפל ומותחן")
    assert "dark" in result["mood"]
    assert "thrilling" in result["mood"]
    assert result["lang"] == "he"


def test_regex_parse_detects_short_length_and_hidden_gem():
    result = llm._regex_parse("looking for a short hidden gem")
    assert result["length_pref"] == "short"
    assert result["popularity_pref"] == "hidden_gem"
    assert result["lang"] == "en"


def test_regex_parse_year_boundaries():
    result = llm._regex_parse("something from 2020")
    assert result["year_min"] == 2020
    assert result["era_pref"] == "recent"

    result2 = llm._regex_parse("a show from before 2000")
    assert result2["year_max"] == 1999
    assert result2["era_pref"] == "classic"


def test_regex_parse_year_and_later_phrasing():
    result = llm._regex_parse("year 2020 and later")
    assert result["year_min"] == 2020
    assert result["era_pref"] == "recent"

    result_he = llm._regex_parse("שנה 2015 ומאוחר יותר")
    assert result_he["year_min"] == 2015


def test_regex_parse_year_and_earlier_phrasing():
    result = llm._regex_parse("year 2010 and earlier")
    assert result["year_max"] == 2010

    result_he = llm._regex_parse("שנה 2005 ומוקדם יותר")
    assert result_he["year_max"] == 2005


def test_detect_followup_type_other_and_shorter():
    assert llm._detect_followup_type("show me other options") == "other"
    assert llm._detect_followup_type("something shorter please") == "shorter"
    assert llm._detect_followup_type("hello there") is None


def test_keyword_classify_search_chat_and_question():
    assert llm._keyword_classify("recommend me a thriller") == "search"
    assert llm._keyword_classify("thanks!") == "chat"
    assert llm._keyword_classify("how many seasons does it have") == "question"


def test_classify_intent_falls_back_to_keywords_without_provider(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: None)
    monkeypatch.setattr(llm, "_provider", None)
    assert llm.classify_intent("recommend me a thriller") == "search"


def test_fallback_explanation_hebrew_with_seeds():
    intent = {"mood": [], "seeds": ["Breaking Bad"]}
    recs = [{"title": "Better Call Saul", "genres": "Crime, Drama", "rating": 8.7}]
    text = llm._fallback_explanation(intent, recs, "he")
    assert "Breaking Bad" in text
    assert "Better Call Saul" in text


def test_fallback_explanation_handles_nan_rating():
    intent = {"mood": [], "seeds": []}
    recs = [{"title": "Nature Untamed", "genres": "Documentary", "rating": float("nan")}]
    text = llm._fallback_explanation(intent, recs, "en")
    assert "nan" not in text.lower()

    text_he = llm._fallback_explanation(intent, recs, "he")
    assert "nan" not in text_he.lower()


def test_explain_recommendations_no_results():
    assert llm.explain_recommendations({}, [], "en") == "No matching results found."
    assert llm.explain_recommendations({}, [], "he") == "לא נמצאו תוצאות מתאימות."


def test_chat_turn_without_provider_returns_search_with_regex_intent(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: None)
    monkeypatch.setattr(llm, "_provider", None)
    conversation = [{"role": "user", "content": "recommend me a dark thriller"}]
    result = llm.chat_turn(conversation, prev_recs=None, lang="en")
    assert result["action"] == "search"
    assert "dark" in result["intent"]["mood"]


def test_chat_turn_fast_followup_shorter(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: None)
    monkeypatch.setattr(llm, "_provider", None)
    conversation = [{"role": "user", "content": "something shorter please"}]
    prev_recs = [{"title": "Show A", "genres": "Drama", "decade_str": "2010s", "rating": 8.0, "overview": "..."}]
    result = llm.chat_turn(conversation, prev_recs=prev_recs, lang="en")
    assert result["action"] == "refine"
    assert result["intent"]["length_pref"] == "short"


def test_chat_turn_fast_followup_not_too_many_seasons(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: None)
    monkeypatch.setattr(llm, "_provider", None)
    conversation = [{"role": "user", "content": "not too many seasons"}]
    prev_recs = [{"title": "Show A", "genres": "Drama", "decade_str": "2010s", "rating": 8.0, "overview": "..."}]
    result = llm.chat_turn(conversation, prev_recs=prev_recs, lang="en")
    assert result["action"] == "refine"
    assert result["intent"]["length_pref"] == "short"


def test_chat_turn_fast_followup_year_filter(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: None)
    monkeypatch.setattr(llm, "_provider", None)
    conversation = [{"role": "user", "content": "year 2020 and later"}]
    prev_recs = [{"title": "Show A", "genres": "Drama", "decade_str": "2010s", "rating": 8.0, "overview": "..."}]
    result = llm.chat_turn(conversation, prev_recs=prev_recs, lang="en")
    assert result["action"] == "refine"
    assert result["intent"]["year_min"] == 2020
    assert result["intent"]["era_pref"] == "recent"


def test_is_gibberish_flags_only_obvious_keyboard_mashing():
    assert llm._is_gibberish("sjfnkfdsngkjdhf") is True
    assert llm._is_gibberish("hello") is False
    assert llm._is_gibberish("hi") is False
    assert llm._is_gibberish("israeli series") is False
    assert llm._is_gibberish("NASA") is False
    assert llm._is_gibberish("שלום") is False
    assert llm._is_gibberish("abc123") is False


def test_chat_turn_gibberish_fast_path_returns_not_in_catalog_message(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: None)
    monkeypatch.setattr(llm, "_provider", None)
    conversation = [{"role": "user", "content": "sjfnkfdsngkjdhf"}]
    result = llm.chat_turn(conversation, prev_recs=None, lang="en")
    assert result["action"] == "chat"
    assert result["reply"] == "That doesn't seem to be in my catalog, maybe try rephrasing?"
    assert result["intent"]["language_pref"] == "any"
