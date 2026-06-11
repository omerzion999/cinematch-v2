from app.i18n import t


def test_t_returns_hebrew_string_with_formatting():
    result = t("recommend_intro", "he", label="דרמות פשע")
    assert "דרמות פשע" in result


def test_t_returns_english_string_with_formatting():
    result = t("recommend_intro", "en", label="Crime Dramas")
    assert "Crime Dramas" in result


def test_t_falls_back_to_english_for_unknown_lang():
    assert t("show_not_found", "fr") == t("show_not_found", "en")


def test_t_unknown_key_returns_empty_string():
    assert t("nonexistent_key", "en") == ""


def test_t_defaults_to_english_lang():
    assert t("show_not_found") == t("show_not_found", "en")
