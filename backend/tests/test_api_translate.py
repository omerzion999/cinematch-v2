import app.routers.translate as translate_module


def test_translate_empty_texts_returns_empty_list(client):
    response = client.post("/api/translate", json={"texts": [], "target_lang": "he"})
    assert response.status_code == 200
    assert response.json() == {"translations": []}


def test_translate_without_provider_returns_texts_unchanged(client, monkeypatch):
    monkeypatch.setattr(translate_module, "_get_client", lambda: None)

    response = client.post(
        "/api/translate",
        json={"texts": ["Hello there", "Try these:"], "target_lang": "he"},
    )
    assert response.status_code == 200
    assert response.json() == {"translations": ["Hello there", "Try these:"]}


def test_translate_with_provider_returns_llm_translations(client, monkeypatch):
    monkeypatch.setattr(translate_module, "_get_client", lambda: True)
    monkeypatch.setattr(
        translate_module,
        "_call_llm",
        lambda system, user, max_tokens=600: '["שלום שם", "נסה את אלה:"]',
    )

    response = client.post(
        "/api/translate",
        json={"texts": ["Hello there", "Try these:"], "target_lang": "he"},
    )
    assert response.status_code == 200
    assert response.json() == {"translations": ["שלום שם", "נסה את אלה:"]}


def test_translate_with_malformed_llm_response_falls_back_to_original(client, monkeypatch):
    monkeypatch.setattr(translate_module, "_get_client", lambda: True)
    monkeypatch.setattr(translate_module, "_call_llm", lambda system, user, max_tokens=600: "not json")

    response = client.post(
        "/api/translate",
        json={"texts": ["Hello there"], "target_lang": "he"},
    )
    assert response.status_code == 200
    assert response.json() == {"translations": ["Hello there"]}


def test_translate_with_mismatched_length_falls_back_to_original(client, monkeypatch):
    monkeypatch.setattr(translate_module, "_get_client", lambda: True)
    monkeypatch.setattr(
        translate_module,
        "_call_llm",
        lambda system, user, max_tokens=600: '["only one"]',
    )

    response = client.post(
        "/api/translate",
        json={"texts": ["Hello there", "Try these:"], "target_lang": "he"},
    )
    assert response.status_code == 200
    assert response.json() == {"translations": ["Hello there", "Try these:"]}


def test_translate_with_no_provider_returns_none_response(client, monkeypatch):
    monkeypatch.setattr(translate_module, "_get_client", lambda: True)
    monkeypatch.setattr(translate_module, "_call_llm", lambda system, user, max_tokens=600: None)

    response = client.post(
        "/api/translate",
        json={"texts": ["Hello there"], "target_lang": "he"},
    )
    assert response.status_code == 200
    assert response.json() == {"translations": ["Hello there"]}
