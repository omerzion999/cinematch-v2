import app.routers.chat as chat_module


def _search_intent(free_text="anything", mood=None, language_pref="any"):
    return {
        "seeds": [], "mood": mood or [], "length_pref": "any",
        "exclude_genres": [], "lang": "en", "language_pref": language_pref,
        "free_text": free_text,
    }


def test_chat_action_chat_returns_reply_without_recommendations(client, monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "chat",
            "intent": _search_intent("greeting"),
            "reply": "Hey, what are you in the mood for?",
            "follow_up": "",
        },
    )

    response = client.post(
        "/api/chat",
        json={"conversation": [{"role": "user", "content": "hi"}], "lang": "en"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Hey, what are you in the mood for?"
    assert body["recommendations"] is None


def test_chat_search_returns_recommendations_and_explanation(client, monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "search",
            "intent": _search_intent("dark thriller", mood=["dark", "thrilling"]),
            "reply": "Try these:",
            "follow_up": "",
        },
    )

    response = client.post(
        "/api/chat",
        json={
            "conversation": [{"role": "user", "content": "something dark and thrilling"}],
            "lang": "en",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 1 <= len(body["recommendations"]) <= 3
    assert body["explanation"]


def test_chat_refine_excludes_previously_shown_titles(client, monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "search",
            "intent": _search_intent(),
            "reply": "Try these:",
            "follow_up": "",
        },
    )
    first = client.post(
        "/api/chat",
        json={"conversation": [{"role": "user", "content": "recommend something"}], "lang": "en"},
    )
    first_recs = first.json()["recommendations"]
    assert len(first_recs) == 3

    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "refine",
            "intent": _search_intent("more like that"),
            "reply": "On it:",
            "follow_up": "",
        },
    )
    second = client.post(
        "/api/chat",
        json={
            "conversation": [{"role": "user", "content": "more"}],
            "prev_recs": first_recs,
            "lang": "en",
        },
    )
    assert second.status_code == 200
    second_recs = second.json()["recommendations"]
    first_titles = {r["title"] for r in first_recs}
    second_titles = {r["title"] for r in second_recs}
    assert second_titles
    assert first_titles.isdisjoint(second_titles)


def test_chat_search_with_israeli_language_pref_returns_hebrew_titles(client, monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "search",
            "intent": _search_intent("israeli series", language_pref="he"),
            "reply": "Sure, here are some Israeli shows:",
            "follow_up": "",
        },
    )

    response = client.post(
        "/api/chat",
        json={"conversation": [{"role": "user", "content": "israeli series"}], "lang": "en"},
    )
    assert response.status_code == 200
    body = response.json()
    recs = body["recommendations"]
    assert recs
    he_titles = {
        "Shtisel", "Prisoners of War", "Valley of Tears", "Shadow of Truth",
        "False Flag", "When Heroes Fly", "The Attach�", "Black Space",
        "Losing Alice", "The Motive",
    }
    assert all(r["title"] in he_titles for r in recs)


def test_chat_search_with_israeli_language_pref_exhausted_returns_not_in_catalog(client, monkeypatch):
    catalog = client.app.state.cinematch["catalog"]
    all_he_titles = catalog[catalog["language"] == "he"]["title"].tolist()
    prev_recs = [
        {
            "title": title, "genres": "Drama", "rating": 7.0, "overview": "",
            "decade_str": "2010s",
        }
        for title in all_he_titles
    ]

    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "search",
            "intent": _search_intent("israeli series", language_pref="he"),
            "reply": "Sure, here are some Israeli shows:",
            "follow_up": "",
        },
    )

    response = client.post(
        "/api/chat",
        json={
            "conversation": [{"role": "user", "content": "more israeli series"}],
            "prev_recs": prev_recs,
            "lang": "en",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] is None
    assert body["reply"] == "That doesn't seem to be in my catalog, maybe try rephrasing?"


def test_chat_swap_slot_with_exhausted_language_pref_returns_not_in_catalog(client, monkeypatch):
    catalog = client.app.state.cinematch["catalog"]
    all_he_titles = catalog[catalog["language"] == "he"]["title"].tolist()
    prev_recs = [
        {
            "title": title, "genres": "Drama", "rating": 7.0, "overview": "",
            "decade_str": "2010s",
        }
        for title in all_he_titles[:3]
    ] + [
        {
            "title": title, "genres": "Drama", "rating": 7.0, "overview": "",
            "decade_str": "2010s",
        }
        for title in all_he_titles[3:]
    ]

    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "swap_slot",
            "intent": _search_intent("already watched the first israeli one", language_pref="he"),
            "reply": "Swapping the first.",
            "swap_slot_index": 0,
            "follow_up": "",
        },
    )

    response = client.post(
        "/api/chat",
        json={
            "conversation": [{"role": "user", "content": "swap the first one"}],
            "prev_recs": prev_recs,
            "lang": "en",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "That doesn't seem to be in my catalog, maybe try rephrasing?"
    assert [r["title"] for r in body["recommendations"]] == [r["title"] for r in prev_recs]


def test_chat_swap_slot_replaces_only_target_slot(client, monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "search",
            "intent": _search_intent(),
            "reply": "Try these:",
            "follow_up": "",
        },
    )
    first = client.post(
        "/api/chat",
        json={"conversation": [{"role": "user", "content": "recommend something"}], "lang": "en"},
    )
    prev_recs = first.json()["recommendations"]
    assert len(prev_recs) == 3

    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "swap_slot",
            "intent": _search_intent("already watched #2"),
            "reply": "Swapping the second.",
            "swap_slot_index": 1,
            "follow_up": "",
        },
    )
    second = client.post(
        "/api/chat",
        json={
            "conversation": [{"role": "user", "content": "swap #2"}],
            "prev_recs": prev_recs,
            "lang": "en",
        },
    )
    assert second.status_code == 200
    new_recs = second.json()["recommendations"]
    assert len(new_recs) == 3
    assert new_recs[0]["title"] == prev_recs[0]["title"]
    assert new_recs[2]["title"] == prev_recs[2]["title"]
    assert new_recs[1]["title"] != prev_recs[1]["title"]
