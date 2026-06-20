def test_full_onboarding_to_chat_flow(client):
    rec_response = client.post(
        "/api/recommend",
        json={
            "answers": {
                "genre": "drama",
                "length": "long",
                "era": "recent",
                "popularity": "well_known",
            },
            "lang": "en",
        },
    )
    assert rec_response.status_code == 200
    rec_body = rec_response.json()
    assert len(rec_body["recommendations"]) == 3
    assert rec_body["intro"]
    assert rec_body["outro"]

    first_title = rec_body["recommendations"][0]["title"]

    show_response = client.get(f"/api/show/{first_title}")
    assert show_response.status_code == 200
    assert show_response.json()["title"] == first_title

    chat_response = client.post(
        "/api/chat",
        json={
            "conversation": [
                {"role": "assistant", "content": rec_body["intro"]},
                {"role": "user", "content": "thanks, anything else?"},
            ],
            "prev_recs": rec_body["recommendations"],
            "lang": "en",
        },
    )
    assert chat_response.status_code == 200
    assert "reply" in chat_response.json()


def test_health_check_works_without_a_frontend_build(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
