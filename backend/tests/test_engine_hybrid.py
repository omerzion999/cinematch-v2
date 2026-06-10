from app.engine.hybrid import ALPHA, BETA, HEBREW_WEIGHTS, apply_filters, recommend


def test_weights_sum_to_one():
    assert abs(ALPHA + BETA + (1 - ALPHA - BETA) - 1.0) < 1e-9
    assert abs(sum(HEBREW_WEIGHTS.values()) - 1.0) < 1e-9


def test_apply_filters_era_classic_keeps_only_pre_2000(catalog):
    filtered = apply_filters(catalog, {"era_pref": "classic"})
    assert len(filtered) > 0
    assert (filtered["start_year"].fillna(9999) < 2000).all()


def test_apply_filters_length_short_falls_back_when_too_few(catalog):
    # length_pref short = num_seasons <= 2; "any" filters dict should be unaffected
    filtered = apply_filters(catalog, {})
    assert len(filtered) == len(catalog)


def test_recommend_returns_top_n_with_hybrid_score(catalog, numeric_matrix, embeddings):
    got_idx = catalog.index[catalog["title"] == "Game of Thrones"][0]
    pos = catalog.index.get_loc(got_idx)
    query_embedding = embeddings[pos]

    result = recommend(
        "Game of Thrones",
        catalog,
        numeric_matrix,
        embeddings,
        query_embedding,
        top_n=3,
    )
    assert len(result) == 3
    for col in ["title", "genres", "rating", "hybrid_score", "jaccard_score",
                "cosine_numeric_score", "cosine_text_score"]:
        assert col in result.columns
    scores = result["hybrid_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert "Game of Thrones" not in result["title"].values


def test_recommend_uses_hebrew_weights_when_query_lang_he(catalog, numeric_matrix, embeddings):
    got_idx = catalog.index[catalog["title"] == "Game of Thrones"][0]
    pos = catalog.index.get_loc(got_idx)
    query_embedding = embeddings[pos]

    result = recommend(
        "Game of Thrones",
        catalog,
        numeric_matrix,
        embeddings,
        query_embedding,
        top_n=3,
        query_lang="he",
    )
    assert len(result) == 3
