from app.engine.jaccard import _build_feature_set, jaccard, top_k_jaccard


def test_build_feature_set_includes_genre_decade_language(catalog):
    row = catalog[catalog["title"] == "Game of Thrones"].iloc[0]
    feat = _build_feature_set(row)
    assert "genre:Action" in feat
    assert "genre:Adventure" in feat
    assert "genre:Drama" in feat
    assert "decade:2010s" in feat
    assert "lang:en" in feat


def test_jaccard_identical_sets_is_one():
    a = frozenset({"genre:Drama", "decade:2010s"})
    assert jaccard(a, a) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    a = frozenset({"genre:Drama"})
    b = frozenset({"genre:Comedy"})
    assert jaccard(a, b) == 0.0


def test_top_k_jaccard_returns_k_rows_sorted_desc(catalog):
    result = top_k_jaccard("Game of Thrones", catalog, k=5)
    assert len(result) == 5
    assert "jaccard_score" in result.columns
    scores = result["jaccard_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert "Game of Thrones" not in result["title"].values
