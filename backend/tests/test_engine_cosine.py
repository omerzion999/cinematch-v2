import numpy as np

from app.engine.cosine import (
    NUMERIC_COLS,
    build_numeric_matrix,
    top_k_cosine_numeric,
    top_k_cosine_text,
)


def test_numeric_cols_are_the_four_z_scores():
    assert NUMERIC_COLS == ["rating_z", "votes_z", "start_year_z", "popularity_z"]


def test_build_numeric_matrix_returns_feature_matrix(catalog):
    matrix = build_numeric_matrix(catalog)
    assert matrix.shape == (len(catalog), len(NUMERIC_COLS))
    expected = catalog[NUMERIC_COLS].fillna(0).values.astype(np.float32)
    assert np.allclose(matrix, expected)


def test_top_k_cosine_numeric_returns_k_rows(catalog, numeric_matrix):
    result = top_k_cosine_numeric("Game of Thrones", catalog, numeric_matrix, k=5)
    assert len(result) == 5
    assert "cosine_numeric_score" in result.columns
    assert "Game of Thrones" not in result["title"].values


def test_top_k_cosine_text_returns_k_rows(catalog, embeddings):
    got_idx = catalog.index[catalog["title"] == "Game of Thrones"][0]
    pos = catalog.index.get_loc(got_idx)
    query_embedding = embeddings[pos]

    result = top_k_cosine_text(
        query_embedding, catalog, embeddings, k=5, query_title="Game of Thrones"
    )
    assert len(result) == 5
    assert "cosine_text_score" in result.columns
    assert "Game of Thrones" not in result["title"].values
