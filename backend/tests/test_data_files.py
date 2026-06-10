import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "data")


def test_catalog_loads_with_expected_shape():
    catalog = pd.read_parquet(os.path.join(DATA_DIR, "catalog.parquet"))
    assert len(catalog) == 11013
    expected_cols = {
        "title", "language", "start_year", "end_year", "genres", "rating",
        "votes", "popularity", "overview", "poster_path", "num_episodes",
        "num_seasons", "source_dataset", "decade", "decade_str",
        "genre_set_str", "rating_z", "votes_z", "start_year_z",
        "popularity_z", "rating_bucket", "binge_fit_score",
    }
    assert expected_cols.issubset(set(catalog.columns))


def test_embeddings_match_catalog_rows():
    catalog = pd.read_parquet(os.path.join(DATA_DIR, "catalog.parquet"))
    embeddings = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
    assert embeddings.shape == (len(catalog), 384)
    # L2-normalized
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
