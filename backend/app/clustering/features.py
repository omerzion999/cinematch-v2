"""
Feature vector construction for K-Means clustering and the cluster-based
recommender.

Each title is represented as a 14-dim vector:
  - 10 genre one-hot dims (1.0 if the genre string appears in the title's
    comma-separated `genres` column, else 0.0)
  - 4 numeric z-score dims: rating_z, popularity_z, start_year_z, num_seasons_z

`rating_z`, `popularity_z`, and `start_year_z` are already precomputed in
catalog.parquet. `num_seasons_z` is computed here: missing `num_seasons`
values are imputed with the catalog median before z-scoring, so titles with
no season-count data do not get an arbitrary/extreme value in the feature
space (per the design spec's edge-case notes).
"""

import numpy as np
import pandas as pd

GENRE_DIMS = [
    "Drama", "Comedy", "Animation", "Crime", "Action & Adventure",
    "Sci-Fi & Fantasy", "Mystery", "Documentary", "Family", "Romance",
]
NUMERIC_DIMS = ["rating_z", "popularity_z", "start_year_z", "num_seasons_z"]
FEATURE_DIMS = [f"genre:{g}" for g in GENRE_DIMS] + NUMERIC_DIMS


def _genre_set(genres_str) -> set:
    if not genres_str:
        return set()
    return {g.strip() for g in str(genres_str).split(",") if g.strip()}


def build_cluster_features(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ["title"] + FEATURE_DIMS, one row per
    catalog title, in the same row order as `catalog`.
    """
    out = pd.DataFrame({"title": catalog["title"].values})

    genre_sets = catalog["genres"].fillna("").apply(_genre_set)
    for genre in GENRE_DIMS:
        out[f"genre:{genre}"] = genre_sets.apply(
            lambda gs, g=genre: 1.0 if g in gs else 0.0
        ).astype(np.float32)

    out["rating_z"] = catalog["rating_z"].fillna(0.0).astype(np.float32).values
    out["popularity_z"] = catalog["popularity_z"].fillna(0.0).astype(np.float32).values
    out["start_year_z"] = catalog["start_year_z"].fillna(0.0).astype(np.float32).values

    num_seasons = pd.to_numeric(catalog["num_seasons"], errors="coerce")
    median = num_seasons.median()
    # Guard against all-NaN case: if median is NaN, use 0.0 as fallback
    if np.isnan(median):
        median = 0.0
    num_seasons = num_seasons.fillna(median)
    mean, std = num_seasons.mean(), num_seasons.std()
    # Guard against zero std (e.g., single row or all identical values)
    if not std or np.isnan(std):
        std = 1.0
    if np.isnan(mean):
        mean = 0.0
    out["num_seasons_z"] = ((num_seasons - mean) / std).astype(np.float32).values

    return out
