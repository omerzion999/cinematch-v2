"""
Weighted preference ranker (the live recommendation engine).

Instead of routing a user to one K-Means bucket (the old path, whose buckets
collapsed into near-identical Drama/Comedy blends), this ranks the WHOLE catalog
against the user's stated preferences. That makes the picks genuinely shaped by
the answers: change the genre and the picks change, change the era and the
decade changes, and so on.

Approach (filter then rank):
  1. Hard-filter by era (recent / modern / classic) so the era always bites.
  2. Soft-filter by length and popularity (relaxed if too few titles survive,
     so a thin combo never returns an empty list).
  3. Score the survivors: a strong reward for matching the chosen genre, blended
     with a quality term (engineered binge_fit_score + rating). Genre is a
     reward rather than a hard filter, so a genre that is thin in the chosen era
     degrades gracefully to the best available titles instead of nothing.
  4. Light diversity de-dup, then take the top N (default 3).

`answers` is the onboarding answers dict (genre / length / era / popularity, and
optionally tone for the chat mood path). See app/clustering/onboarding_map.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.clustering.onboarding_map import GENRE_QUESTION_MAP, TONE_GENRE_MAP

# Onboarding era labels -> inclusive start_year bounds (min, max). None = open.
ERA_BOUNDS = {
    "recent": (2020, None),
    "modern": (2010, 2019),
    "classic": (None, 2009),
}

# Onboarding length labels -> inclusive num_seasons bounds (min, max). None = open.
LENGTH_BOUNDS = {
    "short": (None, 2),
    "medium": (3, 5),
    "long": (6, None),
}

# Quality blend: binge_fit_score carries engineered "watchability", rating adds
# raw critical quality. Fixed scales keep the score stable across pools.
_BINGE_SCALE = 8.0
_RATING_SCALE = 10.0
_RATING_FILL = 6.5  # neutral fill so a missing rating does not zero a good binge title
# IMDb-style vote-weighted rating prior: a high rating from very few votes is
# shrunk toward the mean, so a 9.9 with 8 votes does not beat an 8.7 with 50k.
_VOTE_PRIOR = 1000.0

_W_GENRE = 0.6
_W_QUALITY = 0.4

# Soft filters relax when fewer than this many titles survive.
def _min_pool(top_n: int) -> int:
    return max(top_n, 5)


def _target_genre_cols(answers: dict) -> list[str]:
    """Genre feature columns to reward, from the genre answer and/or chat tone."""
    cols: list[str] = []
    genre = answers.get("genre", "any")
    if genre in GENRE_QUESTION_MAP:
        cols.append(f"genre:{GENRE_QUESTION_MAP[genre]}")
    tone = answers.get("tone", "any")
    if tone in TONE_GENRE_MAP:
        col = f"genre:{TONE_GENRE_MAP[tone]}"
        if col not in cols:
            cols.append(col)
    return cols


def _apply_era(pool: pd.DataFrame, era: str) -> pd.DataFrame:
    bounds = ERA_BOUNDS.get(era)
    if not bounds:
        return pool
    lo, hi = bounds
    year = pool["start_year"]
    out = pool
    if lo is not None:
        out = out[year.fillna(-1) >= lo]
    if hi is not None:
        out = out[year.fillna(99999) <= hi]
    # Era is a hard intent, but never strand the user with nothing.
    return out if not out.empty else pool


def _apply_length(pool: pd.DataFrame, length: str, top_n: int) -> pd.DataFrame:
    bounds = LENGTH_BOUNDS.get(length)
    if not bounds:
        return pool
    lo, hi = bounds
    seasons = pd.to_numeric(pool["num_seasons"], errors="coerce")
    mask = pd.Series(True, index=pool.index)
    if lo is not None:
        mask &= seasons >= lo
    if hi is not None:
        mask &= seasons <= hi
    filtered = pool[mask]
    return filtered if len(filtered) >= _min_pool(top_n) else pool


def _apply_popularity(pool: pd.DataFrame, popularity: str, top_n: int) -> pd.DataFrame:
    votes = pool["votes"].fillna(0)
    rating = pool["rating"].fillna(0)
    if popularity == "well_known":
        filtered = pool[votes > 100000]
    elif popularity == "hidden_gem":
        filtered = pool[(votes < 10000) & (rating > 7.5)]
    else:
        return pool
    return filtered if len(filtered) >= _min_pool(top_n) else pool


def _quality(pool: pd.DataFrame) -> np.ndarray:
    binge = pool["binge_fit_score"].fillna(0.0).to_numpy(dtype=float) / _BINGE_SCALE

    rating = pool["rating"].fillna(_RATING_FILL).to_numpy(dtype=float)
    votes = pool["votes"].fillna(0.0).to_numpy(dtype=float)
    prior = float(pool["rating"].mean()) if pool["rating"].notna().any() else _RATING_FILL
    # IMDb weighted rating: (v/(v+m))*R + (m/(v+m))*C
    weighted_rating = (votes * rating + _VOTE_PRIOR * prior) / (votes + _VOTE_PRIOR)
    rating_norm = weighted_rating / _RATING_SCALE

    q = 0.6 * binge + 0.4 * rating_norm
    return np.clip(q, 0.0, 1.0)


def _diverse_top_n(ranked: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Greedy pick that avoids returning multiple rows with the identical genre
    string, falling back to score order if that would leave fewer than top_n."""
    picked_idx: list = []
    seen_genres: set[str] = set()
    for idx, row in ranked.iterrows():
        g = str(row.get("genres", ""))
        if g in seen_genres:
            continue
        picked_idx.append(idx)
        seen_genres.add(g)
        if len(picked_idx) >= top_n:
            break
    if len(picked_idx) < top_n:
        for idx in ranked.index:
            if idx not in picked_idx:
                picked_idx.append(idx)
            if len(picked_idx) >= top_n:
                break
    return ranked.loc[picked_idx].head(top_n)


def rank_by_preferences(
    catalog: pd.DataFrame,
    answers: dict,
    top_n: int = 3,
    exclude_titles: list[str] | set[str] | None = None,
    exclude_genres: list[str] | None = None,
) -> pd.DataFrame:
    """
    Rank the catalog by the user's preferences and return the top_n rows
    (all original columns, index reset).

    catalog : catalog_with_features (original catalog columns + cluster_id +
              FEATURE_DIMS one-hot genre columns), see app/state.py.
    answers : onboarding answers dict (genre / length / era / popularity, plus
              optional tone for the chat mood path).
    """
    pool = catalog
    if exclude_titles:
        pool = pool[~pool["title"].isin(set(exclude_titles))]
    for genre in (exclude_genres or []):
        col = f"genre:{genre}" if f"genre:{genre}" in pool.columns else None
        if col:
            pool = pool[pool[col] == 0.0]
        else:
            pool = pool[~pool["genres"].fillna("").str.contains(genre, case=False, na=False)]

    if pool.empty:
        return pool.head(0).reset_index(drop=True)

    pool = _apply_era(pool, answers.get("era", "any"))
    pool = _apply_length(pool, answers.get("length", "any"), top_n)
    pool = _apply_popularity(pool, answers.get("popularity", "any"), top_n)

    quality = _quality(pool)
    genre_cols = [c for c in _target_genre_cols(answers) if c in pool.columns]
    if genre_cols:
        genre_match = pool[genre_cols].max(axis=1).to_numpy(dtype=float)
        score = _W_GENRE * genre_match + _W_QUALITY * quality
    else:
        score = quality

    ranked = pool.assign(_score=score).sort_values(
        ["_score", "rating"], ascending=[False, False], na_position="last"
    )
    result = _diverse_top_n(ranked, top_n).drop(columns="_score")
    return result.reset_index(drop=True)
