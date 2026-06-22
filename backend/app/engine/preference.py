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


def _era_mask(pool: pd.DataFrame, era: str) -> pd.Series | None:
    bounds = ERA_BOUNDS.get(era)
    if not bounds:
        return None
    lo, hi = bounds
    year = pool["start_year"]
    mask = pd.Series(True, index=pool.index)
    if lo is not None:
        mask &= year.fillna(-1) >= lo
    if hi is not None:
        mask &= year.fillna(99999) <= hi
    return mask


def _length_mask(pool: pd.DataFrame, length: str) -> pd.Series | None:
    bounds = LENGTH_BOUNDS.get(length)
    if not bounds:
        return None
    lo, hi = bounds
    seasons = pd.to_numeric(pool["num_seasons"], errors="coerce")
    mask = pd.Series(True, index=pool.index)
    if lo is not None:
        mask &= seasons >= lo
    if hi is not None:
        mask &= seasons <= hi
    return mask


def _popularity_mask(pool: pd.DataFrame, popularity: str) -> pd.Series | None:
    votes = pool["votes"].fillna(0)
    rating = pool["rating"].fillna(0)
    if popularity == "well_known":
        return votes > 100000
    if popularity == "hidden_gem":
        return (votes < 10000) & (rating > 7.5)
    return None


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

    # GENRE is a hard FLOOR, never relaxed: when the user names a genre we only
    # recommend titles that actually have it. If the genre yields nothing at all,
    # that is a true no-match (caller shows the graceful message).
    genre_cols = [c for c in _target_genre_cols(answers) if c in pool.columns]
    if genre_cols:
        on_genre = pool[pool[genre_cols].max(axis=1) > 0]
        if on_genre.empty:
            return pool.head(0).reset_index(drop=True)
        pool = on_genre

    # Remaining constraints in priority order (era > popularity > length). Length
    # is weakest and dropped FIRST: num_seasons is only ~75% populated and "6+
    # seasons" conflicts with "recent", so it must never strand the user with
    # off-genre filler. Apply all; if that empties, drop the weakest until >= 1
    # survives, keeping the most-constrained set that still matches (dynamic 1-3).
    constraints = []
    era_mask = _era_mask(pool, answers.get("era", "any"))
    if era_mask is not None:
        constraints.append(era_mask)
    pop_mask = _popularity_mask(pool, answers.get("popularity", "any"))
    if pop_mask is not None:
        constraints.append(pop_mask)
    length_mask = _length_mask(pool, answers.get("length", "any"))
    if length_mask is not None:
        constraints.append(length_mask)

    survivors = pool
    active = list(constraints)
    while active:
        combined = pd.Series(True, index=pool.index)
        for m in active:
            combined &= m
        cand = pool[combined]
        if not cand.empty:
            survivors = cand
            break
        active = active[:-1]  # drop the weakest (length, then popularity, then era)

    ranked = survivors.assign(_score=_quality(survivors)).sort_values(
        ["_score", "rating"], ascending=[False, False], na_position="last"
    )
    return _diverse_top_n(ranked, top_n).drop(columns="_score").reset_index(drop=True)
