"""
Hybrid scorer.
Combines Jaccard, Cosine-numeric, and Cosine-text into one ranked list.

Score = α·Jaccard + β·Cosine_numeric + (1-α-β)·Cosine_text

Default weights tuned on a 10-pair hand-rated preference set:
  α = 0.35  (genre/era matching matters most)
  β = 0.30  (numeric profile: rating, popularity)
  γ = 0.35  (semantic synopsis similarity)
"""

import re

import numpy as np
import pandas as pd
from app.catalog_lookup import find_catalog_index
from app.engine.jaccard import top_k_jaccard
from app.engine.cosine import top_k_cosine_numeric, top_k_cosine_text

ALPHA = 0.35   # Jaccard weight (genre/era)
BETA  = 0.30   # Cosine-numeric weight (rating/popularity profile)
# GAMMA = 1 - ALPHA - BETA = 0.35  → Cosine-text weight (semantic plot)

# ── Bias mitigation ───────────────────────────────────────────────────────────
# The multilingual model performs better on English text; Hebrew embedding quality
# is lower. When the user queries in Hebrew we up-weight Jaccard (language-neutral,
# genre-set based) and down-weight text-cosine (semantic embedding).
HEBREW_WEIGHTS = {"alpha": 0.50, "beta": 0.30, "gamma": 0.20}


def apply_filters(catalog: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Hard-filter a candidate DataFrame by intent filters.
    Year/era filters are always enforced — no fallback to unfiltered results
    when a hard year constraint is present. Other soft filters fall back to
    the full input if fewer than 3 rows survive.
    """
    if not filters:
        return catalog
    df = catalog.copy()

    # ── Hard year filters (ALWAYS enforced, no fallback) ─────────────────────
    year_hard = False

    year_min = filters.get("year_min")
    year_max = filters.get("year_max")
    if year_min:
        df = df[df["start_year"].fillna(0) >= year_min]
        year_hard = True
    if year_max:
        df = df[df["start_year"].fillna(9999) <= year_max]
        year_hard = True

    era = filters.get("era_pref")
    if era and era != "any":
        year_hard = True
        if era in ("recent", "2020s"):
            df = df[df["start_year"].fillna(0) >= 2020]
        elif era == "2010s":
            df = df[(df["start_year"].fillna(0) >= 2010) & (df["start_year"].fillna(0) < 2020)]
        elif era == "2000s":
            df = df[(df["start_year"].fillna(0) >= 2000) & (df["start_year"].fillna(0) < 2010)]
        elif era == "1990s":
            df = df[(df["start_year"].fillna(0) >= 1990) & (df["start_year"].fillna(0) < 2000)]
        elif era == "classic":
            df = df[df["start_year"].fillna(9999) < 2000]

    # If hard year filter left nothing, return empty so caller shows "few results"
    if year_hard and len(df) == 0:
        return df

    # ── Soft filters (skip if fewer than 3 would survive) ────────────────────
    pre_soft = df.copy()

    # length_pref
    lp = filters.get("length_pref")
    if lp == "short":
        tmp = df[df["num_seasons"].fillna(99) <= 2]
        if len(tmp) >= 3: df = tmp
    elif lp == "long":
        tmp = df[df["num_seasons"].fillna(0) >= 4]
        if len(tmp) >= 3: df = tmp
    elif lp == "limited":
        tmp = df[df["num_seasons"].fillna(0) == 1]
        if len(tmp) >= 3: df = tmp

    # language_pref
    lang_col = "language" if "language" in df.columns else (
        "original_language" if "original_language" in df.columns else None)
    if lang_col:
        lpr = filters.get("language_pref")
        if lpr == "foreign":
            tmp = df[df[lang_col] != "en"]
            if len(tmp) >= 3: df = tmp
        elif lpr not in (None, "any"):
            tmp = df[df[lang_col] == lpr]
            if len(tmp) >= 3: df = tmp

    # status
    status = filters.get("status")
    if status == "airing":
        tmp = df[df["end_year"].isna()]
        if len(tmp) >= 3: df = tmp
    elif status == "finished":
        tmp = df[df["end_year"].notna()]
        if len(tmp) >= 3: df = tmp

    # popularity_pref
    pop = filters.get("popularity_pref")
    if pop == "hidden_gem":
        tmp = df[(df["votes"].fillna(0) < 10000) & (df["rating"].fillna(0) > 7.5)]
        if len(tmp) >= 3: df = tmp
    elif pop == "trending":
        df = df.sort_values("popularity", ascending=False)
    elif pop == "well_known":
        tmp = df[df["votes"].fillna(0) > 100000]
        if len(tmp) >= 3: df = tmp

    # binge_pref
    if filters.get("binge_pref") == "binge" and "binge_fit_score" in df.columns:
        df = df.sort_values("binge_fit_score", ascending=False)

    # rating_min
    rmin = filters.get("rating_min")
    if rmin:
        tmp = df[df["rating"].fillna(0) >= rmin]
        if len(tmp) >= 3: df = tmp

    genre_col = "genre_set_str" if "genre_set_str" in df.columns else "genres"

    # include_genres (soft – keep rows matching ANY of the genres, OR logic)
    include_genres = filters.get("include_genres") or []
    if include_genres:
        pattern = "|".join(re.escape(g) for g in include_genres)
        tmp = df[df[genre_col].fillna("").str.contains(pattern, case=False, na=False, regex=True)]
        if len(tmp) >= 3:
            df = tmp

    # exclude_genres
    for genre in (filters.get("exclude_genres") or []):
        tmp = df[~df[genre_col].fillna("").str.contains(genre, case=False, na=False)]
        if len(tmp) >= 3:
            df = tmp

    return df


def recommend(
    query_title: str,
    catalog: pd.DataFrame,
    numeric_matrix: np.ndarray,
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_n: int = 5,
    alpha: float = ALPHA,
    beta: float = BETA,
    exclude_titles: set = None,
    query_lang: str = "en",
    filters: dict = None,
) -> pd.DataFrame:
    # ── Bias mitigation: re-weight for Hebrew queries ───────────────────────
    if query_lang == "he":
        alpha = HEBREW_WEIGHTS["alpha"]
        beta  = HEBREW_WEIGHTS["beta"]
        # gamma derived below = 0.20
    """
    Returns top_n recommendations as a DataFrame with columns:
      title, genres, rating, votes, decade_str, overview, poster_path,
      jaccard_score, cosine_numeric_score, cosine_text_score, hybrid_score
    """
    gamma = 1.0 - alpha - beta
    exclude_titles = exclude_titles or set()

    # Get candidates from each method (union pool)
    j_res = top_k_jaccard(query_title, catalog, k=100, exclude_titles=exclude_titles)
    n_res = top_k_cosine_numeric(query_title, catalog, numeric_matrix, k=100,
                                 exclude_titles=exclude_titles)
    t_res = top_k_cosine_text(query_embedding, catalog, embeddings, k=100,
                              exclude_titles=exclude_titles, query_title=query_title)

    # Build a unified score table indexed by title
    scores: dict[str, dict] = {}

    def _add(df: pd.DataFrame, col: str):
        for _, row in df.iterrows():
            t = row["title"]
            if t not in scores:
                scores[t] = {
                    "jaccard_score": 0.0,
                    "cosine_numeric_score": 0.0,
                    "cosine_text_score": 0.0,
                    "_row": row,
                }
            scores[t][col] = float(row.get(col, 0.0) or 0.0)

    _add(j_res, "jaccard_score")
    _add(n_res, "cosine_numeric_score")
    _add(t_res, "cosine_text_score")

    # Compute hybrid
    for t in scores:
        d = scores[t]
        d["hybrid_score"] = (
            alpha * d["jaccard_score"]
            + beta  * d["cosine_numeric_score"]
            + gamma * d["cosine_text_score"]
        )

    if not scores:
        return pd.DataFrame()

    rows = []
    for t, d in sorted(scores.items(), key=lambda x: -x[1]["hybrid_score"]):
        r = d["_row"].copy()
        r["jaccard_score"]        = round(d["jaccard_score"], 4)
        r["cosine_numeric_score"] = round(d["cosine_numeric_score"], 4)
        r["cosine_text_score"]    = round(d["cosine_text_score"], 4)
        r["hybrid_score"]         = round(d["hybrid_score"], 4)
        rows.append(r)

    # Fetch a wide pool so year/era filters have rows to work with,
    # then apply hard filters without falling back to unfiltered data.
    candidates = pd.DataFrame(rows).reset_index(drop=True)
    if filters:
        candidates = apply_filters(candidates, filters)
    result = candidates.head(top_n).reset_index(drop=True)
    return result


def recommend_from_seeds(
    seed_titles: list[str],
    catalog: pd.DataFrame,
    numeric_matrix: np.ndarray,
    embeddings: np.ndarray,
    top_n: int = 3,
    exclude_titles: set = None,
    filters: dict = None,
    query_lang: str = "en",
    per_seed_k: int = 100,
) -> pd.DataFrame:
    """
    Multi-seed similarity: recommend titles similar to a set of liked shows.

    For each seed we run the single-seed hybrid recommender and then merge the
    candidate pools by the MAX hybrid_score a title earns across seeds (a title
    that is a strong neighbour of ANY picked seed surfaces). The seeds themselves
    are excluded, era/popularity filters (if any) are applied to the merged pool,
    and the top_n (dynamic 1..top_n) are returned with the same columns as
    `recommend`.
    """
    exclude_titles = set(exclude_titles or set())

    # Resolve seeds to their canonical catalog titles; exclude them from results.
    resolved: list[str] = []
    for title in seed_titles:
        idx = find_catalog_index(catalog, title)
        if idx is None:
            continue
        resolved.append(catalog.iloc[idx]["title"])
    if not resolved:
        return pd.DataFrame()
    exclude_titles |= set(resolved)

    # Best row + best score per candidate title across all seeds.
    best: dict[str, dict] = {}
    for seed_title in resolved:
        idx = find_catalog_index(catalog, seed_title)
        res = recommend(
            query_title=seed_title,
            catalog=catalog,
            numeric_matrix=numeric_matrix,
            embeddings=embeddings,
            query_embedding=embeddings[idx],
            top_n=per_seed_k,
            exclude_titles=exclude_titles,
            query_lang=query_lang,
            filters=None,
        )
        for _, row in res.iterrows():
            t = row["title"]
            if t not in best or row["hybrid_score"] > best[t]["hybrid_score"]:
                best[t] = row
    if not best:
        return pd.DataFrame()

    merged = pd.DataFrame(list(best.values())).sort_values(
        "hybrid_score", ascending=False
    ).reset_index(drop=True)

    # Era/popularity are OPTIONAL refine answers in the seed flow: prefer the
    # filtered pool, but never strand the user with nothing when their seeds
    # genuinely matched. Fall back to the unfiltered neighbours if the filter
    # empties the pool.
    if filters:
        filtered = apply_filters(merged, filters)
        if not filtered.empty:
            merged = filtered
    return merged.head(top_n).reset_index(drop=True)


def score_all_pairs(
    catalog: pd.DataFrame,
    numeric_matrix: np.ndarray,
    embeddings: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> np.ndarray:
    """Compute the full N×N hybrid matrix (used by anomaly + evaluate)."""
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos
    from app.engine.jaccard import batch_jaccard_matrix

    gamma = 1.0 - alpha - beta

    j_mat = batch_jaccard_matrix(catalog).astype(np.float32)
    n_mat = sk_cos(numeric_matrix.astype(np.float32)).astype(np.float32)
    # embeddings are already L2-normalized → dot = cosine
    t_mat = (embeddings @ embeddings.T).astype(np.float32)

    return alpha * j_mat + beta * n_mat + gamma * t_mat
