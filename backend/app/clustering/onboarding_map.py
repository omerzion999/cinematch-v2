"""
Maps onboarding answers (or a chat_turn intent dict) to a 14-dim target
vector + boolean mask aligned to FEATURE_DIMS, for use by
clustering/recommend.py.

Onboarding answers dict shape (all 5 keys always present):
  {
    "genre": "drama" | "comedy" | "action_adventure" | "scifi_fantasy"
             | "crime" | "animation" | "romance" | "any",
    "length": "short" | "medium" | "long" | "any",
    "era": "recent" | "classic" | "any",
    "tone": "light_fun" | "serious_drama" | "thriller_action" | "any",
    "popularity": "well_known" | "hidden_gem" | "any",
  }
"""

import numpy as np

from app.clustering.features import FEATURE_DIMS

GENRE_QUESTION_MAP = {
    "drama": "Drama",
    "comedy": "Comedy",
    "action_adventure": "Action & Adventure",
    "scifi_fantasy": "Sci-Fi & Fantasy",
    "crime": "Crime",
    "animation": "Animation",
    "romance": "Romance",
}

TONE_GENRE_MAP = {
    "light_fun": "Comedy",
    "serious_drama": "Drama",
    "thriller_action": "Action & Adventure",
}

LENGTH_Z_MAP = {
    "short": -0.6,
    "medium": 0.0,
    "long": 1.0,
}

ERA_Z_MAP = {
    "recent": 0.8,
    "modern": 0.1,
    "classic": -1.2,
}

POPULARITY_Z_MAP = {
    "well_known": {"popularity_z": 1.0},
    "hidden_gem": {"rating_z": 1.2, "popularity_z": -0.3},
}

# chat_turn intent.mood tags -> onboarding "tone" answer
MOOD_TO_TONE = {
    "funny": "light_fun",
    "light": "light_fun",
    "dark": "thriller_action",
    "thrilling": "thriller_action",
    "emotional": "serious_drama",
}

# chat_turn intent.length_pref -> onboarding "length" answer
LENGTH_PREF_MAP = {
    "short": "short",
    "limited": "short",
    "long": "long",
}

# chat_turn intent.era_pref -> onboarding "era" answer
ERA_PREF_MAP = {
    "recent": "recent",
    "2020s": "recent",
    "2010s": "modern",
    "classic": "classic",
    "1990s": "classic",
    "2000s": "classic",
}

# chat_turn intent.popularity_pref -> onboarding "popularity" answer
POPULARITY_PREF_MAP = {
    "hidden_gem": "hidden_gem",
    "well_known": "well_known",
    "trending": "well_known",
}


def build_user_vector(answers: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (vector, mask), both shape (14,), aligned to FEATURE_DIMS.
    `mask[i] is True` means the user expressed a preference for dim i;
    `vector[i]` is only meaningful where `mask[i]` is True.
    """
    vector = np.zeros(len(FEATURE_DIMS), dtype=np.float32)
    mask = np.zeros(len(FEATURE_DIMS), dtype=bool)

    genre = answers.get("genre", "any")
    if genre in GENRE_QUESTION_MAP:
        idx = FEATURE_DIMS.index(f"genre:{GENRE_QUESTION_MAP[genre]}")
        vector[idx] = 1.0
        mask[idx] = True

    tone = answers.get("tone", "any")
    if tone in TONE_GENRE_MAP:
        idx = FEATURE_DIMS.index(f"genre:{TONE_GENRE_MAP[tone]}")
        vector[idx] = 1.0
        mask[idx] = True

    length = answers.get("length", "any")
    if length in LENGTH_Z_MAP:
        idx = FEATURE_DIMS.index("num_seasons_z")
        vector[idx] = LENGTH_Z_MAP[length]
        mask[idx] = True

    era = answers.get("era", "any")
    if era in ERA_Z_MAP:
        idx = FEATURE_DIMS.index("start_year_z")
        vector[idx] = ERA_Z_MAP[era]
        mask[idx] = True

    popularity = answers.get("popularity", "any")
    if popularity in POPULARITY_Z_MAP:
        for dim, value in POPULARITY_Z_MAP[popularity].items():
            idx = FEATURE_DIMS.index(dim)
            vector[idx] = value
            mask[idx] = True

    return vector, mask


def intent_to_onboarding_answers(intent: dict) -> dict:
    """
    Adapts a chat_turn `intent` dict (see agent/llm.py) to the onboarding
    answers vocabulary, so seedless chat searches can be routed through
    build_user_vector -> recommend_from_cluster.
    """
    answers = {"genre": "any", "length": "any", "era": "any", "tone": "any", "popularity": "any"}

    # Explicit genre (from the offline parser or the LLM) maps straight through.
    genre = intent.get("genre", "any")
    if genre in GENRE_QUESTION_MAP:
        answers["genre"] = genre

    for mood in (intent.get("mood") or []):
        if mood in MOOD_TO_TONE:
            answers["tone"] = MOOD_TO_TONE[mood]
            break

    length_pref = intent.get("length_pref", "any")
    if length_pref in LENGTH_PREF_MAP:
        answers["length"] = LENGTH_PREF_MAP[length_pref]

    era_pref = intent.get("era_pref", "any")
    if era_pref in ERA_PREF_MAP:
        answers["era"] = ERA_PREF_MAP[era_pref]

    popularity_pref = intent.get("popularity_pref", "any")
    if popularity_pref in POPULARITY_PREF_MAP:
        answers["popularity"] = POPULARITY_PREF_MAP[popularity_pref]

    return answers
