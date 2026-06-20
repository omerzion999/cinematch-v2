import numpy as np

from app.clustering.features import FEATURE_DIMS
from app.clustering.onboarding_map import build_user_vector, intent_to_onboarding_answers


def _idx(dim: str) -> int:
    return FEATURE_DIMS.index(dim)


def test_build_user_vector_all_any_returns_empty_mask():
    answers = {"genre": "any", "length": "any", "era": "any", "tone": "any", "popularity": "any"}
    vector, mask = build_user_vector(answers)
    assert vector.shape == (14,)
    assert mask.shape == (14,)
    assert mask.dtype == bool
    assert not mask.any()
    assert np.allclose(vector, 0.0)


def test_build_user_vector_genre_sets_one_hot_dim():
    answers = {"genre": "drama", "length": "any", "era": "any", "tone": "any", "popularity": "any"}
    vector, mask = build_user_vector(answers)
    assert vector[_idx("genre:Drama")] == 1.0
    assert mask[_idx("genre:Drama")] is np.True_ or mask[_idx("genre:Drama")] == True
    assert mask.sum() == 1


def test_build_user_vector_tone_sets_genre_dim():
    answers = {"genre": "any", "length": "any", "era": "any", "tone": "thriller_action", "popularity": "any"}
    vector, mask = build_user_vector(answers)
    assert vector[_idx("genre:Action & Adventure")] == 1.0
    assert mask[_idx("genre:Action & Adventure")]
    assert mask.sum() == 1


def test_build_user_vector_length_and_era():
    answers = {"genre": "any", "length": "short", "era": "recent", "tone": "any", "popularity": "any"}
    vector, mask = build_user_vector(answers)
    assert vector[_idx("num_seasons_z")] == -0.6
    assert mask[_idx("num_seasons_z")]
    assert vector[_idx("start_year_z")] == 0.8
    assert mask[_idx("start_year_z")]
    assert mask.sum() == 2


def test_build_user_vector_modern_era_sets_start_year_dim():
    answers = {"genre": "any", "length": "any", "era": "modern", "popularity": "any"}
    vector, mask = build_user_vector(answers)
    assert vector[_idx("start_year_z")] == 0.1
    assert mask[_idx("start_year_z")]
    assert mask.sum() == 1


def test_build_user_vector_hidden_gem_sets_two_dims():
    answers = {"genre": "any", "length": "any", "era": "any", "tone": "any", "popularity": "hidden_gem"}
    vector, mask = build_user_vector(answers)
    assert vector[_idx("rating_z")] == 1.2
    assert vector[_idx("popularity_z")] == -0.3
    assert mask[_idx("rating_z")] and mask[_idx("popularity_z")]
    assert mask.sum() == 2


def test_intent_to_onboarding_answers_defaults_to_any():
    answers = intent_to_onboarding_answers({})
    assert answers == {"genre": "any", "length": "any", "era": "any", "tone": "any", "popularity": "any"}


def test_intent_to_onboarding_answers_maps_mood_length_era_popularity():
    intent = {
        "mood": ["funny"],
        "length_pref": "short",
        "era_pref": "classic",
        "popularity_pref": "hidden_gem",
    }
    answers = intent_to_onboarding_answers(intent)
    assert answers == {"genre": "any", "length": "short", "era": "classic", "tone": "light_fun", "popularity": "hidden_gem"}


def test_intent_to_onboarding_answers_dark_mood_maps_to_thriller_action():
    intent = {"mood": ["dark"], "length_pref": "any", "era_pref": "any", "popularity_pref": "any"}
    answers = intent_to_onboarding_answers(intent)
    assert answers["tone"] == "thriller_action"


def test_intent_to_onboarding_answers_trending_maps_to_well_known():
    intent = {"mood": [], "length_pref": "any", "era_pref": "any", "popularity_pref": "trending"}
    answers = intent_to_onboarding_answers(intent)
    assert answers["popularity"] == "well_known"


def test_intent_to_onboarding_answers_mood_null_doesnt_crash():
    intent = {"mood": None, "length_pref": "any", "era_pref": "any", "popularity_pref": "any"}
    answers = intent_to_onboarding_answers(intent)
    assert answers["tone"] == "any"
    assert answers == {"genre": "any", "length": "any", "era": "any", "tone": "any", "popularity": "any"}
