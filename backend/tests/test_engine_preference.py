"""
The weighted preference ranker is the fix for "not personal enough": different
answers must produce clearly different picks. These tests assert that on the
real catalog.
"""

import pandas as pd

from app.engine.preference import rank_by_preferences


def _titles(df: pd.DataFrame) -> list[str]:
    return df["title"].tolist()


def _ans(**kw) -> dict:
    base = {"genre": "any", "length": "any", "era": "any", "popularity": "any"}
    base.update(kw)
    return base


def test_genre_shapes_picks(catalog_with_features):
    comedy = rank_by_preferences(catalog_with_features, _ans(genre="comedy"), top_n=3)
    crime = rank_by_preferences(catalog_with_features, _ans(genre="crime"), top_n=3)

    # Each pick matches the chosen genre, and the two sets are disjoint.
    assert all("Comedy" in g for g in comedy["genres"])
    assert all("Crime" in g for g in crime["genres"])
    assert set(_titles(comedy)).isdisjoint(set(_titles(crime)))


def test_era_shapes_decade(catalog_with_features):
    recent = rank_by_preferences(catalog_with_features, _ans(era="recent"), top_n=3)
    classic = rank_by_preferences(catalog_with_features, _ans(era="classic"), top_n=3)

    assert (recent["start_year"] >= 2020).all()
    assert (classic["start_year"] <= 2009).all()
    assert set(_titles(recent)).isdisjoint(set(_titles(classic)))


def test_popularity_shapes_votes(catalog_with_features):
    hits = rank_by_preferences(catalog_with_features, _ans(popularity="well_known"), top_n=3)
    gems = rank_by_preferences(catalog_with_features, _ans(popularity="hidden_gem"), top_n=3)

    assert (hits["votes"].fillna(0) > 100000).all()
    assert (gems["votes"].fillna(0) < 10000).all()


def test_length_shapes_seasons(catalog_with_features):
    short = rank_by_preferences(catalog_with_features, _ans(length="short"), top_n=3)
    long_ = rank_by_preferences(catalog_with_features, _ans(length="long"), top_n=3)

    assert (pd.to_numeric(short["num_seasons"], errors="coerce") <= 2).all()
    assert (pd.to_numeric(long_["num_seasons"], errors="coerce") >= 6).all()


def test_all_any_returns_high_quality_picks(catalog_with_features):
    picks = rank_by_preferences(catalog_with_features, _ans(), top_n=3)
    assert len(picks) == 3
    # With no preference, picks should be strong on the engineered quality column.
    assert picks["binge_fit_score"].min() >= catalog_with_features["binge_fit_score"].median()


def test_exclude_titles_are_dropped(catalog_with_features):
    first = rank_by_preferences(catalog_with_features, _ans(genre="comedy"), top_n=3)
    excluded = set(_titles(first))
    second = rank_by_preferences(
        catalog_with_features, _ans(genre="comedy"), top_n=3, exclude_titles=excluded
    )
    assert excluded.isdisjoint(set(_titles(second)))


def test_genre_is_a_floor_not_just_a_bonus(catalog_with_features):
    # Every returned title must actually carry the chosen genre (floor), and the
    # result count is dynamic (at most 3).
    for g, label in [("crime", "Crime"), ("animation", "Animation"),
                     ("scifi_fantasy", "Sci-Fi & Fantasy")]:
        picks = rank_by_preferences(catalog_with_features, _ans(genre=g), top_n=3)
        assert 1 <= len(picks) <= 3
        assert all(label in genres for genres in picks["genres"])


def test_impossible_combo_relaxes_length_not_genre(catalog_with_features):
    # Drama + Long(6+) + New(2020+) + Big-hits is near-impossible (a 2020+ show
    # cannot have 6 seasons yet). The engine must keep GENRE (and ideally era +
    # popularity) and drop the brittle length rule, never returning kids animation.
    picks = rank_by_preferences(
        catalog_with_features,
        _ans(genre="drama", length="long", era="recent", popularity="well_known"),
        top_n=3,
    )
    assert 1 <= len(picks) <= 3
    assert all("Drama" in g for g in picks["genres"])      # genre floor held
    assert not any("Kids" in g for g in picks["genres"])   # no Trolls/Gabby's junk


def test_combined_genre_and_era(catalog_with_features):
    picks = rank_by_preferences(
        catalog_with_features, _ans(genre="drama", era="recent"), top_n=3
    )
    assert len(picks) == 3
    assert all("Drama" in g for g in picks["genres"])
    assert (picks["start_year"] >= 2020).all()
