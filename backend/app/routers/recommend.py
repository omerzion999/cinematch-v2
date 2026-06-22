"""POST /api/recommend - onboarding answers -> cluster + 3 picks + explanations."""

from typing import Literal

import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.agent.explanations import explain_picks
from app.clustering.onboarding_map import build_user_vector
from app.clustering.recommend import nearest_cluster
from app.engine.hybrid import recommend_from_seeds
from app.engine.preference import rank_by_preferences
from app.i18n import t
from app.poster import resolve_poster_path

# Onboarding era/popularity answers -> hybrid apply_filters keys, used to refine
# the multi-seed similarity results when the user answered those refine questions.
_ERA_FILTER = {"recent": "recent", "modern": "2010s", "classic": "classic"}

router = APIRouter()

# Bilingual taste-profile labels, built from the user's own answers so the intro
# and per-pick explanations stay coherent with the weighted-ranking picks (the
# old collapsed-cluster label could say "Drama & Comedy" for Crime picks).
_GENRE_LABELS = {
    "drama": {"he": "דרמה", "en": "Drama"},
    "comedy": {"he": "קומדיה", "en": "Comedy"},
    "action_adventure": {"he": "אקשן והרפתקאות", "en": "Action & Adventure"},
    "scifi_fantasy": {"he": 'מד"ב ופנטזיה', "en": "Sci-Fi & Fantasy"},
    "crime": {"he": "פשע", "en": "Crime"},
    "animation": {"he": "אנימציה", "en": "Animation"},
}
_ERA_LABELS = {
    "recent": {"he": "חדש", "en": "recent"},
    "modern": {"he": "מודרני", "en": "modern"},
    "classic": {"he": "קלאסי", "en": "classic"},
}
_POPULARITY_LABELS = {
    "hidden_gem": {"he": "פנינים נסתרות", "en": "hidden gems"},
    "well_known": {"he": "להיטים מוכרים", "en": "well-known hits"},
}


def _taste_profile(answers: dict) -> dict:
    """A short bilingual taste-profile label derived from the answers."""
    genre = _GENRE_LABELS.get(answers.get("genre"))
    era = _ERA_LABELS.get(answers.get("era"))
    pop = _POPULARITY_LABELS.get(answers.get("popularity"))

    def _compose(lang: str) -> str:
        if genre and era:
            base = f"{genre[lang]} ({era[lang]})"
        elif genre:
            base = genre[lang]
        elif era:
            base = era[lang] if lang == "en" else f"סדרות {era[lang]}ות"
        elif pop:
            return pop[lang]
        else:
            return "מבחר מנצח" if lang == "he" else "a great mix"
        if pop:
            base += f", {pop[lang]}"
        return base

    return {"label_he": _compose("he"), "label_en": _compose("en")}


class OnboardingAnswers(BaseModel):
    genre: Literal[
        "drama", "comedy", "action_adventure", "scifi_fantasy",
        "crime", "animation", "any",
    ] = "any"
    era: Literal["recent", "modern", "classic", "any"] = "any"
    popularity: Literal["well_known", "hidden_gem", "any"] = "any"


class RecommendRequest(BaseModel):
    answers: OnboardingAnswers
    seeds: list[str] = []
    lang: Literal["he", "en"] = "he"


class ShowSummary(BaseModel):
    title: str
    genres: str
    rating: float
    overview: str
    poster_path: str | None = None
    decade_str: str
    num_seasons: float | None = None
    binge_fit_score: float
    explanation: str


class RecommendResponse(BaseModel):
    intro: str
    outro: str
    cluster_id: int
    recommendations: list[ShowSummary]


def _nan_to_none(value):
    return None if pd.isna(value) else value


@router.post("/api/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest, request: Request) -> RecommendResponse:
    state = request.app.state.cinematch
    answers = payload.answers.model_dump()
    lang = payload.lang

    # Weighted full-catalog ranking drives the picks (Option A). The K-Means
    # cluster is still computed for the cluster_id field (clustering stays wired
    # for the report / Option B), but no longer chooses the titles.
    vector, mask = build_user_vector(answers)
    cluster_id = nearest_cluster(
        vector, mask, state["cluster_centroids"], state["cluster_profiles"]
    )
    profile = _taste_profile(answers)

    if payload.seeds:
        # Seed-picker path: recommend titles similar to the shows the user picked,
        # refined by the era/popularity answers (length is no longer asked).
        filters: dict = {}
        if answers.get("era") in _ERA_FILTER:
            filters["era_pref"] = _ERA_FILTER[answers["era"]]
        if answers.get("popularity") in ("well_known", "hidden_gem"):
            filters["popularity_pref"] = answers["popularity"]
        picks_df = recommend_from_seeds(
            payload.seeds,
            state["catalog"],
            state["numeric_matrix"],
            state["embeddings"],
            top_n=3,
            filters=filters or None,
            query_lang=lang,
        )
    else:
        picks_df = rank_by_preferences(state["catalog_with_features"], answers, top_n=3)

    if picks_df.empty:
        return RecommendResponse(
            intro=t("no_recommendations", lang),
            outro="",
            cluster_id=cluster_id,
            recommendations=[],
        )

    picks = picks_df.to_dict(orient="records")
    explanations = explain_picks(answers, profile, picks, lang)

    recommendations = [
        ShowSummary(
            title=pick["title"],
            genres=pick["genres"],
            rating=0.0 if pd.isna(pick["rating"]) else float(pick["rating"]),
            overview=pick["overview"],
            poster_path=resolve_poster_path(pick),
            decade_str=pick["decade_str"],
            num_seasons=_nan_to_none(pick.get("num_seasons")),
            binge_fit_score=float(pick["binge_fit_score"]),
            explanation=explanation,
        )
        for pick, explanation in zip(picks, explanations)
    ]

    label_key = "label_he" if lang == "he" else "label_en"
    intro = t("recommend_intro", lang, label=profile[label_key])
    outro = t("recommend_outro", lang)

    return RecommendResponse(
        intro=intro,
        outro=outro,
        cluster_id=cluster_id,
        recommendations=recommendations,
    )
