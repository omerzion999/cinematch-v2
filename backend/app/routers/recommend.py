"""POST /api/recommend - onboarding answers -> cluster + 3 picks + explanations."""

from typing import Literal

import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.agent.explanations import explain_picks
from app.agent.tmdb import search_tv_show
from app.clustering.onboarding_map import build_user_vector
from app.clustering.recommend import nearest_cluster, recommend_from_cluster
from app.i18n import t

router = APIRouter()


class OnboardingAnswers(BaseModel):
    genre: Literal[
        "drama", "comedy", "action_adventure", "scifi_fantasy",
        "crime", "animation", "romance", "any",
    ] = "any"
    length: Literal["short", "medium", "long", "any"] = "any"
    era: Literal["recent", "classic", "any"] = "any"
    tone: Literal["light_fun", "serious_drama", "thriller_action", "any"] = "any"
    popularity: Literal["well_known", "hidden_gem", "any"] = "any"


class RecommendRequest(BaseModel):
    answers: OnboardingAnswers
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


def _clean_poster_path(value) -> str | None:
    """Normalize a catalog/TMDB poster_path: NaN and empty strings become None."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _resolve_poster_path(pick: dict) -> str | None:
    """Use the catalog's poster_path if present, otherwise look it up on TMDB."""
    poster_path = _clean_poster_path(pick.get("poster_path"))
    if poster_path:
        return poster_path

    year = _nan_to_none(pick.get("start_year"))
    result = search_tv_show(pick["title"], year=int(year) if year else None)
    if result:
        return _clean_poster_path(result.get("poster_path"))
    return None


@router.post("/api/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest, request: Request) -> RecommendResponse:
    state = request.app.state.cinematch
    answers = payload.answers.model_dump()
    lang = payload.lang

    vector, mask = build_user_vector(answers)
    cluster_id = nearest_cluster(
        vector, mask, state["cluster_centroids"], state["cluster_profiles"]
    )
    cluster_profile = state["cluster_profiles"][str(cluster_id)]

    picks_df = recommend_from_cluster(
        state["catalog_with_features"], cluster_id, vector, mask, top_n=3
    )

    if picks_df.empty:
        return RecommendResponse(
            intro=t("no_recommendations", lang),
            outro="",
            cluster_id=cluster_id,
            recommendations=[],
        )

    picks = picks_df.to_dict(orient="records")
    explanations = explain_picks(answers, cluster_profile, picks, lang)

    recommendations = [
        ShowSummary(
            title=pick["title"],
            genres=pick["genres"],
            rating=0.0 if pd.isna(pick["rating"]) else float(pick["rating"]),
            overview=pick["overview"],
            poster_path=_resolve_poster_path(pick),
            decade_str=pick["decade_str"],
            num_seasons=_nan_to_none(pick.get("num_seasons")),
            binge_fit_score=float(pick["binge_fit_score"]),
            explanation=explanation,
        )
        for pick, explanation in zip(picks, explanations)
    ]

    label_key = "label_he" if lang == "he" else "label_en"
    intro = t("recommend_intro", lang, label=cluster_profile[label_key])
    outro = t("recommend_outro", lang)

    return RecommendResponse(
        intro=intro,
        outro=outro,
        cluster_id=cluster_id,
        recommendations=recommendations,
    )
