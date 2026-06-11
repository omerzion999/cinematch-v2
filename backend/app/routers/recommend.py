"""POST /api/recommend - onboarding answers -> cluster + 3 picks + explanations."""

import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.agent.explanations import explain_picks
from app.clustering.onboarding_map import build_user_vector
from app.clustering.recommend import nearest_cluster, recommend_from_cluster
from app.i18n import t

router = APIRouter()


class OnboardingAnswers(BaseModel):
    genre: str = "any"
    length: str = "any"
    era: str = "any"
    tone: str = "any"
    popularity: str = "any"


class RecommendRequest(BaseModel):
    answers: OnboardingAnswers
    lang: str = "he"


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
            rating=float(pick["rating"]),
            overview=pick["overview"],
            poster_path=_nan_to_none(pick.get("poster_path")),
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
