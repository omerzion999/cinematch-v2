"""POST /api/chat - conversational turn handler (chat_turn + recommendations)."""

import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.agent.llm import chat_turn, explain_recommendations
from app.clustering.onboarding_map import build_user_vector, intent_to_onboarding_answers
from app.clustering.recommend import nearest_cluster, recommend_from_cluster
from app.engine.anomaly import is_anomalous
from app.engine.hybrid import recommend as hybrid_recommend
from app.i18n import t

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class RecCard(BaseModel):
    title: str
    genres: str
    rating: float
    overview: str
    poster_path: str | None = None
    decade_str: str
    num_seasons: float | None = None


class ChatRequest(BaseModel):
    conversation: list[ChatMessage]
    prev_recs: list[RecCard] | None = None
    lang: str = "he"


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecCard] | None = None
    explanation: str | None = None


def _nan_to_none(value):
    return None if pd.isna(value) else value


def _to_rec_cards(df: pd.DataFrame) -> list[RecCard]:
    return [
        RecCard(
            title=row["title"],
            genres=row["genres"],
            rating=float(row["rating"]),
            overview=row["overview"],
            poster_path=_nan_to_none(row.get("poster_path")),
            decade_str=row["decade_str"],
            num_seasons=_nan_to_none(row.get("num_seasons")),
        )
        for _, row in df.iterrows()
    ]


def _find_catalog_index(catalog: pd.DataFrame, title: str) -> int | None:
    matches = catalog.index[catalog["title"].str.lower() == title.lower()]
    if len(matches) == 0:
        return None
    return int(matches[0])


def _seed_based_picks(state, seed_title, lang, exclude_titles, top_n=3):
    catalog = state["catalog"]
    idx = _find_catalog_index(catalog, seed_title)
    if idx is None:
        return pd.DataFrame()

    df = hybrid_recommend(
        query_title=catalog.iloc[idx]["title"],
        catalog=catalog,
        numeric_matrix=state["numeric_matrix"],
        embeddings=state["embeddings"],
        query_embedding=state["embeddings"][idx],
        top_n=top_n,
        exclude_titles=exclude_titles,
        query_lang=lang,
    )
    if df.empty:
        return df
    if is_anomalous(float(df.iloc[0]["hybrid_score"]), state["anomaly_threshold"]):
        return pd.DataFrame()
    return df


def _cluster_based_picks(state, intent, exclude_titles, top_n=3):
    answers = intent_to_onboarding_answers(intent)
    vector, mask = build_user_vector(answers)
    cluster_id = nearest_cluster(
        vector, mask, state["cluster_centroids"], state["cluster_profiles"]
    )
    return recommend_from_cluster(
        state["catalog_with_features"], cluster_id, vector, mask,
        top_n=top_n, exclude_titles=list(exclude_titles),
    )


def _picks_for_intent(state, intent, lang, exclude_titles, top_n=3):
    seeds = intent.get("seeds") or []
    if seeds:
        picks = _seed_based_picks(state, seeds[0], lang, exclude_titles, top_n)
        if not picks.empty:
            return picks
    return _cluster_based_picks(state, intent, exclude_titles, top_n)


@router.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    state = request.app.state.cinematch
    conversation = [m.model_dump() for m in payload.conversation]
    prev_recs = [r.model_dump() for r in payload.prev_recs] if payload.prev_recs else None

    result = chat_turn(conversation, prev_recs=prev_recs, lang=payload.lang)
    action = result["action"]
    intent = result["intent"]

    if action == "chat":
        return ChatResponse(reply=result["reply"])

    exclude_titles = {r["title"] for r in (prev_recs or [])}

    if action == "swap_slot":
        slot_index = result["swap_slot_index"]
        new_picks = _picks_for_intent(state, intent, payload.lang, exclude_titles, top_n=1)
        cards = list(payload.prev_recs or [])
        if not new_picks.empty:
            cards[slot_index] = _to_rec_cards(new_picks)[0]
        return ChatResponse(reply=result["reply"], recommendations=cards)

    # action in ("search", "refine")
    picks = _picks_for_intent(state, intent, payload.lang, exclude_titles, top_n=3)
    if picks.empty:
        return ChatResponse(reply=result["reply"] or t("no_recommendations", payload.lang))

    cards = _to_rec_cards(picks)
    explanation = explain_recommendations(intent, picks.to_dict(orient="records"), payload.lang)
    return ChatResponse(reply=result["reply"], recommendations=cards, explanation=explanation)
