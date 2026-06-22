"""POST /api/chat - conversational turn handler (chat_turn + recommendations)."""

import re

import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.agent.llm import chat_turn
from app.catalog_lookup import find_catalog_index
from app.clustering.onboarding_map import intent_to_onboarding_answers
from app.engine.anomaly import is_anomalous
from app.engine.hybrid import apply_filters, recommend as hybrid_recommend
from app.engine.preference import rank_by_preferences
from app.i18n import t
from app.poster import resolve_poster_path

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
            rating=0.0 if pd.isna(row["rating"]) else float(row["rating"]),
            overview=row["overview"],
            # Resolve via TMDB when the catalog has no poster, so chat cards show
            # artwork too (the catalog poster_path is missing for about 25%).
            poster_path=resolve_poster_path(row.to_dict()),
            decade_str=row["decade_str"],
            num_seasons=_nan_to_none(row.get("num_seasons")),
        )
        for _, row in df.iterrows()
    ]


def _seed_based_picks(state, seed_title, lang, exclude_titles, top_n=3, filters=None):
    catalog = state["catalog"]
    idx = find_catalog_index(catalog, seed_title)
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
        filters=filters,
    )
    if df.empty:
        return df
    if is_anomalous(float(df.iloc[0]["hybrid_score"]), state["anomaly_threshold"]):
        return pd.DataFrame()
    return df


_ANY_ANSWERS = {"genre": "any", "length": "any", "era": "any", "popularity": "any"}


def _keyword_picks(state, keywords, preferred_genres, exclude_titles, top_n=3):
    """Bridge search: catalog shows about the topic, ranked by the same quality
    blend as the preference ranker. Two stages keep results on-topic:
      1. keep titles whose overview/title mention a topic keyword;
      2. of those, keep only ones whose GENRE is a topic genre OR whose TITLE
         contains a keyword. Step 2 drops shows that merely mention a keyword in
         passing (e.g. Daredevil's "Hell's Kitchen"). Falls back to step 1 if
         step 2 is empty, so the bridge still offers something."""
    catalog = state["catalog_with_features"]
    # Word-boundary match so topic keywords do not hit substrings of unrelated
    # words ("world" inside "Westworld", "rock" inside "Rocket").
    kw_pattern = "|".join(rf"\b{re.escape(k)}\b" for k in keywords if k)
    if not kw_pattern:
        return catalog.head(0)

    title = catalog["title"].fillna("")
    text = catalog["overview"].fillna("") + " " + title
    pool = catalog[text.str.contains(kw_pattern, case=False, na=False, regex=True)]
    if pool.empty:
        return pool

    genres = pool["genres"].fillna("")
    genre_pattern = "|".join(re.escape(g) for g in (preferred_genres or []))
    on_genre = genres.str.contains(genre_pattern, case=False, na=False) if genre_pattern else False
    in_title = pool["title"].fillna("").str.contains(kw_pattern, case=False, na=False, regex=True)
    relevant = pool[on_genre | in_title]
    if not relevant.empty:
        pool = relevant

    return rank_by_preferences(pool, _ANY_ANSWERS, top_n=top_n, exclude_titles=exclude_titles)


def _preference_picks(state, intent, exclude_titles, top_n=3):
    """Seedless personalization: rank the whole catalog by the user's intent
    (mood, length, era, popularity), the same weighted ranker used by onboarding."""
    answers = intent_to_onboarding_answers(intent)
    return rank_by_preferences(
        state["catalog_with_features"],
        answers,
        top_n=top_n,
        exclude_titles=exclude_titles,
        exclude_genres=intent.get("exclude_genres") or None,
    )


_MOOD_TO_GENRE: dict[str, str] = {
    "thrilling": "Action",
    "funny": "Comedy",
    "emotional": "Drama",
}


def _language_filtered_picks(state, language: str, exclude_titles, top_n=3, intent=None) -> pd.DataFrame:
    catalog = state["catalog"]
    df = catalog[catalog["language"] == language]
    if exclude_titles:
        df = df[~df["title"].isin(exclude_titles)]
    if intent:
        df = apply_filters(df, intent)
    return df.sort_values("rating", ascending=False, na_position="last").head(top_n)


def _picks_for_intent(state, intent, lang, exclude_titles, top_n=3, prev_recs=None):
    # Derive genre filter from mood so "action" → filters Action, "funny" → Comedy, etc.
    mood_genres = [_MOOD_TO_GENRE[m] for m in (intent.get("mood") or []) if m in _MOOD_TO_GENRE]
    if mood_genres and not intent.get("include_genres"):
        intent = {**intent, "include_genres": mood_genres}

    keywords = intent.get("keywords") or []
    if keywords:
        picks = _keyword_picks(
            state, keywords, intent.get("bridge_genres") or [], exclude_titles, top_n
        )
        if not picks.empty:
            return picks

    if intent.get("language_pref") == "he":
        return _language_filtered_picks(state, "he", exclude_titles, top_n, intent)
    seeds = intent.get("seeds") or []
    if not seeds and prev_recs:
        seeds = [prev_recs[0]["title"]]
    if seeds:
        picks = _seed_based_picks(state, seeds[0], lang, exclude_titles, top_n, filters=intent)
        if not picks.empty:
            return picks
    return _preference_picks(state, intent, exclude_titles, top_n)


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
        cards = list(payload.prev_recs or [])
        swap_seed = [{"title": cards[slot_index].title}] if cards else None
        new_picks = _picks_for_intent(
            state, intent, payload.lang, exclude_titles, top_n=1, prev_recs=swap_seed
        )
        if new_picks.empty:
            return ChatResponse(reply=t("not_in_catalog", payload.lang), recommendations=cards)
        cards[slot_index] = _to_rec_cards(new_picks)[0]
        return ChatResponse(reply=result["reply"], recommendations=cards)

    # action in ("search", "refine")
    picks = _picks_for_intent(state, intent, payload.lang, exclude_titles, top_n=3, prev_recs=prev_recs)
    if picks.empty:
        return ChatResponse(reply=t("not_in_catalog", payload.lang))

    # Keep answers short: the one-line reply plus the cards are enough. We do not
    # attach a long explanation bubble (it tended to ramble and even critique its
    # own picks). Per-show context lives in the card modal.
    cards = _to_rec_cards(picks)
    return ChatResponse(reply=result["reply"], recommendations=cards)
