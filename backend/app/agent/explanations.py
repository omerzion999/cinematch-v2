"""
Per-show "why we picked this" explanations for onboarding recommendations.

Unlike agent/llm.py's explain_recommendations() (which writes one paragraph
for a seed/mood-driven search with a free-text query), explain_picks()
writes one short sentence PER show, referencing the user's onboarding
answers and the matched cluster's taste-profile label.
"""

import json
import math
import re

from app.agent.llm import _call_llm, _get_client

_PICKS_EXPLAINER_SYSTEM = """\
You are a bilingual TV recommendation assistant. The user just answered a short
onboarding quiz about their taste, and the system matched them to a cluster of
shows and picked exactly N titles from that cluster.

Your job: write ONE short, warm sentence per show explaining why it fits the
user's stated taste profile. Reference the user's actual answers and/or the
cluster's theme where relevant.

Rules:
- Reply with a JSON array of exactly N strings, one per show, in the same
  order as the input list. No markdown fences, no other keys, no prose
  outside the array.
- Each string: ONE sentence, no em dashes (use commas/periods instead).
- When lang is "he": write entirely in simple modern Hebrew.
- When lang is "en": write entirely in English.
- Do not invent facts about the show beyond what's given (genres, rating, overview, decade).
"""


def explain_picks(answers: dict, cluster_profile: dict, picks: list[dict], lang: str = "en") -> list[str]:
    """
    Returns one short explanation per pick, in the same order as `picks`.
    Uses the LLM if available and it returns a same-length JSON array of
    strings; otherwise falls back to a deterministic genre/rating/cluster
    based sentence per pick.
    """
    if not picks:
        return []

    if not _get_client():
        return [_fallback_pick_explanation(p, cluster_profile, lang) for p in picks]

    label = cluster_profile.get("label_he" if lang == "he" else "label_en", "")
    picks_text = "\n".join(
        f"{i+1}. {p['title']} ({p.get('decade_str','')}, {p.get('genres','')}) "
        f"— Rating: {p.get('rating','')} — {(p.get('overview') or '')[:160]}"
        for i, p in enumerate(picks)
    )
    user_msg = (
        f"User's taste profile: {label}\n"
        f"Onboarding answers: {answers}\n"
        f"Language: {lang}\n\n"
        f"Shows (write exactly {len(picks)} explanations, in this order):\n{picks_text}"
    )

    raw = _call_llm(_PICKS_EXPLAINER_SYSTEM, user_msg, max_tokens=500)
    if raw:
        try:
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == len(picks) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass

    return [_fallback_pick_explanation(p, cluster_profile, lang) for p in picks]


def _fallback_pick_explanation(pick: dict, cluster_profile: dict, lang: str) -> str:
    genres = pick.get("genres", "")
    rating = pick.get("rating", "")
    if isinstance(rating, float) and math.isnan(rating):
        rating = 0.0
    rating_str = f"{rating:.1f}" if isinstance(rating, float) else str(rating)
    label = cluster_profile.get("label_he" if lang == "he" else "label_en", "")

    if lang == "he":
        if label:
            return f"מתאים לטעם שלך ({label}): {genres}, דירוג {rating_str}."
        return f"{genres}, דירוג {rating_str}."

    if label:
        return f"Matches your taste profile ({label}): {genres}, rated {rating_str}."
    return f"{genres}, rated {rating_str}."
