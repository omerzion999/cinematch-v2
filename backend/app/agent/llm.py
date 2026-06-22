"""
CineMatch AI — LLM Agent
Priority: GROQ_API_KEY → ANTHROPIC_API_KEY → offline regex fallback.
"""
from __future__ import annotations

import json, math, os, re
from typing import Optional

from app.i18n import t

# ── Client state ───────────────────────────────────────────────────────────────

_groq_client = None
_anthropic_client = None
_provider = None  # "groq" | "anthropic" | None


def _read_secret(key: str) -> Optional[str]:
    return os.environ.get(key)


def _get_client():
    global _groq_client, _anthropic_client, _provider
    if _provider is not None:
        return True

    # Primary: Groq
    groq_key = _read_secret("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=groq_key)
            _provider = "groq"
            return True
        except Exception:
            pass

    # Fallback: Anthropic
    anthropic_key = _read_secret("ANTHROPIC_API_KEY") or _read_secret("ANTHROPIC_KEY")
    if anthropic_key:
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            _provider = "anthropic"
            return True
        except Exception:
            pass

    _provider = None
    return None


# ── System prompts ─────────────────────────────────────────────────────────────

_PARSER_SYSTEM = """\
You are an assistant that extracts structured intent from TV/movie recommendation queries.
The user may write in Hebrew or English. Always reply with valid JSON only.

Output schema:
{
  "seeds": ["Title 1"],
  "mood": ["dark", "funny"],
  "length_pref": "short|long|limited|any",
  "language_pref": "en|ko|es|de|fr|ja|foreign|any",
  "era_pref": "classic|1990s|2000s|2010s|2020s|recent|any",
  "year_min": null,
  "year_max": null,
  "status": "airing|finished|any",
  "popularity_pref": "trending|hidden_gem|well_known|any",
  "binge_pref": "binge|casual|any",
  "rating_min": null,
  "exclude_genres": [],
  "lang": "he|en",
  "free_text": "..."
}

CRITICAL RULE: BE CONSERVATIVE
Every filter you set narrows the engine's results. Only set a field if the user EXPLICITLY mentioned that preference. When in doubt, leave the field at "any" or null. Over-filtering kills the user's recommendations and surfaces bad shows.

Rules:
- seeds: ONLY titles the user explicitly mentions. Do NOT add titles from your own knowledge.
- mood: ONLY if user used an explicit mood word ("dark"→dark, "מצחיק"→funny, "מרגש"→emotional, "אפל"→dark). For neutral queries like "shows like Breaking Bad", leave mood empty.
- length_pref: ONLY if user said short/קצר/mini/limited/long/ארוך/epic/"many seasons". Default "any".
- language_pref: ONLY if user said a specific language or "foreign". Default "any". Do NOT set "en" just because the query is in English.
- era_pref: ONLY if user mentioned an era (old/classic/recent/specific decade). Default "any". A query like "shows like Breaking Bad" does NOT imply any era.
- year_min: ONLY if user said "from YEAR", "after YEAR", "YEAR and later", "משנת YEAR", "אחרי YEAR"
- year_max: ONLY if user said "before YEAR", "until YEAR", "לפני YEAR", "עד YEAR"
  Example: "from 2020" → year_min=2020, era_pref="recent"
  Example: "before 2000" → year_max=1999, era_pref="classic"
- status: ONLY if user said "still airing/ongoing/עדיין משודר" or "finished/completed/ended/הסתיים". Default "any". Do NOT auto-set "finished" just because they mentioned an old show.
- popularity_pref: ONLY if user said "hidden gem/underrated/obscure" or "trending/popular/hot" or "well-known/famous". Default "any". Do NOT auto-set "trending" just because they named a popular show.
- binge_pref: ONLY if user said "binge/weekend/סוף שבוע" or "casual/light/episodic". Default "any". Do NOT auto-set "binge" for serialized dramas.
- rating_min: ONLY if user said "highly rated/best" (then 8.5) or "decent/סביר" (then 7.0). Default null. Do NOT auto-set 7.0.
- exclude_genres: ONLY if user said "not X" or "without Y".
- lang: "he" if query contains Hebrew characters, else "en".
- Use null/any for undetermined fields. Repeat: do NOT guess or infer filters that the user did not explicitly state.
"""

_EXPLAINER_SYSTEM = """\
You are a bilingual TV/movie recommendation assistant.
Your ONLY job is to explain WHY the shows in the provided list match the user's query.
CRITICAL RULES:
- Do NOT suggest, mention, or reference any show that is not in the provided recommendations list.
- Do NOT add shows from your own knowledge. Only work with the exact list given to you.
- Do NOT say "you might also like X" or recommend anything beyond the list.
- NEVER try to explain why a show satisfies a filter it does not meet. The filtering has already
  been applied by the engine — trust the list. Do not say things like "although this is from 2010,
  it still qualifies because..." — simply explain why each show matches the mood/genre/style intent.
- When lang is "he": reply ENTIRELY in simple, clear, modern Hebrew (Israeli everyday language).
  Use short sentences. Avoid overly formal or archaic phrasing.
- When lang is "en": reply entirely in English.
- Be concise: one warm opening sentence, then one short sentence per show explaining why it fits.
"""


# ── LLM call helper ────────────────────────────────────────────────────────────

_GROQ_MODEL_CHAT = "llama-3.1-8b-instant"      # conversational handler (rate-limit-friendly)
_GROQ_MODEL_FAST = "llama-3.1-8b-instant"      # parser, classifier, explainer (cheap, fast)
# NOTE: tried llama-3.3-70b-versatile for richer persona (in PR #12) but it
# was still failing on Groq free tier even with the new 3x retry loop. The
# 70b model has tight token-per-minute limits that make it unreliable for
# interactive demos. Reverted to 8b-instant for guaranteed responsiveness.
# Persona is flatter but the agent actually replies.


def _call_llm(
    system: str, user: str, max_tokens: int = 600, *,
    model: Optional[str] = None, temperature: Optional[float] = None,
) -> Optional[str]:
    # Groq path: retry up to 3 times with a 5-second per-attempt timeout.
    # Total worst-case still ~15 seconds, but transient slowness on the first
    # attempt now gets two more chances instead of immediately falling back.
    # On non-timeout errors (auth, malformed request), we break early since
    # those won't fix themselves with retries.
    if _provider == "groq":
        for _attempt in range(3):
            try:
                kwargs = dict(
                    model=model or _GROQ_MODEL_FAST,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    timeout=5,
                )
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = _groq_client.chat.completions.create(**kwargs)
                return response.choices[0].message.content.strip()
            except Exception as _e:
                # Heuristic: retry on timeout-shaped errors, break on others.
                _err = (str(type(_e).__name__) + " " + str(_e)).lower()
                if any(s in _err for s in ("timeout", "timed out", "503", "504", "429")):
                    continue
                return None
        return None

    if _provider == "anthropic":
        try:
            kwargs = dict(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                timeout=15,
            )
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = _anthropic_client.messages.create(**kwargs)
            return response.content[0].text.strip()
        except Exception:
            return None

    return None


# ── Intent parser ─────────────────────────────────────────────────────────────

def parse_intent(query: str) -> dict:
    if not _get_client():
        return _regex_parse(query)

    raw = _call_llm(_PARSER_SYSTEM, f"Query: {query}", max_tokens=512)
    if raw:
        try:
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            parsed = json.loads(raw)
            parsed.setdefault("free_text", query)
            parsed.setdefault("language_pref", "any")
            parsed.setdefault("era_pref", "any")
            parsed.setdefault("year_min", None)
            parsed.setdefault("year_max", None)
            parsed.setdefault("status", "any")
            parsed.setdefault("popularity_pref", "any")
            parsed.setdefault("binge_pref", "any")
            parsed.setdefault("rating_min", None)
            return parsed
        except Exception:
            pass

    return _regex_parse(query)


def _regex_parse(query: str) -> dict:
    q = query.strip()
    ql = q.lower()
    has_hebrew = bool(re.search(r"[֐-׿]", q))
    lang = "he" if has_hebrew else "en"

    mood_map = {
        "dark":      ["dark","אפל","כהה","depressing"],
        "funny":     ["funny","comedy","fun","מצחיק","הומור","קומדי","sitcom"],
        "emotional": ["emotional","sad","cry","מרגש","עצוב"],
        "thrilling": ["thriller","thrill","suspense","מותחן","מפחיד","horror",
                      "scary","action","אקשן","אימה","crime","פשע","מתח"],
        "light":     ["light","lighthearted","קליל","קל","cheerful"],
    }
    mood = []
    for tag, kws in mood_map.items():
        for kw in kws:
            if kw.lower() in ql:
                mood.append(tag)
                break

    # length_pref
    if re.search(r"\bone season\b|עונה אחת", q, re.IGNORECASE):
        length_pref = "limited"
    elif re.search(
        r"\b(short|shorter|קצר|קצרות|קצרה|קצרים|mini|limited|פחות פרקים|fewer episodes"
        r"|not too many seasons|not many seasons|fewer seasons"
        r"|לא הרבה עונות|לא יותר מדי עונות|לא הרבה פרקים)\b",
        q, re.IGNORECASE,
    ):
        length_pref = "short"
    elif re.search(r"\b(long|longer|ארוך|epic|many seasons|הרבה עונות)\b", q, re.IGNORECASE):
        length_pref = "long"
    else:
        length_pref = "any"

    # language_pref
    if re.search(r"\b(foreign|זר|זרה|not english|לא אנגלית|בשפה זרה)\b", q, re.IGNORECASE):
        language_pref = "foreign"
    elif re.search(r"\b(english|אנגלית)\b", q, re.IGNORECASE):
        language_pref = "en"
    elif re.search(r"\b(korean|קוריאני|קוריאנית)\b", q, re.IGNORECASE):
        language_pref = "ko"
    elif re.search(r"\b(spanish|ספרדית|ספרדי)\b", q, re.IGNORECASE):
        language_pref = "es"
    elif re.search(r"\b(german|גרמנית|גרמני)\b", q, re.IGNORECASE):
        language_pref = "de"
    elif re.search(r"\b(french|צרפתית|צרפתי)\b", q, re.IGNORECASE):
        language_pref = "fr"
    elif re.search(r"\b(japanese|יפנית|יפני|anime)\b", q, re.IGNORECASE):
        language_pref = "ja"
    else:
        language_pref = "any"

    # year_min / year_max (explicit year boundaries)
    year_min = None
    year_max = None
    _from_m = re.search(
        r"(?:from|after|since|משנת|אחרי|מ-?|החל מ)\s*((?:19|20)\d{2})"
        r"|(?:year\s+)?((?:19|20)\d{2})\s+(?:and\s+(?:more|later|up|above|onwards?)"
        r"|ומעלה|ואילך|ומאוחר\s+יותר)",
        q, re.IGNORECASE)
    if _from_m:
        yr = int(next(g for g in _from_m.groups() if g))
        year_min = yr
    _to_m = re.search(
        r"(?:before|until|up\s+to|לפני|עד)\s*((?:19|20)\d{2})"
        r"|(?:year\s+)?((?:19|20)\d{2})\s+(?:and\s+(?:earlier|before)|ומוקדם\s+יותר)",
        q, re.IGNORECASE)
    if _to_m:
        if _to_m.group(1):
            year_max = int(_to_m.group(1)) - 1  # "before 2000" → max 1999
        else:
            year_max = int(_to_m.group(2))  # "2010 and earlier" → max 2010 (inclusive)

    # era_pref (derived from year_min or explicit decade keywords)
    if year_min and year_min >= 2020:
        era_pref = "recent"
    elif year_min and year_min >= 2010:
        era_pref = "2010s"
    elif year_min and year_min >= 2000:
        era_pref = "2000s"
    elif year_max and year_max < 2000:
        era_pref = "classic"
    elif re.search(r"\b(classic|קלאסי|old|ישן|pre.?2000)\b", q, re.IGNORECASE):
        era_pref = "classic"
    elif re.search(r"\b(90s|שנות ה.?90|1990s)\b", q, re.IGNORECASE):
        era_pref = "1990s"
    elif re.search(r"\b(2000s|שנות ה.?2000)\b", q, re.IGNORECASE):
        era_pref = "2000s"
    elif re.search(r"\b(2010s|שנות ה.?2010)\b", q, re.IGNORECASE):
        era_pref = "2010s"
    elif re.search(r"\b(recent|חדש|new|2020s|שנות ה.?2020)\b", q, re.IGNORECASE):
        era_pref = "recent"
    else:
        era_pref = "any"

    # status
    if re.search(r"\b(still airing|ongoing|עדיין משודר|ממשיך)\b", q, re.IGNORECASE):
        status = "airing"
    elif re.search(r"\b(finished|completed|ended|הסתיימה|הסתיים)\b", q, re.IGNORECASE):
        status = "finished"
    else:
        status = "any"

    # popularity_pref
    if re.search(r"\b(hidden gem|underrated|אוצר נסתר|לא מוכר)\b", q, re.IGNORECASE):
        popularity_pref = "hidden_gem"
    elif re.search(r"\b(trending|popular|פופולרי|hot)\b", q, re.IGNORECASE):
        popularity_pref = "trending"
    elif re.search(r"\b(well.?known|famous|מפורסם)\b", q, re.IGNORECASE):
        popularity_pref = "well_known"
    else:
        popularity_pref = "any"

    # binge_pref
    if re.search(r"\b(binge|weekend|סוף שבוע|בינג)\b", q, re.IGNORECASE):
        binge_pref = "binge"
    elif re.search(r"\b(casual|קליל|episodic)\b", q, re.IGNORECASE):
        binge_pref = "casual"
    else:
        binge_pref = "any"

    # rating_min
    if re.search(r"\b(highly rated|best rated|top rated|מדורג גבוה)\b", q, re.IGNORECASE):
        rating_min = 8.5
    elif re.search(r"\b(decent|סביר|acceptable)\b", q, re.IGNORECASE):
        rating_min = 7.0
    else:
        rating_min = None

    # ── Seed extraction ("something like X", "shows like X", "כמו X") ─────────
    seeds: list[str] = []
    _seed_m = re.search(
        r"\b(?:something|shows?|series)\s+like\s+[\"']?([A-Za-z][A-Za-z0-9 :&'.-]{1,40}?)[\"']?(?:\s*[,!?.]|$)"
        r"|\blike\s+[\"']([A-Za-z][A-Za-z0-9 :&'.-]{1,40}?)[\"']"
        r"|\bsimilar\s+to\s+[\"']?([A-Za-z][A-Za-z0-9 :&'.-]{1,40}?)[\"']?(?:\s*[,!?.]|$)"
        r"|כמו\s+[\"']?([^\s,!?.]{2,40})"
        r"|דומה\s+ל-?[\"']?([^\s,!?.]{2,40})",
        q, re.IGNORECASE,
    )
    if _seed_m:
        seed_title = next((g for g in _seed_m.groups() if g), "").strip().strip("\"'")
        if seed_title:
            seeds = [seed_title]

    return {
        "seeds": seeds, "mood": mood, "length_pref": length_pref,
        "language_pref": language_pref, "era_pref": era_pref,
        "year_min": year_min, "year_max": year_max,
        "status": status, "popularity_pref": popularity_pref,
        "binge_pref": binge_pref, "rating_min": rating_min,
        "exclude_genres": [], "lang": lang, "free_text": query,
    }


# ── Explanation generator ──────────────────────────────────────────────────────

def explain_recommendations(intent: dict, recommendations: list[dict], lang: str = "en") -> str:
    if not recommendations:
        return "לא נמצאו תוצאות מתאימות." if lang == "he" else "No matching results found."

    # Both languages go through the LLM (the explainer system prompt enforces
    # natural casual Hebrew). The deterministic template stays as the fallback
    # for when the LLM is unavailable or returns nothing.
    if not _get_client():
        return _fallback_explanation(intent, recommendations, lang)

    recs_text = "\n".join(
        f"{i+1}. {r['title']} ({r.get('decade_str','')}, {r.get('genres','')}) "
        f"— Rating: {r.get('rating','')} — Hybrid score: {r.get('hybrid_score','')}"
        for i, r in enumerate(recommendations)
    )

    user_msg = (
        f"User query: {intent.get('free_text','')}\n"
        f"Detected mood: {intent.get('mood',[])}, Language: {lang}\n\n"
        f"The recommendation engine found EXACTLY these {len(recommendations)} shows from our database:\n"
        f"{recs_text}\n\n"
        f"Explain ONLY these shows and why they match the query. "
        f"Do not mention any other shows. Reply in {'Hebrew' if lang=='he' else 'English'}."
    )

    result = _call_llm(_EXPLAINER_SYSTEM, user_msg, max_tokens=600)
    return result if result else _fallback_explanation(intent, recommendations, lang)


# ── Follow-up pattern detection (fast, no LLM needed) ─────────────────────────

_OTHER_OPTIONS_TOKENS = [
    "other", "different", "something else", "more options", "new ones",
    "give me more", "show me more", "more results", "again", "next",
    "אחר", "אחרות", "שונה", "עוד", "אופציות אחרות", "משהו אחר",
    "נוספות", "תוצאות נוספות", "עוד תוצאות", "תן לי עוד",
]
_SHORTER_TOKENS = [
    "short", "shorter", "fewer episodes", "mini series", "mini-series",
    "קצר", "קצרות", "פחות פרקים", "קצרה", "קצרים",
]
_LIGHTER_TOKENS = [
    "less dark", "lighter", "not so dark", "not dark", "less serious",
    "less heavy", "more fun", "more light",
    "פחות אפל", "לא אפל", "קליל", "מצחיק יותר", "פחות כבד", "יותר קליל",
]
_FOREIGN_TOKENS = [
    "foreign", "non-english", "not in english", "foreign language",
    "non english", "subtitles",
    "זר", "זרה", "בשפה זרה", "לא באנגלית", "שפה אחרת", "כתוביות",
]


def _detect_followup_type(msg: str) -> Optional[str]:
    """Detect common follow-up patterns without calling the LLM."""
    m = msg.lower().strip()
    if any(tok in m for tok in _OTHER_OPTIONS_TOKENS):
        return "other"
    if any(tok in m for tok in _SHORTER_TOKENS):
        return "shorter"
    if any(tok in m for tok in _LIGHTER_TOKENS):
        return "lighter"
    if any(tok in m for tok in _FOREIGN_TOKENS):
        return "foreign"
    return None


_VOWELS = set("aeiouAEIOU")


def _is_gibberish(msg: str) -> bool:
    """Very conservative, low-threshold detector for keyboard-mashing input.

    Only flags a single Latin-alphabet "word" (no spaces) of 6+ letters that
    contains no vowels at all, e.g. "sjfnkfdsngkjdhf". Real words, acronyms,
    short input, numbers, punctuation, and Hebrew text are never flagged, so
    this never blocks a genuine (if terse) request.
    """
    text = msg.strip()
    if len(text) < 6 or " " in text or "\n" in text:
        return False
    if not re.fullmatch(r"[A-Za-z]+", text):
        return False
    return not any(c in _VOWELS for c in text)


# ── Off-topic bridge map ──────────────────────────────────────────────────────
# Maps a real-world topic the user might mention (in Hebrew or English) to TV
# catalog search keywords. When a user asks for something off-topic but
# bridgeable ("I want to make pasta"), we search the catalog for shows ABOUT that
# topic (matched on title/overview) instead of hard-declining. Keywords are
# matched against the English overview/title, so they are English stems.
# Each entry: trigger_tokens -> (topic_label, search_keywords, preferred_genres).
#
# Search keywords are curated to be topic-specific (matched on word boundaries
# against the English overview/title), so generic words that cross topics
# ("world", "team", "band", "rock") are deliberately excluded. preferred_genres
# is the relevance filter the chat router applies on top of the keyword match: a
# candidate is kept only if its genre is one of these OR a keyword appears in its
# title. That drops off-topic shows that merely mention a keyword in passing
# (e.g. Daredevil's "Hell's Kitchen").
_BRIDGE_MAP: list[tuple[list[str], tuple[str, list[str], list[str]]]] = [
    (["cook", "cooking", "recipe", "pasta", "bake", "baking", "chef", "kitchen",
      "food", "מבשל", "לבשל", "בישול", "מתכון", "אוכל", "אפיה", "מאפה"],
     ("cooking", ["chef", "cook", "cooking", "kitchen", "culinary", "recipe",
                  "restaurant", "cuisine", "baking", "food"],
      ["Documentary", "Reality", "Family"])),
    (["sport", "sports", "football", "soccer", "basketball", "workout", "gym",
      "ספורט", "כדורגל", "כדורסל", "אימון", "כושר"],
     ("sports", ["football", "soccer", "basketball", "baseball", "athlete",
                 "olympic", "championship"],
      ["Documentary", "Reality", "Sport"])),
    (["travel", "trip", "vacation", "flight", "tourism",
      "טיול", "לטייל", "חופשה", "נסיעה", "תיירות"],
     ("travel", ["travel", "traveler", "tourist", "expedition", "backpacking"],
      ["Documentary", "Reality"])),
    (["music", "song", "songs", "concert", "guitar", "band",
      "מוזיקה", "מוסיקה", "שיר", "שירים", "הופעה", "גיטרה", "להקה"],
     ("music", ["music", "musician", "singer", "songwriter", "rapper",
                "orchestra", "jazz"],
      ["Music", "Documentary"])),
    (["space", "science", "physics", "universe", "astronomy",
      "חלל", "מדע", "פיזיקה", "יקום", "אסטרונומיה"],
     ("science", ["space", "science", "scientist", "physics", "cosmos",
                  "universe", "astronaut"],
      ["Documentary"])),
    (["history", "historical", "war", "ancient",
      "היסטוריה", "היסטורי", "מלחמה", "עתיק"],
     ("history", ["history", "historical", "ancient", "empire", "medieval",
                  "dynasty"],
      ["Documentary", "History", "War"])),
    (["nature", "animal", "animals", "wildlife", "ocean",
      "טבע", "חיות", "בעלי חיים", "אוקיינוס"],
     ("nature", ["nature", "wildlife", "animal", "ocean", "jungle", "safari",
                 "species"],
      ["Documentary", "Family"])),
    (["car", "cars", "racing", "motor", "drive",
      "מכונית", "מכוניות", "רכב", "מירוץ", "נהיגה"],
     ("cars", ["racing", "motorsport", "automobile", "supercar"],
      ["Documentary", "Reality"])),
    (["fashion", "style", "model", "design",
      "אופנה", "סטייל", "דוגמנית", "עיצוב"],
     ("fashion", ["fashion", "runway", "couture", "designer", "modeling"],
      ["Documentary", "Reality"])),
]

_BRIDGE_LABELS_HE = {
    "cooking": "בישול", "sports": "ספורט", "travel": "טיולים", "music": "מוזיקה",
    "science": "מדע", "history": "היסטוריה", "nature": "טבע", "cars": "רכב",
    "fashion": "אופנה",
}


def _detect_bridge(msg: str) -> Optional[tuple[str, list[str], list[str]]]:
    """If the message mentions a bridgeable real-world topic, return
    (topic_label, search_keywords, preferred_genres); otherwise None."""
    m = msg.lower()
    for tokens, payload in _BRIDGE_MAP:
        for tok in tokens:
            # word-ish boundary for short Latin tokens to avoid false hits
            if tok.isascii() and tok.isalpha() and len(tok) <= 4:
                if re.search(rf"\b{re.escape(tok)}\b", m):
                    return payload
            elif tok in m:
                return payload
    return None


def _bridge_reply(topic: str, lang: str) -> str:
    if lang == "he":
        label = _BRIDGE_LABELS_HE.get(topic, topic)
        return f"אני על סדרות, אבל הנה כמה סדרות בנושא {label} שתוכל לראות:"
    return f"I'm all about TV, but here are some great {topic} series to watch:"


# ── Additional keyword patterns ───────────────────────────────────────────────

_QUESTION_TOKENS = [
    "what is", "what's", "tell me about", "tell me more", "how many",
    "when was", "who made", "who stars", "plot", "seasons", "episodes",
    "about the", "explain", "describe", "overview",
    "על מה", "כמה עונות", "כמה פרקים", "מתי", "מי עשה", "ספר לי על",
    "מה העלילה", "בכמה עונות", "מה זה",
]

_CHAT_TOKENS = [
    "thanks", "thank you", "great", "ok", "okay", "cool", "nice",
    "perfect", "awesome", "love it", "got it", "sounds good", "sure",
    "no thanks", "nevermind", "never mind", "that's all", "bye",
    "תודה", "מעולה", "נהדר", "אחלה", "סבבה", "טוב", "הבנתי",
    "ממש טוב", "יופי", "בסדר", "תענוג",
]


def _keyword_classify(message: str) -> str:
    """Keyword-based fallback classifier — used when LLM is unavailable."""
    m = message.lower().strip()

    if any(tok in m for tok in _OTHER_OPTIONS_TOKENS):
        return "more_options"

    if any(tok in m for tok in _SHORTER_TOKENS + _LIGHTER_TOKENS + _FOREIGN_TOKENS):
        return "refine"

    if any(tok in m for tok in _QUESTION_TOKENS):
        return "question"

    if any(tok in m for tok in _CHAT_TOKENS):
        return "chat"

    return "search"


_CLASSIFIER_SYSTEM = """\
Classify the user's latest message into exactly one of these intents:
- search: wants to find a show/movie (new query, e.g. "something like Breaking Bad", "dark thriller")
- more_options: wants different results from the same search ("more", "other options", "something else", "אחרות", "עוד")
- refine: wants to adjust current results ("shorter", "less dark", "foreign", "קצר", "פחות אפל")
- question: asks about a specific show ("what is it about", "how many seasons", "על מה זה")
- chat: acknowledgement or general chat ("thanks", "great", "ok", "תודה")

Reply with ONLY the single intent word. Nothing else. No punctuation.
"""


def classify_intent(message: str, conversation_history: list[dict] | None = None) -> str:
    """
    Classify a user message into: search | more_options | refine | question | chat

    Uses LLM (fast, max 10 tokens) with last-3-message context.
    Falls back to keyword matching if LLM unavailable or returns unexpected output.
    """
    _get_client()

    if _provider is not None:
        ctx_msgs = (conversation_history or [])[-3:]
        ctx = "\n".join(
            f"{m['role'].upper()}: {m['content'][:120]}" for m in ctx_msgs
        )
        user_prompt = (
            f"Context:\n{ctx}\n\nLatest message: {message}\n\nIntent:"
            if ctx else
            f"Message: {message}\n\nIntent:"
        )
        raw = _call_llm(_CLASSIFIER_SYSTEM, user_prompt, max_tokens=10)
        if raw:
            word = raw.strip().lower().split()[0].rstrip(".,!") if raw.strip() else ""
            if word in ("search", "more_options", "refine", "question", "chat"):
                return word

    return _keyword_classify(message)


# ── Conversational chat turn ───────────────────────────────────────────────────

_CHAT_SYSTEM = """\
You are CineMatch AI, a warm, witty bilingual assistant whose specialty is TV and movie recommendations from a curated catalog of 11,013 titles. You can also chat about anything else, like ChatGPT or Claude would.

PERSONALITY
- You are CineMatch AI, a TV and movie recommendation specialist. That is your one job.
- Friendly, confident, a real person. Never robotic. Never preachy. Never sycophantic.
- Reply in the user's language. If they wrote in Hebrew, reply in Hebrew. If they wrote in English, reply in English.
- TV and movie related chat is ALL welcome: opinions on shows, character discussion, plot trivia, actor questions, availability questions, jokes ABOUT shows or movies, recommendations.
- For OFF-TOPIC requests (recipes, weather, math, code, news, sports scores, anything not TV/movie): politely acknowledge you cannot help with that, mention your purpose in a natural one-liner, then offer to find something to watch instead. NEVER offer to actually do the off-topic thing. Do not pretend you might be able to help.
- One light scope-mention per off-topic redirect is plenty. Do not be preachy or repeat the disclaimer.
- Keep replies short and conversational, 1 to 3 sentences usually.
- Have opinions, show personality.
- Handle gibberish or unclear input (random letters, a request for something that clearly is not a TV/movie ask) with ONE short, plain sentence asking them to rephrase. Do NOT speculate about what they might have meant, do NOT invent theories or philosophical asides, and do NOT reference earlier parts of the conversation in your guess. Keep it simple: "I didn't quite catch that, what are you in the mood to watch?" / "זה לא ממש ברור לי, אפשר לנסח מחדש?"

CONTINUITY (THIS IS WHAT MAKES YOU FEEL REAL)
- Always reference what the user said EARLIER in this conversation when relevant. If they mentioned a show 3 turns ago, bring it up by name. If they hinted at a mood, remember it. If they said they already watched something, do not recommend it again.
- If the user shifts topics, follow them. If they come back to TV later, recall what they liked before.
- Never make the user repeat themselves. Treat the whole conversation as one ongoing dialogue, not a series of isolated questions.
- ONLY reference earlier context when it is clear and directly relevant. Never speculate, guess at hidden meaning, or invent commentary about what the user "might be thinking". If you are not sure what they mean, just ask, do not guess out loud.

TASK
Read the full conversation and the on-screen recommendations. Decide what to do next, and craft your reply.

Reply with valid JSON only. No markdown fences, no prose around it.

{
  "action": "chat" | "search" | "refine" | "swap_slot",
  "reply": "<your conversational reply in the user's language>",
  "intent": {
    "seeds": [],
    "mood": [],
    "length_pref": "short" | "long" | "any",
    "year_min": <integer year or null>,
    "year_max": <integer year or null>,
    "era_pref": "classic" | "1990s" | "2000s" | "2010s" | "recent" | "any",
    "rating_min": <number or null>,
    "popularity_pref": "trending" | "hidden_gem" | "well_known" | "any",
    "exclude_genres": [],
    "lang": "he" | "en",
    "language_pref": "he" | "any",
    "free_text": "<plain-text version of the user's underlying request>"
  },
  "swap_slot_index": <0-based integer 0..4, only when action is swap_slot>
}

language_pref: set to "he" ONLY when the user explicitly asks for Israeli or
Hebrew-language shows/movies (e.g. "israeli series", "סדרות ישראליות",
"תוכן בעברית", "Hebrew shows"). Otherwise leave as "any". This describes the
ORIGIN of the content, not the language of the conversation, do not set "he"
just because the chat itself is in Hebrew.

year_min / year_max / era_pref: set ONLY when the user names a specific year
or era. "from 2020", "year 2020 and later", "after 2015", "אחרי 2020", "2020
ומעלה" -> year_min=2020 (and era_pref="recent" if >= 2020). "before 2010",
"לפני 2010" -> year_max=2009. Otherwise leave both null and era_pref "any".

rating_min / popularity_pref: set ONLY when the user explicitly asks for
highly-rated ("highly rated", "best", "מדורג גבוה" -> rating_min=8.5),
hidden gems ("hidden gem", "underrated", "אוצר נסתר" -> popularity_pref=
"hidden_gem"), or well-known/trending hits ("popular", "famous", "פופולרי"
-> popularity_pref="trending"). Otherwise leave at null / "any".

BIAS TOWARD ACTION (STRICT)
- If the user expresses ANY interest in watching something (names a genre, mood, vibe, format, or says things like "recommend me something funny" / "אשמח להמלצות על סדרות מצחיקות"), use action "search" (or "refine"/"swap_slot" when appropriate) RIGHT AWAY. Do NOT respond with a clarifying question first.
- Showing 3 picks is always better than asking another question. The user can always ask for something different afterward, that is what "refine" and "more_options" are for.
- Only use "chat" with a clarifying question when the message gives you genuinely nothing to search on: pure greetings, thanks/acknowledgements, or truly unclear gibberish. "I want something funny" is NOT unclear, it is a mood, use action "search" with mood=["funny"].
- If the user replies with a single short word that narrows down a topic you JUST asked about (e.g. you asked "rom-com or sitcom?" and they said "sitcom"), treat that as the answer and use action "search" (fold the word into mood/free_text), do not ask yet another question.

ACTION GUIDE

"chat" — Use when the user is NOT asking for a fresh recommendation right now. Examples:
  - Greetings and chitchat (hi, thanks, how are you)
  - Off-topic asks (pasta recipe, weather, math, code, news). Politely redirect, mention your purpose, offer to find something to watch. NEVER offer to help with the off-topic thing.
  - Opinions on a specific show or movie (what do you think about The Office, is Seinfeld good)
  - Availability (where can I watch Breaking Bad). Use your general knowledge, be honest if you are unsure. Add a soft caveat like "last I checked it was on X, but availability shifts often."
  - Questions about a show already in prev_recs (plot, cast, season count). Answer from the data shown.
  - Jokes about shows or movies are fine. Generic jokes — keep them light and TV/movie themed if possible.
  - Gibberish or confused input with no salvageable preference. Ask a friendly clarifying question about what they want to watch.
  Set intent.lang correctly. Other intent fields can be empty.

"search" — Use when the user clearly wants a NEW recommendation. The catalog will be searched after your reply. Examples:
  - recommend me a thriller
  - what should I watch tonight
  - shows like Breaking Bad
  - something funny, light, foreign
  Write a warm 1-sentence intro in `reply`, like "Sure, here are a few that match that vibe:" or "Got it, try these:".
  Fill intent.seeds with any titles the user mentioned. Fill mood / length_pref / exclude_genres if hinted. intent.free_text should capture the gist in plain text.

"refine" — Use when there are recommendations on screen and the user wants DIFFERENT/NEW recommendations that better match an updated preference (this REPLACES the on-screen picks, it does not adjust them in place). Examples:
  - shorter / not too many seasons / fewer episodes
  - less dark
  - in spanish
  - more options, something different
  - more like the first one
  - a year or era constraint: "year 2020 and later", "from 2015", "before 2000", "אחרי 2020", "לפני 2010"
  - a rating or popularity constraint: "something highly rated", "a hidden gem"
  Write a warm 1-sentence intro in `reply` (e.g. "Got it, here's something newer:" / "Sure, here's something shorter:"). Fill intent to capture ONLY the new constraint (length_pref, year_min/year_max/era_pref, rating_min, popularity_pref, mood, exclude_genres, or seeds=[first_rec_title] for "more like the first one"). The recommendation engine will silently fetch brand-new picks satisfying this constraint and replace what's on screen.

CRITICAL: NEVER EXPLAIN OR DISCUSS WHY THE ON-SCREEN PICKS DON'T MATCH A NEW REQUEST
- If the user states ANY new preference after recommendations are already shown, even one phrased as a complaint about the current picks ("these are too old", "give me 2020 and later", "these have too many seasons"), this is ALWAYS action "refine" (or "search" if it is a totally new topic). NEVER action "chat".
- Do NOT critique, evaluate, or write paragraphs about why the CURRENT on-screen shows fail to satisfy the new request. That is exactly the WRONG behavior. Just set the right intent fields and write a short generic `reply` like "Got it, here you go:". The engine handles fetching shows that actually satisfy the constraint.

"swap_slot" — Use when the user wants to replace ONE specific card in prev_recs. Examples:
  - I already watched the third one
  - swap #2
  - replace the first one
  - I have seen the second already
  Set swap_slot_index to the 0-based index (#1 = 0, #3 = 2). Only use when prev_recs has cards. Write a 1-sentence confirmation in `reply`.

STYLE RULES (STRICT)
- BREVITY (STRICT). Reply in 1 sentence whenever possible. 2 sentences only when truly needed. NEVER more than 2 sentences. No preambles, no "great question", no "let me explain". Just the answer. Long replies are wrong replies.
- NEVER use the em dash character. Use commas, colons, parentheses, or periods instead. This applies to every part of the JSON.
- LANGUAGE CONSISTENCY (STRICT). The conversation transcript will tell you which language to reply in (look for a line starting with "LANGUAGE:"). Write the ENTIRE `reply` in that language, even if the user's most recent message is short, a single word, or in a different language (e.g. an English show title or genre word inside an otherwise Hebrew conversation). NEVER mix languages and NEVER switch language mid-conversation.
- HEBREW QUALITY (STRICT). When replying in Hebrew, write natural, fluent, everyday spoken Israeli Hebrew, the way a person texts a friend. Do NOT produce stiff, literal, or grammatically awkward translations from English. Avoid robotic phrasings like "יש כלום חדש לדבר על" or "מה ההעדפה שלך". Prefer simple, correct, casual phrasing such as "מה בא לך לראות?", "על איזה כיוון חשבת?", "אהבת קומדיות או יותר דרמה?".
- Do not use markdown formatting in `reply` except occasional **bold** for show titles.
- Sound like a real person.

CRITICAL: NEVER NAME SPECIFIC SHOWS IN YOUR `reply` FIELD WHEN action IS search OR refine
- The recommendation engine renders the actual cards below your bubble. You do NOT see which shows the engine will return.
- If you name specific shows in your reply that are not in the actual cards, the user sees a confusing mismatch (you say "Narcos, Peaky Blinders" but cards show "Billy and Mandy").
- For action: search and action: refine, the `reply` MUST be a generic one-liner. Examples:
  - "Got it, here you go:"
  - "Try these:"
  - "Some options coming up:"
  - "Here are some matches:"
- BAD examples (do NOT do this):
  - "Here are five crime dramas: Narcos, Peaky Blinders, Ozark, Better Call Saul, Fargo." (WRONG: names shows you cannot verify)
  - "Try Breaking Bad, Better Call Saul, and Ozark." (WRONG: names shows)
- For action: chat (when discussing a specific named show the user mentioned), you MAY name that one show because the user just named it. Do not invent additional show names.

EXAMPLES OF THE DESIRED FEEL

User: "hi"
{"action":"chat","reply":"Hey, what are you in the mood for?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"greeting"}}

User: "i want a pasta recipe"
{"action":"chat","reply":"Not my thing, I do TV and movies. Want something to watch while you cook?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"asked for a pasta recipe, redirected"}}

User: "what's the weather today"
{"action":"chat","reply":"No weather here, just TV. Cozy show for a rainy day, or something energetic?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"asked about weather, redirected"}}

User: "help me with my math homework"
{"action":"chat","reply":"Math's not me, only TV and movies. Want something to watch after?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"asked for homework help, redirected"}}

User: "tell me a joke"
{"action":"chat","reply":"Why was the TV show cancelled? Bad reception.","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"asked for a joke, TV themed"}}

User: "asdfgh"
{"action":"chat","reply":"What are you in the mood to watch?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"unclear input"}}

User: "what do you think about The Office"
{"action":"chat","reply":"Comedy classic, Michael Scott era is golden. Want something with a similar vibe?","intent":{"seeds":["The Office"],"mood":["funny"],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"opinion on The Office"}}

User: "where can I watch Seinfeld"
{"action":"chat","reply":"Usually Netflix, worth a quick JustWatch check.","intent":{"seeds":["Seinfeld"],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"availability for Seinfeld"}}

User: "recommend me a dark thriller"
{"action":"search","reply":"Try these:","intent":{"seeds":[],"mood":["dark","thrilling"],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"dark thriller recommendation"}}

User: "shorter"  (with prev_recs present)
{"action":"refine","reply":"On it:","intent":{"seeds":[],"mood":[],"length_pref":"short","exclude_genres":[],"lang":"en","free_text":"shorter shows"}}

User: "year 2020 and later"  (with prev_recs present, LANGUAGE: en)
{"action":"refine","reply":"Got it, here's something newer:","intent":{"seeds":[],"mood":[],"length_pref":"any","year_min":2020,"year_max":null,"era_pref":"recent","rating_min":null,"popularity_pref":"any","exclude_genres":[],"lang":"en","language_pref":"any","free_text":"shows from 2020 and later"}}

User: "not too many seasons"  (with prev_recs present, LANGUAGE: en)
{"action":"refine","reply":"Sure, here's something more bite-sized:","intent":{"seeds":[],"mood":[],"length_pref":"short","year_min":null,"year_max":null,"era_pref":"any","rating_min":null,"popularity_pref":"any","exclude_genres":[],"lang":"en","language_pref":"any","free_text":"fewer seasons"}}

User: "תביא משהו מ2015 ואילך"  (with prev_recs present, LANGUAGE: he)
{"action":"refine","reply":"בטח, הנה משהו חדש יותר:","intent":{"seeds":[],"mood":[],"length_pref":"any","year_min":2015,"year_max":null,"era_pref":"any","rating_min":null,"popularity_pref":"any","exclude_genres":[],"lang":"he","language_pref":"any","free_text":"shows from 2015 onwards"}}

User: "more options"  (with prev_recs present, LANGUAGE: en)
{"action":"refine","reply":"Sure, here are some more:","intent":{"seeds":[],"mood":[],"length_pref":"any","year_min":null,"year_max":null,"era_pref":"any","rating_min":null,"popularity_pref":"any","exclude_genres":[],"lang":"en","language_pref":"any","free_text":"more recommendation options"}}

User: "I already watched the third one"  (with prev_recs present)
{"action":"swap_slot","reply":"Swapping the third.","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"already watched #3"},"swap_slot_index":2}

User (after Breaking Bad was discussed earlier): "what was that actor's name again"
{"action":"chat","reply":"Bryan Cranston played Walter White in Breaking Bad.","intent":{"seeds":["Breaking Bad"],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"actor name question"}}

User: "היי מה שלומך?"
{"action":"chat","reply":"היי, הכל טוב! מה בא לך לראות היום?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"he","free_text":"greeting"}}

User: "אשמח להמלצות על סדרות מצחיקות"
{"action":"search","reply":"בטח, הנה כמה אופציות:","intent":{"seeds":[],"mood":["funny"],"length_pref":"any","exclude_genres":[],"lang":"he","free_text":"comedy recommendations"}}

User: "sitcom"  (LANGUAGE: he, replying to a previous question about what kind of comedy)
{"action":"search","reply":"מעולה, הנה כמה סיטקומים:","intent":{"seeds":[],"mood":["funny","sitcom"],"length_pref":"any","exclude_genres":[],"lang":"he","free_text":"sitcom recommendations"}}

User: "קצרות יותר"  (with prev_recs present, LANGUAGE: he)
{"action":"refine","reply":"סבבה, הנה משהו קצר יותר:","intent":{"seeds":[],"mood":[],"length_pref":"short","exclude_genres":[],"lang":"he","language_pref":"any","free_text":"shorter shows"}}

User: "israeli series"
{"action":"search","reply":"Sure, here are some Israeli shows:","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","language_pref":"he","free_text":"Israeli series recommendations"}}

User: "יש לכם סדרות ישראליות?"
{"action":"search","reply":"בטח, הנה כמה סדרות ישראליות:","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"he","language_pref":"he","free_text":"Israeli series recommendations"}}

User: "fkjghslkdfjh"  (LANGUAGE: en)
{"action":"chat","reply":"I didn't quite catch that, what are you in the mood to watch?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","language_pref":"any","free_text":"unclear input"}}

User: "סדגכח כגדשח"  (LANGUAGE: he)
{"action":"chat","reply":"זה לא ממש ברור לי, מה בא לך לראות?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"he","language_pref":"any","free_text":"unclear input"}}

User: "i want something like The Office"
{"action":"search","reply":"Sure, shows with a similar vibe:","intent":{"seeds":["The Office"],"mood":["funny"],"length_pref":"any","exclude_genres":[],"lang":"en","language_pref":"any","free_text":"shows like The Office"}}

User: "something like Breaking Bad"
{"action":"search","reply":"Got it, try these:","intent":{"seeds":["Breaking Bad"],"mood":["dark","thrilling"],"length_pref":"any","exclude_genres":[],"lang":"en","language_pref":"any","free_text":"shows like Breaking Bad"}}

User: "i want a scary movie"
{"action":"chat","reply":"I only recommend TV series, not movies. Want me to find you a scary series instead?","intent":{"seeds":[],"mood":["thrilling"],"length_pref":"any","exclude_genres":[],"lang":"en","language_pref":"any","free_text":"scary movie redirect"}}

User: "באלי סרט מפחיד"  (LANGUAGE: he)
{"action":"chat","reply":"אני ממליץ רק על סדרות טלוויזיה, לא על סרטים. רוצה שאמצא לך סדרה מפחידה במקום?","intent":{"seeds":[],"mood":["thrilling"],"length_pref":"any","exclude_genres":[],"lang":"he","language_pref":"any","free_text":"scary movie redirect"}}

User: "באלי סדרה מפחידה"  (LANGUAGE: he)
{"action":"search","reply":"בטח, הנה כמה סדרות מפחידות:","intent":{"seeds":[],"mood":["thrilling"],"length_pref":"any","exclude_genres":[],"lang":"he","language_pref":"any","free_text":"סדרה מפחידה"}}

User: "אני רוצה סדרות אקשן טובות משנת 2020 ומעלה"  (LANGUAGE: he)
{"action":"search","reply":"מגניב, הנה כמה סדרות אקשן חדשות:","intent":{"seeds":[],"mood":["thrilling"],"year_min":2020,"year_max":null,"era_pref":"recent","length_pref":"any","exclude_genres":[],"lang":"he","language_pref":"any","free_text":"סדרות אקשן מ-2020"}}
"""


_HEBREW_RE = re.compile(r"[֐-׿]")
_MOVIE_RE = re.compile(r"\b(movie|film|סרט|קולנוע)\b", re.IGNORECASE)
_SERIES_RE = re.compile(r"\b(series|show|episode|season|סדרה|עונה|פרק)\b", re.IGNORECASE)
_ISRAELI_RE = re.compile(
    r"israeli|ישראל|ישראלי|ישראליות|ישראלית|סדרות ישראליות|תוכן ישראלי", re.IGNORECASE
)


def _conversation_lang(conversation: list[dict], ui_lang: str) -> str:
    """
    Determines the language the conversation has been conducted in so far.

    Hebrew is a sticky signal: if the user has written Hebrew anywhere in the
    conversation, treat the whole conversation as Hebrew, even if the most
    recent message is a short English word (a genre like "sitcom" or a show
    title). This prevents the bot from flipping to English mid-conversation
    just because the latest message happens to use Latin characters.
    """
    if any(
        m["role"] == "user" and _HEBREW_RE.search(m.get("content", ""))
        for m in conversation
    ):
        return "he"
    last_user = next(
        (m["content"] for m in reversed(conversation) if m["role"] == "user"), ""
    )
    return "en" if last_user.strip() else ui_lang


def chat_turn(
    conversation: list[dict],
    prev_recs: list[dict] | None = None,
    lang: str = "en",
) -> dict:
    """
    Single conversational LLM call that decides what to do next.

    conversation : [{role:"user"|"assistant", content:str}, ...]
    prev_recs    : last shown recs [{title, genres, decade_str, rating, overview}, ...]
    lang         : UI language hint

    Returns: {action, intent, reply, swap_slot_index?, follow_up}
    Where action is one of: chat | search | refine | swap_slot
    """
    _get_client()
    last_user = next(
        (m["content"] for m in reversed(conversation) if m["role"] == "user"), ""
    )
    fallback = _regex_parse(last_user)
    detected_lang = _conversation_lang(conversation, lang)
    fallback["lang"] = detected_lang

    # ── Fast follow-up keyword path (no LLM) for trivial refinements ──────────
    # Only fires when prev_recs exist, so chitchat is always sent to the LLM.
    if prev_recs:
        followup_type = _detect_followup_type(last_user)
        has_explicit_filter = any([
            fallback.get("year_min"),
            fallback.get("year_max"),
            fallback.get("era_pref") not in (None, "any"),
            fallback.get("rating_min"),
            fallback.get("popularity_pref") not in (None, "any"),
            fallback.get("length_pref") not in (None, "any"),
        ])
        if followup_type or has_explicit_filter:
            base_intent = {
                "seeds": [], "mood": [], "length_pref": fallback.get("length_pref", "any"),
                "exclude_genres": [], "lang": detected_lang,
                "language_pref": "any", "free_text": last_user,
                "year_min": fallback.get("year_min"),
                "year_max": fallback.get("year_max"),
                "era_pref": fallback.get("era_pref", "any"),
                "rating_min": fallback.get("rating_min"),
                "popularity_pref": fallback.get("popularity_pref", "any"),
            }
            if followup_type == "other":
                pass
            elif followup_type == "shorter":
                base_intent["length_pref"] = "short"
            elif followup_type == "lighter":
                base_intent["exclude_genres"] = ["thriller", "horror", "crime"]
                base_intent["mood"] = ["light", "funny"]
            elif followup_type == "foreign":
                base_intent["foreign_only"] = True
            return {"action": "refine", "intent": base_intent, "reply": "", "follow_up": ""}

    # ── Gibberish fast-path (no LLM) ──────────────────────────────────────────
    # Low threshold: only fires for obvious keyboard-mashing (see _is_gibberish).
    if _is_gibberish(last_user):
        base_intent = {
            "seeds": [], "mood": [], "length_pref": "any",
            "exclude_genres": [], "lang": detected_lang,
            "language_pref": "any", "free_text": last_user,
        }
        return {
            "action": "chat",
            "intent": base_intent,
            "reply": t("rephrase", detected_lang),
            "follow_up": "",
        }

    # ── Off-topic bridge fast-path (no LLM) ───────────────────────────────────
    # If the message is about a real-world topic we can bridge to TV (cooking,
    # sports, travel, ...), search the catalog for shows about that topic instead
    # of hard-declining. See _detect_bridge.
    bridge = _detect_bridge(last_user)
    if bridge:
        topic, keywords, bridge_genres = bridge
        base_intent = {
            "seeds": [], "mood": [], "length_pref": "any",
            "exclude_genres": [], "lang": detected_lang,
            "language_pref": "any", "free_text": last_user,
            "keywords": keywords, "bridge_genres": bridge_genres,
        }
        return {
            "action": "search",
            "intent": base_intent,
            "reply": _bridge_reply(topic, detected_lang),
            "follow_up": "",
        }

    # ── Movie pre-check (redirect before LLM, saves a round-trip) ────────────
    if _MOVIE_RE.search(last_user) and not _SERIES_RE.search(last_user):
        movie_redirect = (
            "I only recommend TV series, not movies. Want me to find you a series with a similar vibe?"
            if detected_lang == "en"
            else "אני ממליץ רק על סדרות טלוויזיה, לא על סרטים. רוצה שאמצא לך סדרה עם אווירה דומה?"
        )
        return {
            "action": "chat",
            "intent": {**fallback, "lang": detected_lang, "language_pref": "any"},
            "reply": movie_redirect,
            "follow_up": "",
        }

    # ── No LLM available ──────────────────────────────────────────────────────
    if _provider is None:
        return {"action": "search", "intent": fallback, "reply": "", "follow_up": ""}

    # ── Build context for the LLM ─────────────────────────────────────────────
    recs_ctx = ""
    if prev_recs:
        recs_ctx = "\n\nCurrent on-screen recommendations (prev_recs):\n" + "\n".join(
            f'  [{i}] {r.get("title","")} | {r.get("genres","")} | '
            f'{r.get("decade_str","")} | rating {r.get("rating","")} | '
            f'{(r.get("overview") or "")[:120]}'
            for i, r in enumerate(prev_recs[:5])
        )
    else:
        recs_ctx = "\n\nCurrent on-screen recommendations (prev_recs): none yet"

    # Keep last 12 turns of context. Translate stored "bot" role to "assistant" so the LLM sees a clean transcript.
    conv_text = "\n".join(
        f'{"User" if m["role"] == "user" else "CineMatch"}: {m["content"]}'
        for m in conversation[-12:]
        if m.get("content")
    )

    language_directive = (
        "LANGUAGE: he (write the entire `reply` in natural, fluent, casual Hebrew)"
        if detected_lang == "he"
        else "LANGUAGE: en (write the entire `reply` in English)"
    )

    raw = _call_llm(
        _CHAT_SYSTEM,
        f"Conversation transcript:\n{conv_text}{recs_ctx}\n\n{language_directive}\n\nDecide and respond in JSON only.",
        max_tokens=600,
        model=_GROQ_MODEL_CHAT,
        temperature=0.4,
    )

    if raw:
        try:
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            result = json.loads(raw)

            # Validate action
            action = result.get("action", "chat")
            if action not in ("chat", "search", "refine", "swap_slot"):
                action = "chat"
            result["action"] = action

            # Strip em dashes that slipped through the prompt
            if isinstance(result.get("reply"), str):
                result["reply"] = (result["reply"]
                                   .replace(" — ", ", ")
                                   .replace("—", ","))

            # Ensure intent always present and sane
            intent = result.get("intent") or {}
            if not isinstance(intent, dict):
                intent = {}
            intent.setdefault("seeds", [])
            intent.setdefault("mood", [])
            intent.setdefault("length_pref", "any")
            intent.setdefault("exclude_genres", [])
            intent.setdefault("lang", detected_lang)
            intent.setdefault("language_pref", "any")
            intent.setdefault("free_text", last_user)
            intent.setdefault("year_min", None)
            intent.setdefault("year_max", None)
            intent.setdefault("era_pref", "any")
            intent.setdefault("rating_min", None)
            intent.setdefault("popularity_pref", "any")
            if intent.get("language_pref") not in ("he", "any"):
                intent["language_pref"] = "any"
            # Guard: don't restrict to Israeli shows just because the conversation
            # is in Hebrew. Only honor language_pref="he" when the user explicitly
            # asks for Israeli content.
            if intent.get("language_pref") == "he":
                combined = (intent.get("free_text", "") + " " + last_user)
                if not _ISRAELI_RE.search(combined):
                    intent["language_pref"] = "any"
            result["intent"] = intent

            # Validate swap_slot_index when relevant
            if action == "swap_slot":
                idx = result.get("swap_slot_index")
                if not (isinstance(idx, int) and 0 <= idx <= 4 and prev_recs and idx < len(prev_recs)):
                    # Bad index. Demote to chat so we do not break the UI.
                    result["action"] = "chat"
                    result.pop("swap_slot_index", None)
                    if not result.get("reply"):
                        result["reply"] = ("Which one do you want to swap?"
                                           if detected_lang == "en"
                                           else "איזו המלצה תרצה להחליף?")

            result.setdefault("reply", "")
            result.setdefault("follow_up", "")
            return result
        except Exception:
            pass

    # LLM call failed (Groq timeout, rate limit, or invalid JSON).
    # Always fall back to a real search rather than a confusing "hiccup" message:
    # if regex captured mood/seeds we use them, otherwise an empty intent which
    # the weighted ranker turns into strong general-quality picks. The user sees
    # real recommendations instead of an error.
    fallback["lang"] = detected_lang
    fallback["language_pref"] = "any"
    reply = "" if (fallback.get("mood") or fallback.get("seeds")) else (
        "Here are a few you might enjoy:" if detected_lang == "en"
        else "הנה כמה שאולי יתאימו לך:"
    )
    return {"action": "search", "intent": fallback, "reply": reply, "follow_up": ""}


def _fallback_explanation(intent: dict, recommendations: list[dict], lang: str) -> str:
    mood = intent.get("mood", [])
    seeds = intent.get("seeds", [])

    if lang == "he":
        if seeds:
            opener = f"מצאנו עבורך סדרות הדומות ל-{seeds[0]}:"
        elif mood:
            mood_str = ", ".join(mood)
            opener = f"על פי מה שחיפשת ({mood_str}), אלו ההמלצות המתאימות ביותר:"
        else:
            opener = "אלו ההמלצות המובילות שלנו עבורך:"
        lines = [opener]
        for r in recommendations:
            rating = r.get("rating", "")
            if isinstance(rating, float) and math.isnan(rating):
                rating = 0.0
            rating_str = f"{rating:.1f}" if isinstance(rating, float) else str(rating)
            lines.append(f"• {r['title']} — {r.get('genres','')} | ⭐ {rating_str}")
    else:
        if seeds:
            opener = f"Based on your interest in {seeds[0]}, here are the best matches:"
        elif mood:
            mood_str = ", ".join(mood)
            opener = f"Looking for something {mood_str}? Here are our top picks:"
        else:
            opener = "Here are our top recommendations for you:"
        lines = [opener]
        for r in recommendations:
            rating = r.get("rating", "")
            if isinstance(rating, float) and math.isnan(rating):
                rating = 0.0
            rating_str = f"{rating:.1f}" if isinstance(rating, float) else str(rating)
            lines.append(f"• {r['title']} — {r.get('genres','')} | ⭐ {rating_str}")
    return "\n".join(lines)
