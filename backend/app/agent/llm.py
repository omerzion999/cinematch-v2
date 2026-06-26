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
    # Groq path: retry up to 3 times with an 8-second per-attempt timeout.
    # Render's free-tier egress can be slow on a cold start, so 5s was sometimes
    # tripping even on healthy calls; 8s is more forgiving while staying snappy.
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
                    timeout=8,
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
        "thrilling": ["thriller","thrill","suspense","מותחן","action","אקשן","crime","פשע","מתח"],
        "horror":    ["horror","scary","מפחיד","אימה","zombie","gore","supernatural"],
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

    # Genre detection so the offline path (no LLM / rate-limited) still searches
    # on the right genre. Maps to the onboarding genre vocabulary.
    genre = "any"
    for g, kws in _GENRE_WORD_MAP.items():
        if any(kw in ql for kw in kws):
            genre = g
            break

    return {
        "seeds": seeds, "mood": mood, "length_pref": length_pref,
        "language_pref": language_pref, "era_pref": era_pref,
        "year_min": year_min, "year_max": year_max,
        "status": status, "popularity_pref": popularity_pref,
        "binge_pref": binge_pref, "rating_min": rating_min,
        "exclude_genres": [], "lang": lang, "genre": genre, "free_text": query,
    }


# Free-text genre words (en + he) -> onboarding genre vocabulary, for the offline
# parser. Ordered so more specific genres win over generic ones.
_GENRE_WORD_MAP = {
    "crime": ["crime", "פשע", "פשעים"],
    "scifi_fantasy": ["sci-fi", "scifi", "science fiction", "fantasy", "מדע בדיוני", "פנטזיה", "מדע-בדיוני"],
    "animation": ["animation", "animated", "anime", "cartoon", "אנימציה", "מצויר", "אנימה"],
    "action_adventure": ["action", "adventure", "אקשן", "הרפתקא", "פעולה"],
    "comedy": ["comedy", "comedies", "sitcom", "funny", "קומדיה", "מצחיק", "סיטקום"],
    "drama": ["drama", "dramas", "dramatic", "דרמה", "דרמטי"],
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
    """Conservative keyboard-mashing detector. Two paths:

    Latin: a single word of 6+ letters with no vowels (e.g. "sjfnkfds").
    Hebrew: a single word of 7+ Hebrew-only letters where fewer than half the
    characters are unique (e.g. "יגכיגיגכ" — 3 distinct / 8 total = 0.375).
    Real Hebrew words at that length consistently have ≥ 0.5 unique-char ratio.
    """
    text = msg.strip()
    if len(text) < 6 or " " in text or "\n" in text:
        return False
    if re.fullmatch(r"[A-Za-z]+", text):
        return not any(c in _VOWELS for c in text)
    if re.fullmatch(r"[א-ת]+", text) and len(text) >= 7:
        return (len(set(text)) / len(text)) < 0.5
    return False


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
                  "restaurant", "cuisine", "baking", "bake", "baker", "food"],
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
You are CineMatch AI, a warm, witty bilingual TV-series recommender over a curated catalog of 11,013 titles. Lead every conversation toward a great pick. Reply with VALID JSON only (no markdown, no prose).

STYLE
- Reply in the user's language; obey the "LANGUAGE:" line in the transcript and never switch mid-chat, even for a one-word or English-title reply.
- Real person: friendly, opinionated, SHORT. 1 sentence ideally, never more than 2. No preambles. Never use the em dash character (use commas or periods).
- Hebrew: natural casual spoken Israeli Hebrew, never a stiff translation.
- Use the conversation history: do not re-ask what they already told you, do not re-recommend what they watched. Reference earlier context only when clearly relevant; if unsure what they mean, just ask.

OUTPUT
{"action":"chat"|"search"|"refine"|"swap_slot","reply":"<reply in user's language>","intent":{"seeds":[],"mood":[],"length_pref":"short"|"long"|"any","year_min":<int|null>,"year_max":<int|null>,"era_pref":"classic"|"1990s"|"2000s"|"2010s"|"recent"|"any","rating_min":<num|null>,"popularity_pref":"trending"|"hidden_gem"|"well_known"|"any","exclude_genres":[],"lang":"he"|"en","language_pref":"he"|"any","free_text":"<gist>"},"swap_slot_index":<0-4, swap_slot only>}

INTENT (set a field ONLY when explicit, else any/null/[])
- seeds: titles the user named. mood: explicit vibe words (funny/dark/emotional/thrilling/light).
- language_pref="he" ONLY if they ask for Israeli/Hebrew content (origin, not chat language).
- year_min/year_max/era_pref ONLY for a stated year/era ("from 2020"->year_min=2020,era_pref="recent"; "before 2010"->year_max=2009).
- rating_min/popularity_pref ONLY if stated ("best"->rating_min=8.5; "hidden gem"->hidden_gem; "popular"->trending).

ACTIONS
- search: a NEW recommendation request with something to go on (a genre, mood, vibe, or "shows like X"). If they NAME a show ("a series like The Office", "similar to Friends"), set seeds=[that show] and search, EVEN IF recs are already on screen: a newly named title is a NEW request, never a swap or refine of the old picks. reply = a GENERIC one-liner ("Try these:"). NEVER name shows in a search/refine reply: you cannot see the engine's picks, and naming wrong ones creates a mismatch.
- refine: recs already on screen and a new/changed preference (shorter, less dark, in spanish, "2020 and later", "a hidden gem", "more options", "more like the first one"->seeds=[first title]). Put ONLY the new constraint in intent; reply = short generic line. NEVER critique why the current picks fail, just set intent.
- swap_slot: replace ONE card. ONLY for "swap #2" / "I watched the third" / "not the second one"; swap_slot_index 0-based; only if prev_recs exist. A message that names a NEW show ("like The Office") is a search, NOT a swap.
- chat: NOT a fresh recommendation. Greetings, thanks, opinions on a named show, availability (general knowledge + soft caveat), questions about a show in prev_recs, jokes, gibberish (one short "rephrase" line), and statements/identity ("i am israeli", "i'm tired") -> react briefly and LEAD toward a pick. A totally BLANK request with nothing specific ("recommend something", "what should I watch", "תמליץ לי משהו") -> ask ONE short guiding question (which genre or vibe), do not dump generic picks. Off-topic (recipe/weather/math): one line "that's not me, I do TV", then offer to find something to watch; never offer to do the off-topic thing. Movies: say you only do series, offer a series instead. You MAY name a show the user just named; never invent other titles.

EXAMPLES (intent fields not shown default to any/null/[])
User: "hi" -> {"action":"chat","reply":"Hey, what are you in the mood for?","intent":{"lang":"en","free_text":"greeting"}}
User: "i am an israeli" -> {"action":"chat","reply":"Nice! Want Israeli series, or a genre you're into?","intent":{"lang":"en","language_pref":"any","free_text":"identity, lead to a pick"}}
User: "i'm so tired today" -> {"action":"chat","reply":"Then something easy and fun, a comedy maybe?","intent":{"lang":"en","free_text":"statement, lead to a pick"}}
User: "i want a pasta recipe" -> {"action":"chat","reply":"Not my thing, I do TV. Something to watch while you cook?","intent":{"lang":"en","free_text":"off-topic redirect"}}
User: "what do you think about The Office" -> {"action":"chat","reply":"Comedy gold, the Michael Scott era especially. Want a similar vibe?","intent":{"seeds":["The Office"],"mood":["funny"],"lang":"en","free_text":"opinion"}}
User: "where can I watch Seinfeld" -> {"action":"chat","reply":"Usually Netflix, worth a quick JustWatch check.","intent":{"seeds":["Seinfeld"],"lang":"en","free_text":"availability"}}
User: "recommend a dark thriller" -> {"action":"search","reply":"Try these:","intent":{"mood":["dark","thrilling"],"lang":"en","free_text":"dark thriller"}}
User: "something like Breaking Bad" -> {"action":"search","reply":"Got it, try these:","intent":{"seeds":["Breaking Bad"],"mood":["dark","thrilling"],"lang":"en","free_text":"like Breaking Bad"}}
User: "can you recommend a series like The Office" (prev_recs are crime shows) -> {"action":"search","reply":"Sure, similar vibes coming up:","intent":{"seeds":["The Office"],"lang":"en","free_text":"like The Office"}}
User: "recommend something" -> {"action":"chat","reply":"Happy to help, what genre or vibe are you after?","intent":{"lang":"en","free_text":"blank request, ask a question"}}
User: "i want a scary movie" -> {"action":"chat","reply":"I only do series, not movies. Want a scary series instead?","intent":{"mood":["thrilling"],"lang":"en","free_text":"movie redirect"}}
User: "year 2020 and later" (prev_recs) -> {"action":"refine","reply":"Got it, something newer:","intent":{"year_min":2020,"era_pref":"recent","lang":"en","free_text":"2020+"}}
User: "more options" (prev_recs) -> {"action":"refine","reply":"Sure, more coming up:","intent":{"lang":"en","free_text":"more options"}}
User: "I already watched the third one" (prev_recs) -> {"action":"swap_slot","reply":"Swapping the third.","intent":{"lang":"en","free_text":"watched #3"},"swap_slot_index":2}
User: "fkjghslkdfjh" -> {"action":"chat","reply":"I didn't quite catch that, what are you in the mood to watch?","intent":{"lang":"en","free_text":"unclear"}}
User: "אשמח להמלצות על סדרות מצחיקות" -> {"action":"search","reply":"בטח, הנה כמה אופציות:","intent":{"mood":["funny"],"lang":"he","free_text":"comedies"}}
User: "יש לכם סדרות ישראליות?" -> {"action":"search","reply":"בטח, הנה כמה סדרות ישראליות:","intent":{"lang":"he","language_pref":"he","free_text":"israeli series"}}
User: "קצרות יותר" (prev_recs) -> {"action":"refine","reply":"סבבה, משהו קצר יותר:","intent":{"length_pref":"short","lang":"he","free_text":"shorter"}}
"""


_HEBREW_RE = re.compile(r"[֐-׿]")
_MOVIE_RE = re.compile(r"\b(movie|film|סרט|קולנוע)\b", re.IGNORECASE)
_SERIES_RE = re.compile(r"\b(series|show|episode|season|סדרה|עונה|פרק)\b", re.IGNORECASE)
_ISRAELI_RE = re.compile(
    r"israeli|ישראל|ישראלי|ישראליות|ישראלית|סדרות ישראליות|תוכן ישראלי", re.IGNORECASE
)

# Signals used by the offline fallback (no-LLM / LLM-failure path) to decide
# whether a message is actually a recommendation request.
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|howdy|sup|good\s+(morning|evening|afternoon))\b"
    r"|שלום|היי|היי|מה\s+(נשמע|קורה|המצב|העניינים)",
    re.IGNORECASE,
)
_REQUEST_VERB_RE = re.compile(
    r"\b(recommend|recommendation|suggest|watch|find|show me|looking for|give me|"
    r"what should i watch|something to watch)\b"
    r"|תמליצ|תמליץ|המלצ|לראות|תביא|מחפש|בא לי|תן לי|מה לראות",
    re.IGNORECASE,
)
_GENRE_SIGNAL_RE = re.compile(
    r"\b(drama|comedy|comedies|sitcom|crime|thriller|horror|scary|sci-?fi|fantasy|"
    r"animation|anime|documentary|docu|romance|romantic|action|adventure|mystery)\b"
    r"|דרמה|קומדיה|סיטקום|פשע|מותחן|אימה|מפחיד|מדע\s*בדיוני|פנטזיה|אנימציה|דוקו|"
    r"רומנט|אקשן|הרפתק|מתח|תעלומ",
    re.IGNORECASE,
)


def _offline_chat(reply: str, lang: str, fallback: dict) -> dict:
    return {
        "action": "chat",
        "intent": {**fallback, "lang": lang, "language_pref": "any"},
        "reply": reply,
        "follow_up": "",
    }


def _offline_turn(last_user: str, lang: str, fallback: dict) -> dict:
    """
    Decide what to do WITHOUT the LLM (no key, or the LLM call failed). Only run a
    search when there is a real recommendation signal; otherwise reply
    conversationally and steer toward a recommendation, so chitchat like
    "i am an israeli" never dumps random picks. Gibberish, bridge, and movie
    redirects are already handled before this is reached.
    """
    he = lang == "he"
    m = (last_user or "").strip()

    # A real recommendation signal = something to personalize on (a genre, mood,
    # or a named seed). Only then do we search.
    strong_signal = bool(
        fallback.get("mood")
        or fallback.get("seeds")
        or _GENRE_SIGNAL_RE.search(m)
    )
    if strong_signal:
        return {
            "action": "search",
            "intent": {**fallback, "lang": lang, "language_pref": "any"},
            "reply": "",
            "follow_up": "",
        }

    # Bare request ("recommend something", "what should I watch") with nothing to
    # personalize on -> guide the user with one short question instead of dumping
    # generic picks. This makes the agent direct the conversation toward a good,
    # personal match (the way a real assistant would).
    if _REQUEST_VERB_RE.search(m):
        return _offline_chat(
            "בשמחה! איזה ז'אנר או אווירה בא לך? למשל פשע, קומדיה, או משהו אפל."
            if he else
            "Happy to help! What genre or vibe are you in the mood for, like crime, comedy, or something dark?",
            lang, fallback,
        )

    if any(tok in m.lower() for tok in _CHAT_TOKENS):
        return _offline_chat(
            "בכיף! רוצה עוד המלצה?" if he else "Anytime. Want another pick?", lang, fallback
        )
    if _GREETING_RE.search(m):
        return _offline_chat(
            "היי! אני CineMatch, ממליץ על סדרות. מה בא לך לראות, ז'אנר או אווירה?"
            if he else
            "Hey! I'm CineMatch, I recommend TV series. What are you in the mood for, a genre or a vibe?",
            lang, fallback,
        )
    if _ISRAELI_RE.search(m):
        return _offline_chat(
            "מגניב! רוצה סדרות ישראליות, או שיש ז'אנר שבא לך?"
            if he else
            "Cool. Want Israeli series, or is there a genre you're after?",
            lang, fallback,
        )
    return _offline_chat(
        "אני ממליץ על סדרות טלוויזיה. ספר לי ז'אנר או אווירה ואמצא לך משהו."
        if he else
        "I recommend TV series. Tell me a genre or a vibe and I'll find you something.",
        lang, fallback,
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

    # ── Explicit similarity-request fast-path (no LLM, robust to rate limits) ──
    # "recommend a series like The Office", "shows like X", "similar to X",
    # "כמו X" -> a FRESH similarity search seeded on X, even when recs are already
    # on screen. A newly named title is a NEW request, never a swap/refine of the
    # stale picks (the bug where "a series like The Office" swapped a Breaking Bad
    # pick). Deterministic so it works even when the LLM is rate-limited.
    if fallback.get("seeds"):
        base_intent = {
            "seeds": fallback["seeds"], "mood": [], "length_pref": fallback.get("length_pref", "any"),
            "exclude_genres": [], "lang": detected_lang, "language_pref": "any",
            "free_text": last_user,
            "year_min": fallback.get("year_min"), "year_max": fallback.get("year_max"),
            "era_pref": fallback.get("era_pref", "any"),
            "rating_min": fallback.get("rating_min"),
            "popularity_pref": fallback.get("popularity_pref", "any"),
        }
        return {
            "action": "search",
            "intent": base_intent,
            "reply": "הנה כמה המלצות דומות:" if detected_lang == "he" else "Here are a few similar picks:",
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
        return _offline_turn(last_user, detected_lang, fallback)

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

    # LLM call failed (Groq timeout, rate limit, or invalid JSON). Use the same
    # offline decision as the no-key path: search only on a real signal, else a
    # short conversational reply that steers toward a recommendation. This keeps
    # the agent sane even when Groq is down or rate-limited (e.g. mid-demo).
    return _offline_turn(last_user, detected_lang, fallback)


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
