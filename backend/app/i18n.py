"""
Bilingual (Hebrew/English) templates for bot messages generated server-side.

Most UI text lives in the React frontend's own i18n. This module only covers
messages assembled by the backend itself: the onboarding-recommendation
intro/outro and a couple of fallback/error strings.
"""

STRINGS: dict[str, dict[str, str]] = {
    "recommend_intro": {
        "he": "הנה כמה המלצות בכיוון {label}:",
        "en": "A few {label} picks for you:",
    },
    "recommend_outro": {
        "he": "רוצה עוד? פשוט תכתוב לי.",
        "en": "Want more? Just say the word.",
    },
    "no_recommendations": {
        "he": "לא הצלחתי למצוא המלצות מתאימות הפעם. אפשר לנסות עם תשובות אחרות?",
        "en": "I couldn't find matching recommendations this time. Want to try different answers?",
    },
    "show_not_found": {
        "he": "לא מצאתי את הסדרה הזו במאגר שלנו.",
        "en": "I couldn't find that show in our catalog.",
    },
    "not_in_catalog": {
        "he": "זה לא מופיע במאגר הנתונים שלי, אולי תנסה לנסח את זה מחדש?",
        "en": "That doesn't seem to be in my catalog, maybe try rephrasing?",
    },
    "rephrase": {
        "he": "זה לא ממש ברור לי, מה בא לך לראות?",
        "en": "I didn't quite catch that, what are you in the mood to watch?",
    },
}


def t(key: str, lang: str = "en", **kwargs) -> str:
    entry = STRINGS.get(key, {})
    template = entry.get(lang, entry.get("en", ""))
    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError):
            return template
    return template
