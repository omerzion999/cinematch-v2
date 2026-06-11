"""
Bilingual (Hebrew/English) templates for bot messages generated server-side.

Most UI text lives in the React frontend's own i18n. This module only covers
messages assembled by the backend itself: the onboarding-recommendation
intro/outro and a couple of fallback/error strings.
"""

STRINGS: dict[str, dict[str, str]] = {
    "recommend_intro": {
        "he": "על סמך מה שסיפרת לי, אני חושב שתתחבר לטעם הזה: {label}. הנה כמה סדרות שכדאי לבדוק:",
        "en": "Based on what you told me, I think you're into: {label}. Here are a few shows worth checking out:",
    },
    "recommend_outro": {
        "he": "מקווה שאהבת את ההמלצות! אפשר להמשיך לדבר איתי על סדרות, לבקש עוד המלצות, או לשאול אותי כל דבר.",
        "en": "Hope you like these picks! Feel free to keep chatting, ask for more recommendations, or anything else.",
    },
    "no_recommendations": {
        "he": "לא הצלחתי למצוא המלצות מתאימות הפעם. אפשר לנסות עם תשובות אחרות?",
        "en": "I couldn't find matching recommendations this time. Want to try different answers?",
    },
    "show_not_found": {
        "he": "לא מצאתי את הסדרה הזו במאגר שלנו.",
        "en": "I couldn't find that show in our catalog.",
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
