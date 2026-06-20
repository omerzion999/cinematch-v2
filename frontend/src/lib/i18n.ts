import type { Lang } from "./types";

export const UI_STRINGS: Record<Lang, Record<string, string>> = {
  he: {
    openingMessage:
      "היי! אני CineMatch 🎬 - הבוט שעוזר למצוא את הסדרה הבאה שתאהב. רוצה לענות על כמה שאלות קצרות כדי שאכיר את הטעם שלך?",
    startOnboarding: "כן, בוא נתחיל",
    skipToChat: "לא תודה, אני רוצה להתקדם לצ'אט הרגיל",
    skipGreeting:
      "בסדר גמור! אפשר לבקש ממני המלצות, לשאול על סדרה ספציפית, או סתם לפטפט.",
    processing: "רגע אחד, מחפש את ההמלצות הכי מתאימות לך...",
    inputPlaceholder: "כתוב הודעה...",
    send: "שלח",
    languageToggleLabel: "English",
    ratingLabel: "דירוג",
    seasonsLabel: "עונות",
    castLabel: "שחקנים ראשיים",
    watchProvidersLabel: "איפה לצפות",
    trailerLabel: "טריילר",
    closeModal: "סגור",
    closeDialog: "סגור חלון",
    detailsLoading: "טוען פרטים נוספים...",
    genericError: "מצטערים, קרתה תקלה. נסה שוב מאוחר יותר.",
    newConversation: "שיחה חדשה",
    questionProgress: "שאלה {step} מתוך {total}",
    onboardingTypeHint: "אפשר לבחור למעלה, או לכתוב משהו כמו \"קומדיה\".",
  },
  en: {
    openingMessage:
      "Hi! I'm CineMatch 🎬 - here to help you find your next favorite show. Want to answer a few quick questions so I can learn your taste?",
    startOnboarding: "Yes, let's start",
    skipToChat: "No thanks, take me to the regular chat",
    skipGreeting:
      "Sure thing! Ask me for recommendations, ask about a specific show, or just chat.",
    processing: "One sec, finding the best picks for you...",
    inputPlaceholder: "Type a message...",
    send: "Send",
    languageToggleLabel: "עברית",
    ratingLabel: "Rating",
    seasonsLabel: "Seasons",
    castLabel: "Cast",
    watchProvidersLabel: "Where to watch",
    trailerLabel: "Trailer",
    closeModal: "Close",
    closeDialog: "Close dialog",
    detailsLoading: "Loading more details...",
    genericError: "Sorry, something went wrong. Please try again later.",
    newConversation: "New conversation",
    questionProgress: "Question {step} of {total}",
    onboardingTypeHint: "Pick one above, or type something like \"comedy\".",
  },
};

/**
 * Mirrors the non-templated entries of the backend's `app/i18n.py` STRINGS
 * dict, so assistant messages built from those strings can be retranslated
 * instantly via lookup instead of a round-trip to /api/translate.
 */
export const BACKEND_STRINGS: Record<Lang, Record<string, string>> = {
  he: {
    recommend_outro:
      "מקווה שאהבת את ההמלצות! אפשר להמשיך לדבר איתי על סדרות, לבקש עוד המלצות, או לשאול אותי כל דבר.",
    no_recommendations:
      "לא הצלחתי למצוא המלצות מתאימות הפעם. אפשר לנסות עם תשובות אחרות?",
    show_not_found: "לא מצאתי את הסדרה הזו במאגר שלנו.",
    not_in_catalog: "זה לא מופיע במאגר הנתונים שלי, אולי תנסה לנסח את זה מחדש?",
  },
  en: {
    recommend_outro:
      "Hope you like these picks! Feel free to keep chatting, ask for more recommendations, or anything else.",
    no_recommendations:
      "I couldn't find matching recommendations this time. Want to try different answers?",
    show_not_found: "I couldn't find that show in our catalog.",
    not_in_catalog: "That doesn't seem to be in my catalog, maybe try rephrasing?",
  },
};
