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
    detailsLoading: "טוען פרטים נוספים...",
    genericError: "מצטערים, קרתה תקלה. נסו שוב מאוחר יותר.",
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
    detailsLoading: "Loading more details...",
    genericError: "Sorry, something went wrong. Please try again later.",
  },
};
