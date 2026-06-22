import type { Lang, OnboardingAnswers } from "./types";

export interface OnboardingOption<K extends keyof OnboardingAnswers = keyof OnboardingAnswers> {
  value: OnboardingAnswers[K];
  label: { he: string; en: string };
}

export interface OnboardingQuestion<K extends keyof OnboardingAnswers = keyof OnboardingAnswers> {
  id: K;
  prompt: { he: string; en: string };
  options: OnboardingOption<K>[];
}

export const ONBOARDING_QUESTIONS: { [K in keyof OnboardingAnswers]: OnboardingQuestion<K> }[keyof OnboardingAnswers][] = [
  {
    id: "genre",
    prompt: {
      he: "איזה ז'אנר הכי מדבר אליך?",
      en: "Which genre speaks to you the most?",
    },
    options: [
      { value: "drama", label: { he: "דרמה", en: "Drama" } },
      { value: "comedy", label: { he: "קומדיה", en: "Comedy" } },
      { value: "crime", label: { he: "פשע", en: "Crime" } },
      {
        value: "scifi_fantasy",
        label: { he: "מד\"ב ופנטזיה", en: "Sci-Fi & Fantasy" },
      },
      {
        value: "action_adventure",
        label: { he: "אקשן והרפתקאות", en: "Action & Adventure" },
      },
      { value: "animation", label: { he: "אנימציה", en: "Animation" } },
      { value: "any", label: { he: "הפתע אותי", en: "Surprise me" } },
    ],
  },
  {
    id: "era",
    prompt: {
      he: "איזה עידן מעניין אותך?",
      en: "Which era interests you?",
    },
    options: [
      { value: "recent", label: { he: "חדש (2020 ואילך)", en: "New (2020+)" } },
      { value: "modern", label: { he: "מודרני (שנות ה-2010)", en: "Modern (2010s)" } },
      { value: "classic", label: { he: "קלאסי (לפני 2010)", en: "Classic (pre-2010)" } },
      { value: "any", label: { he: "לא משנה", en: "Any" } },
    ],
  },
  {
    id: "popularity",
    prompt: {
      he: "מה בא לך?",
      en: "What do you prefer?",
    },
    options: [
      {
        value: "well_known",
        label: { he: "להיטים גדולים", en: "Big hits" },
      },
      {
        value: "hidden_gem",
        label: { he: "פנינים נסתרות", en: "Hidden gems" },
      },
      { value: "any", label: { he: "לא משנה", en: "Any" } },
    ],
  },
];

/**
 * Free-text synonyms per option value, both languages, so a typed answer can be
 * mapped to a tap option. Keys are option values; entries are lowercase tokens
 * matched as substrings against the normalized user text.
 */
const TYPED_SYNONYMS: Record<string, string[]> = {
  // genre
  drama: ["drama", "דרמה", "דרמטי"],
  comedy: ["comedy", "comed", "funny", "sitcom", "קומדיה", "מצחיק", "הומור", "קומי"],
  crime: ["crime", "detective", "mafia", "פשע", "פשיעה", "בלש", "מאפיה", "פושע"],
  scifi_fantasy: [
    "sci-fi", "scifi", "sci fi", "science fiction", "fantasy", "מד\"ב", "מדב",
    "מדע בדיוני", "פנטזיה", "פנטסיה",
  ],
  action_adventure: [
    "action", "adventure", "אקשן", "פעולה", "הרפתקה", "הרפתקאות",
  ],
  animation: ["animation", "animated", "cartoon", "anime", "אנימציה", "מצויר", "אנימה"],
  // era
  recent: ["recent", "new", "newest", "latest", "2020", "2021", "2022", "2023", "2024", "חדש", "עכשווי", "אחרון"],
  modern: ["modern", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "מודרני", "עשור"],
  classic: ["classic", "old", "older", "retro", "vintage", "90", "2000", "קלאסי", "ישן", "פעם", "וינטג"],
  // popularity
  well_known: ["popular", "big hit", "hits", "famous", "well known", "well-known", "mainstream", "להיט", "להיטים", "מפורסם", "פופולרי", "מוכר"],
  hidden_gem: ["hidden", "gem", "gems", "underrated", "obscure", "niche", "unknown", "פנינה", "פנינים", "נסתר", "לא מוכר", "אנדרגראונד"],
  // shared catch-all
  any: ["any", "anything", "whatever", "surprise", "surprise me", "doesn't matter", "dont care", "לא משנה", "הכל", "הפתע", "מה שבא"],
};

/**
 * Maps a free-text answer to one of `question`'s option values, or null if no
 * confident match. Tries exact value/label match first, then synonym tokens.
 */
export function resolveTypedAnswer(
  question: OnboardingQuestion,
  text: string,
  lang: Lang
): OnboardingAnswers[keyof OnboardingAnswers] | null {
  const normalized = text.trim().toLowerCase();
  if (!normalized) return null;

  // Exact value or label match (either language).
  for (const option of question.options) {
    if (
      normalized === String(option.value) ||
      normalized === option.label[lang].toLowerCase() ||
      normalized === option.label.he.toLowerCase() ||
      normalized === option.label.en.toLowerCase()
    ) {
      return option.value;
    }
  }

  // Synonym token match, restricted to this question's option values.
  for (const option of question.options) {
    const tokens = TYPED_SYNONYMS[String(option.value)] ?? [];
    if (tokens.some((token) => normalized.includes(token))) {
      return option.value;
    }
  }

  return null;
}
