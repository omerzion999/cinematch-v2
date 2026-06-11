import type { OnboardingAnswers } from "./types";

export interface OnboardingOption {
  value: string;
  label: { he: string; en: string };
}

export interface OnboardingQuestion {
  id: keyof OnboardingAnswers;
  prompt: { he: string; en: string };
  options: OnboardingOption[];
}

export const ONBOARDING_QUESTIONS: OnboardingQuestion[] = [
  {
    id: "genre",
    prompt: {
      he: "איזה ז'אנר הכי מדבר אליך?",
      en: "Which genre speaks to you the most?",
    },
    options: [
      { value: "drama", label: { he: "דרמה", en: "Drama" } },
      { value: "comedy", label: { he: "קומדיה", en: "Comedy" } },
      {
        value: "action_adventure",
        label: { he: "אקשן והרפתקאות", en: "Action & Adventure" },
      },
      {
        value: "scifi_fantasy",
        label: { he: "מד\"ב ופנטזיה", en: "Sci-Fi & Fantasy" },
      },
      { value: "crime", label: { he: "פשע", en: "Crime" } },
      { value: "animation", label: { he: "אנימציה", en: "Animation" } },
      { value: "romance", label: { he: "רומנטיקה", en: "Romance" } },
      { value: "any", label: { he: "לא משנה", en: "Doesn't matter" } },
    ],
  },
  {
    id: "length",
    prompt: {
      he: "כמה עונות אתה מעדיף?",
      en: "How many seasons do you prefer?",
    },
    options: [
      {
        value: "short",
        label: { he: "קצר / מיני (1-2 עונות)", en: "Short / mini-series (1-2 seasons)" },
      },
      {
        value: "medium",
        label: { he: "בינוני (3-5 עונות)", en: "Medium (3-5 seasons)" },
      },
      { value: "long", label: { he: "ארוך (6+ עונות)", en: "Long (6+ seasons)" } },
      { value: "any", label: { he: "לא משנה", en: "Doesn't matter" } },
    ],
  },
  {
    id: "era",
    prompt: {
      he: "איזה עידן מעניין אותך?",
      en: "Which era interests you?",
    },
    options: [
      { value: "recent", label: { he: "חדש (2020 ואילך)", en: "Recent (2020+)" } },
      { value: "classic", label: { he: "קלאסי", en: "Classic" } },
      { value: "any", label: { he: "לא משנה", en: "Doesn't matter" } },
    ],
  },
  {
    id: "tone",
    prompt: {
      he: "איזה טון אתה מחפש?",
      en: "What tone are you in the mood for?",
    },
    options: [
      { value: "light_fun", label: { he: "קליל ומשעשע", en: "Light & fun" } },
      {
        value: "serious_drama",
        label: { he: "רציני ודרמטי", en: "Serious & dramatic" },
      },
      {
        value: "thriller_action",
        label: { he: "מותח ואקשן", en: "Thriller & action" },
      },
      { value: "any", label: { he: "לא משנה", en: "Doesn't matter" } },
    ],
  },
  {
    id: "popularity",
    prompt: {
      he: "מה אתה מעדיף?",
      en: "What do you prefer?",
    },
    options: [
      {
        value: "well_known",
        label: { he: "להיטים מוכרים", en: "Well-known hits" },
      },
      {
        value: "hidden_gem",
        label: { he: "פנינים נסתרות", en: "Hidden gems" },
      },
      { value: "any", label: { he: "לא משנה", en: "Doesn't matter" } },
    ],
  },
];
