import { BACKEND_STRINGS, UI_STRINGS } from "./i18n";
import { ONBOARDING_QUESTIONS, type OnboardingQuestion } from "./onboarding";
import type { Lang, OnboardingAnswers, RecCard } from "./types";

export type Phase =
  | "intro"
  | "onboarding"
  | "seed_pick"
  | "loading_recommend"
  | "loading_chat"
  | "chat";

export interface TextMessage {
  id: string;
  type: "text";
  role: "user" | "assistant";
  content: string;
}

export interface ChoiceOption {
  value: string;
  label: string;
}

export interface ChoiceMessage {
  id: string;
  type: "choice";
  role: "assistant";
  prompt: string;
  options: ChoiceOption[];
  selectedValue?: string;
  selectedLabel?: string;
  /** 1-based step + total for onboarding questions ("Question 2 of 4"). */
  progress?: { step: number; total: number };
}

export interface RecommendationsMessage {
  id: string;
  type: "recommendations";
  role: "assistant";
  cards: RecCard[];
}

/** Seed-picker step: selectable recognizable shows for the chosen genre. */
export interface SeedPickMessage {
  id: string;
  type: "seedpick";
  role: "assistant";
  /** null while /api/seeds is loading. */
  cards: RecCard[] | null;
  selectedTitles: string[];
  /** true once the user confirmed or skipped (locks the step). */
  done: boolean;
}

export type ChatMessage = TextMessage | ChoiceMessage | RecommendationsMessage | SeedPickMessage;

export interface ChatState {
  phase: Phase;
  lang: Lang;
  onboardingStepIndex: number;
  onboardingAnswers: OnboardingAnswers;
  /** Titles the user picked in the seed step (drives multi-seed similarity). */
  seeds: string[];
  messages: ChatMessage[];
  prevRecs: RecCard[] | null;
}

export type ChatAction =
  | { type: "START_ONBOARDING" }
  | { type: "SKIP_TO_CHAT" }
  | { type: "ANSWER_ONBOARDING_QUESTION"; questionId: keyof OnboardingAnswers; value: string }
  | { type: "SEEDS_LOADED"; cards: RecCard[] }
  | { type: "TOGGLE_SEED"; title: string }
  | { type: "CONFIRM_SEEDS" }
  | { type: "SKIP_SEEDS" }
  | { type: "RECOMMEND_SUCCESS"; intro: string; outro: string; cards: RecCard[] }
  | { type: "RECOMMEND_ERROR"; message: string }
  | { type: "SEND_USER_MESSAGE"; content: string }
  | {
      type: "CHAT_SUCCESS";
      reply: string;
      cards?: RecCard[] | null;
      explanation?: string | null;
    }
  | { type: "CHAT_ERROR"; message: string }
  | { type: "TOGGLE_LANG" }
  | { type: "APPLY_TRANSLATIONS"; translations: { id: string; content: string }[] }
  | { type: "RESTORE"; state: ChatState }
  | { type: "RESET_CONVERSATION" };

const DEFAULT_ONBOARDING_ANSWERS: OnboardingAnswers = {
  genre: "any",
  era: "any",
  popularity: "any",
};

/** Max seeds a user may pick in the seed step. */
export const MAX_SEEDS = 3;

export function createInitialState(lang: Lang): ChatState {
  const strings = UI_STRINGS[lang];
  return {
    phase: "intro",
    lang,
    onboardingStepIndex: -1,
    onboardingAnswers: { ...DEFAULT_ONBOARDING_ANSWERS },
    seeds: [],
    messages: [
      {
        id: crypto.randomUUID(),
        type: "choice",
        role: "assistant",
        prompt: strings.openingMessage,
        options: [
          { value: "start", label: strings.startOnboarding },
          { value: "skip", label: strings.skipToChat },
        ],
      },
    ],
    prevRecs: null,
  };
}

function buildQuestionMessage(
  question: OnboardingQuestion<keyof OnboardingAnswers>,
  lang: Lang,
  stepIndex: number
): ChoiceMessage {
  return {
    id: crypto.randomUUID(),
    type: "choice",
    role: "assistant",
    prompt: question.prompt[lang],
    options: question.options.map((o) => ({ value: o.value, label: o.label[lang] })),
    progress: { step: stepIndex + 1, total: ONBOARDING_QUESTIONS.length },
  };
}

function buildSeedPickMessage(): SeedPickMessage {
  return {
    id: crypto.randomUUID(),
    type: "seedpick",
    role: "assistant",
    cards: null,
    selectedTitles: [],
    done: false,
  };
}

/**
 * Advances onboarding past the question at `fromIndex`: pushes the next question,
 * or moves to loading_recommend when there are no more questions.
 */
function advanceAfterQuestion(
  state: ChatState,
  fromIndex: number,
  answers: OnboardingAnswers,
  messages: ChatMessage[]
): ChatState {
  const nextIndex = fromIndex + 1;
  if (nextIndex < ONBOARDING_QUESTIONS.length) {
    return {
      ...state,
      onboardingAnswers: answers,
      onboardingStepIndex: nextIndex,
      phase: "onboarding",
      messages: [...messages, buildQuestionMessage(ONBOARDING_QUESTIONS[nextIndex], state.lang, nextIndex)],
    };
  }
  return { ...state, onboardingAnswers: answers, phase: "loading_recommend", messages };
}

/**
 * Looks up `content` in known static string tables (UI_STRINGS / BACKEND_STRINGS)
 * and returns its `toLang` equivalent, or null if `content` isn't a known
 * static string (e.g. LLM-generated free text).
 */
export function findStaticTranslation(content: string, fromLang: Lang, toLang: Lang): string | null {
  const uiFrom = UI_STRINGS[fromLang];
  const uiTo = UI_STRINGS[toLang];
  for (const key of ["skipGreeting", "genericError"] as const) {
    if (uiFrom[key] === content) return uiTo[key];
  }

  const backendFrom = BACKEND_STRINGS[fromLang];
  const backendTo = BACKEND_STRINGS[toLang];
  for (const key of Object.keys(backendFrom)) {
    if (backendFrom[key] === content) return backendTo[key];
  }

  return null;
}

/**
 * Rebuilds a ChoiceMessage's prompt/options/selectedLabel in `toLang`, using
 * the same static tables the message was originally built from
 * (createInitialState's intro choice or an ONBOARDING_QUESTIONS entry).
 * Returns the message unchanged if it doesn't match either table.
 */
export function translateChoiceMessage(message: ChoiceMessage, fromLang: Lang, toLang: Lang): ChoiceMessage {
  const fromStrings = UI_STRINGS[fromLang];
  const toStrings = UI_STRINGS[toLang];

  const isIntro =
    message.prompt === fromStrings.openingMessage &&
    message.options.some((o) => o.value === "start") &&
    message.options.some((o) => o.value === "skip");

  if (isIntro) {
    const options = message.options.map((o) => ({
      value: o.value,
      label: o.value === "start" ? toStrings.startOnboarding : toStrings.skipToChat,
    }));
    let selectedLabel = message.selectedLabel;
    if (message.selectedValue === "start") selectedLabel = toStrings.startOnboarding;
    else if (message.selectedValue === "skip") selectedLabel = toStrings.skipToChat;
    return { ...message, prompt: toStrings.openingMessage, options, selectedLabel };
  }

  const question = ONBOARDING_QUESTIONS.find((q) => q.prompt[fromLang] === message.prompt);
  if (!question) return message;

  const options = message.options.map((o) => {
    const match = question.options.find((qo) => qo.value === o.value);
    return { value: o.value, label: match ? match.label[toLang] : o.label };
  });
  const selectedOption = question.options.find((qo) => qo.value === message.selectedValue);
  const selectedLabel = selectedOption ? selectedOption.label[toLang] : message.selectedLabel;

  return { ...message, prompt: question.prompt[toLang], options, selectedLabel };
}

// Assumes the last message is the open ChoiceMessage being answered; if not
// (e.g. a malformed RESTORE), this is a silent no-op rather than a crash.
function closeLastChoice(
  messages: ChatMessage[],
  selectedValue: string,
  selectedLabel: string
): ChatMessage[] {
  const last = messages[messages.length - 1];
  if (last.type !== "choice") return messages;
  const updated: ChoiceMessage = { ...last, selectedValue, selectedLabel };
  return [...messages.slice(0, -1), updated];
}

function textMessage(role: "user" | "assistant", content: string): TextMessage {
  return { id: crypto.randomUUID(), type: "text", role, content };
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "START_ONBOARDING": {
      const strings = UI_STRINGS[state.lang];
      const firstQuestion = ONBOARDING_QUESTIONS[0];
      return {
        ...state,
        phase: "onboarding",
        onboardingStepIndex: 0,
        messages: [
          ...closeLastChoice(state.messages, "start", strings.startOnboarding),
          buildQuestionMessage(firstQuestion, state.lang, 0),
        ],
      };
    }

    case "SKIP_TO_CHAT": {
      const strings = UI_STRINGS[state.lang];
      return {
        ...state,
        phase: "chat",
        messages: [
          ...closeLastChoice(state.messages, "skip", strings.skipToChat),
          textMessage("assistant", strings.skipGreeting),
        ],
      };
    }

    case "ANSWER_ONBOARDING_QUESTION": {
      const question = ONBOARDING_QUESTIONS[state.onboardingStepIndex];
      if (!question) return state;
      const option = question.options.find((o) => o.value === action.value);
      const label = option ? option.label[state.lang] : action.value;

      const updatedAnswers = {
        ...state.onboardingAnswers,
        [action.questionId]: action.value,
      } as OnboardingAnswers;
      const messages = closeLastChoice(state.messages, action.value, label);

      // Seed detour: after a concrete genre (not "Surprise me"), show the seed
      // picker before the remaining refine questions. "any" has no seed list, so
      // it falls through to the normal question flow.
      if (question.id === "genre" && action.value !== "any") {
        return {
          ...state,
          onboardingAnswers: updatedAnswers,
          seeds: [],
          phase: "seed_pick",
          messages: [...messages, buildSeedPickMessage()],
        };
      }

      return advanceAfterQuestion(state, state.onboardingStepIndex, updatedAnswers, messages);
    }

    case "SEEDS_LOADED": {
      const messages = state.messages.map((m) =>
        m.type === "seedpick" && !m.done ? { ...m, cards: action.cards } : m
      );
      return { ...state, messages };
    }

    case "TOGGLE_SEED": {
      const messages = state.messages.map((m) => {
        if (m.type !== "seedpick" || m.done) return m;
        const has = m.selectedTitles.includes(action.title);
        if (has) {
          return { ...m, selectedTitles: m.selectedTitles.filter((t) => t !== action.title) };
        }
        if (m.selectedTitles.length >= MAX_SEEDS) return m; // cap reached, ignore
        return { ...m, selectedTitles: [...m.selectedTitles, action.title] };
      });
      return { ...state, messages };
    }

    case "CONFIRM_SEEDS":
    case "SKIP_SEEDS": {
      const open = state.messages.find((m) => m.type === "seedpick" && !m.done) as
        | SeedPickMessage
        | undefined;
      const picked = action.type === "CONFIRM_SEEDS" && open ? open.selectedTitles : [];
      const messages = state.messages.map((m) =>
        m.type === "seedpick" && !m.done ? { ...m, selectedTitles: picked, done: true } : m
      );
      // Genre is always step 0; resume the remaining refine questions after it.
      return advanceAfterQuestion(
        { ...state, seeds: picked },
        state.onboardingStepIndex,
        state.onboardingAnswers,
        messages
      );
    }

    case "RECOMMEND_SUCCESS": {
      const recsMessage: RecommendationsMessage = {
        id: crypto.randomUUID(),
        type: "recommendations",
        role: "assistant",
        cards: action.cards,
      };
      return {
        ...state,
        phase: "chat",
        prevRecs: action.cards,
        messages: [
          ...state.messages,
          textMessage("assistant", action.intro),
          recsMessage,
          textMessage("assistant", action.outro),
        ],
      };
    }

    case "RECOMMEND_ERROR":
      return {
        ...state,
        phase: "chat",
        messages: [...state.messages, textMessage("assistant", action.message)],
      };

    case "SEND_USER_MESSAGE":
      return {
        ...state,
        phase: "loading_chat",
        messages: [...state.messages, textMessage("user", action.content)],
      };

    case "CHAT_SUCCESS": {
      const messages: ChatMessage[] = [...state.messages];
      if (action.reply && action.reply.trim()) {
        messages.push(textMessage("assistant", action.reply));
      }
      let prevRecs = state.prevRecs;

      if (action.cards && action.cards.length > 0) {
        messages.push({
          id: crypto.randomUUID(),
          type: "recommendations",
          role: "assistant",
          cards: action.cards,
        });
        prevRecs = action.cards;

        if (action.explanation) {
          messages.push(textMessage("assistant", action.explanation));
        }
      }

      return { ...state, phase: "chat", messages, prevRecs };
    }

    case "CHAT_ERROR":
      return {
        ...state,
        phase: "chat",
        messages: [...state.messages, textMessage("assistant", action.message)],
      };

    case "TOGGLE_LANG": {
      const fromLang = state.lang;
      const toLang = fromLang === "he" ? "en" : "he";
      const messages = state.messages.map((message) => {
        if (message.type === "choice") {
          return translateChoiceMessage(message, fromLang, toLang);
        }
        if (message.type === "text" && message.role === "assistant") {
          const translated = findStaticTranslation(message.content, fromLang, toLang);
          if (translated !== null) {
            return { ...message, content: translated };
          }
        }
        return message;
      });
      return { ...state, lang: toLang, messages };
    }

    case "APPLY_TRANSLATIONS": {
      const translationById = new Map(action.translations.map((t) => [t.id, t.content]));
      const messages = state.messages.map((message) => {
        if (message.type === "text" && translationById.has(message.id)) {
          return { ...message, content: translationById.get(message.id)! };
        }
        return message;
      });
      return { ...state, messages };
    }

    case "RESTORE":
      return action.state;

    case "RESET_CONVERSATION":
      return createInitialState(state.lang);

    default: {
      const _exhaustive: never = action;
      return _exhaustive;
    }
  }
}
