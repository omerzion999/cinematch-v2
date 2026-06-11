import { UI_STRINGS } from "./i18n";
import { ONBOARDING_QUESTIONS, type OnboardingQuestion } from "./onboarding";
import type { Lang, OnboardingAnswers, RecCard } from "./types";

export type Phase = "intro" | "onboarding" | "loading_recommend" | "loading_chat" | "chat";

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
}

export interface RecommendationsMessage {
  id: string;
  type: "recommendations";
  role: "assistant";
  cards: RecCard[];
}

export type ChatMessage = TextMessage | ChoiceMessage | RecommendationsMessage;

export interface ChatState {
  phase: Phase;
  lang: Lang;
  onboardingStepIndex: number;
  onboardingAnswers: OnboardingAnswers;
  messages: ChatMessage[];
  prevRecs: RecCard[] | null;
}

export type ChatAction =
  | { type: "START_ONBOARDING" }
  | { type: "SKIP_TO_CHAT" }
  | { type: "ANSWER_ONBOARDING_QUESTION"; questionId: keyof OnboardingAnswers; value: string }
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
  | { type: "RESTORE"; state: ChatState };

const DEFAULT_ONBOARDING_ANSWERS: OnboardingAnswers = {
  genre: "any",
  length: "any",
  era: "any",
  tone: "any",
  popularity: "any",
};

export function createInitialState(lang: Lang): ChatState {
  const strings = UI_STRINGS[lang];
  return {
    phase: "intro",
    lang,
    onboardingStepIndex: -1,
    onboardingAnswers: { ...DEFAULT_ONBOARDING_ANSWERS },
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
  lang: Lang
): ChoiceMessage {
  return {
    id: crypto.randomUUID(),
    type: "choice",
    role: "assistant",
    prompt: question.prompt[lang],
    options: question.options.map((o) => ({ value: o.value, label: o.label[lang] })),
  };
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
          buildQuestionMessage(firstQuestion, state.lang),
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

      const nextIndex = state.onboardingStepIndex + 1;
      if (nextIndex < ONBOARDING_QUESTIONS.length) {
        return {
          ...state,
          onboardingAnswers: updatedAnswers,
          onboardingStepIndex: nextIndex,
          messages: [...messages, buildQuestionMessage(ONBOARDING_QUESTIONS[nextIndex], state.lang)],
        };
      }
      return {
        ...state,
        onboardingAnswers: updatedAnswers,
        phase: "loading_recommend",
        messages,
      };
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
      const messages: ChatMessage[] = [...state.messages, textMessage("assistant", action.reply)];
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

    case "TOGGLE_LANG":
      return { ...state, lang: state.lang === "he" ? "en" : "he" };

    case "RESTORE":
      return action.state;

    default: {
      const _exhaustive: never = action;
      return _exhaustive;
    }
  }
}
