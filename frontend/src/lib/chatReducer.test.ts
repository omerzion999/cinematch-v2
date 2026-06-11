import { describe, it, expect } from "vitest";
import {
  chatReducer,
  createInitialState,
  type ChatState,
  type ChoiceMessage,
  type RecommendationsMessage,
  type TextMessage,
} from "./chatReducer";
import { ONBOARDING_QUESTIONS } from "./onboarding";
import type { RecCard } from "./types";

const SAMPLE_CARD: RecCard = {
  title: "Severance",
  genres: "Drama, Sci-Fi & Fantasy",
  rating: 8.7,
  overview: "A team at Lumon Industries...",
  poster_path: "/abc.jpg",
  decade_str: "2020s",
  num_seasons: 2,
};

describe("createInitialState", () => {
  it("starts in the intro phase with one choice message offering onboarding or skip", () => {
    const state = createInitialState("he");

    expect(state.phase).toBe("intro");
    expect(state.lang).toBe("he");
    expect(state.messages).toHaveLength(1);

    const opening = state.messages[0] as ChoiceMessage;
    expect(opening.type).toBe("choice");
    expect(opening.options.map((o) => o.value)).toEqual(["start", "skip"]);
    expect(opening.selectedValue).toBeUndefined();
  });
});

describe("chatReducer", () => {
  it("START_ONBOARDING closes the opening message and shows the first onboarding question", () => {
    const state = createInitialState("he");

    const next = chatReducer(state, { type: "START_ONBOARDING" });

    expect(next.phase).toBe("onboarding");
    expect(next.onboardingStepIndex).toBe(0);
    expect(next.messages).toHaveLength(2);

    const opening = next.messages[0] as ChoiceMessage;
    expect(opening.selectedValue).toBe("start");
    expect(opening.selectedLabel).toBeTruthy();

    const firstQuestion = next.messages[1] as ChoiceMessage;
    expect(firstQuestion.type).toBe("choice");
    expect(firstQuestion.options.map((o) => o.value)).toEqual(
      ONBOARDING_QUESTIONS[0].options.map((o) => o.value)
    );
  });

  it("SKIP_TO_CHAT closes the opening message and goes straight to chat with a greeting", () => {
    const state = createInitialState("he");

    const next = chatReducer(state, { type: "SKIP_TO_CHAT" });

    expect(next.phase).toBe("chat");
    expect(next.messages).toHaveLength(2);

    const opening = next.messages[0] as ChoiceMessage;
    expect(opening.selectedValue).toBe("skip");

    const greeting = next.messages[1] as TextMessage;
    expect(greeting.type).toBe("text");
    expect(greeting.role).toBe("assistant");
    expect(greeting.content.length).toBeGreaterThan(0);
  });

  it("walks through all 5 onboarding questions and ends in loading_recommend with all answers recorded", () => {
    let state = chatReducer(createInitialState("he"), { type: "START_ONBOARDING" });

    const answeredValues = ["drama", "medium", "recent", "serious_drama", "hidden_gem"];
    for (let i = 0; i < ONBOARDING_QUESTIONS.length; i++) {
      const question = ONBOARDING_QUESTIONS[i];
      state = chatReducer(state, {
        type: "ANSWER_ONBOARDING_QUESTION",
        questionId: question.id,
        value: answeredValues[i],
      });
    }

    expect(state.phase).toBe("loading_recommend");
    expect(state.onboardingAnswers).toEqual({
      genre: "drama",
      length: "medium",
      era: "recent",
      tone: "serious_drama",
      popularity: "hidden_gem",
    });

    // opening message + 5 questions, all closed (answered)
    const choiceMessages = state.messages.filter(
      (m): m is ChoiceMessage => m.type === "choice"
    );
    expect(choiceMessages).toHaveLength(6);
    for (const message of choiceMessages) {
      expect(message.selectedValue).toBeTruthy();
    }
  });

  it("RECOMMEND_SUCCESS appends intro, recommendation cards, and outro, and returns to chat phase", () => {
    let state = chatReducer(createInitialState("he"), { type: "START_ONBOARDING" });
    state = { ...state, phase: "loading_recommend" };

    const next = chatReducer(state, {
      type: "RECOMMEND_SUCCESS",
      intro: "אתה בטעם של: דרמות מתח",
      outro: "מקווה שאהבת!",
      cards: [SAMPLE_CARD],
    });

    expect(next.phase).toBe("chat");
    expect(next.prevRecs).toEqual([SAMPLE_CARD]);

    const last3 = next.messages.slice(-3);
    expect((last3[0] as TextMessage).content).toBe("אתה בטעם של: דרמות מתח");
    expect((last3[1] as RecommendationsMessage).type).toBe("recommendations");
    expect((last3[1] as RecommendationsMessage).cards).toEqual([SAMPLE_CARD]);
    expect((last3[2] as TextMessage).content).toBe("מקווה שאהבת!");
  });

  it("RECOMMEND_ERROR appends an assistant message and returns to chat phase", () => {
    const state: ChatState = { ...createInitialState("he"), phase: "loading_recommend" };

    const next = chatReducer(state, { type: "RECOMMEND_ERROR", message: "שגיאה זמנית" });

    expect(next.phase).toBe("chat");
    const last = next.messages[next.messages.length - 1] as TextMessage;
    expect(last.content).toBe("שגיאה זמנית");
    expect(last.role).toBe("assistant");
  });

  it("SEND_USER_MESSAGE appends a user message and moves to loading_chat", () => {
    const state: ChatState = { ...createInitialState("he"), phase: "chat" };

    const next = chatReducer(state, { type: "SEND_USER_MESSAGE", content: "תמליץ לי על קומדיה" });

    expect(next.phase).toBe("loading_chat");
    const last = next.messages[next.messages.length - 1] as TextMessage;
    expect(last.role).toBe("user");
    expect(last.content).toBe("תמליץ לי על קומדיה");
  });

  it("CHAT_SUCCESS with recommendations appends reply, cards, and explanation, and updates prevRecs", () => {
    const state: ChatState = { ...createInitialState("he"), phase: "loading_chat" };

    const next = chatReducer(state, {
      type: "CHAT_SUCCESS",
      reply: "הנה כמה הצעות:",
      cards: [SAMPLE_CARD],
      explanation: "שתיהן דרמות מתח עכשוויות.",
    });

    expect(next.phase).toBe("chat");
    expect(next.prevRecs).toEqual([SAMPLE_CARD]);

    const last3 = next.messages.slice(-3);
    expect((last3[0] as TextMessage).content).toBe("הנה כמה הצעות:");
    expect((last3[1] as RecommendationsMessage).cards).toEqual([SAMPLE_CARD]);
    expect((last3[2] as TextMessage).content).toBe("שתיהן דרמות מתח עכשוויות.");
  });

  it("CHAT_SUCCESS with recommendations but no explanation appends reply and cards only", () => {
    const state: ChatState = { ...createInitialState("he"), phase: "loading_chat" };

    const next = chatReducer(state, {
      type: "CHAT_SUCCESS",
      reply: "הנה כמה הצעות:",
      cards: [SAMPLE_CARD],
    });

    expect(next.phase).toBe("chat");
    expect(next.prevRecs).toEqual([SAMPLE_CARD]);

    const last2 = next.messages.slice(-2);
    expect((last2[0] as TextMessage).content).toBe("הנה כמה הצעות:");
    expect((last2[1] as RecommendationsMessage).cards).toEqual([SAMPLE_CARD]);
    expect(next.messages).toHaveLength(state.messages.length + 2);
  });

  it("CHAT_SUCCESS without recommendations only appends the reply", () => {
    const state: ChatState = { ...createInitialState("he"), phase: "loading_chat" };

    const next = chatReducer(state, { type: "CHAT_SUCCESS", reply: "שלום! איך אפשר לעזור?" });

    expect(next.phase).toBe("chat");
    expect(next.prevRecs).toBeNull();
    const last = next.messages[next.messages.length - 1] as TextMessage;
    expect(last.content).toBe("שלום! איך אפשר לעזור?");
  });

  it("CHAT_ERROR appends an assistant message and returns to chat phase", () => {
    const state: ChatState = { ...createInitialState("he"), phase: "loading_chat" };

    const next = chatReducer(state, { type: "CHAT_ERROR", message: "שגיאה זמנית" });

    expect(next.phase).toBe("chat");
    const last = next.messages[next.messages.length - 1] as TextMessage;
    expect(last.content).toBe("שגיאה זמנית");
  });

  it("TOGGLE_LANG flips between he and en", () => {
    const state = createInitialState("he");

    const next = chatReducer(state, { type: "TOGGLE_LANG" });
    expect(next.lang).toBe("en");

    const back = chatReducer(next, { type: "TOGGLE_LANG" });
    expect(back.lang).toBe("he");
  });

  it("RESTORE replaces the entire state (used to load persisted state)", () => {
    const restored: ChatState = { ...createInitialState("en"), phase: "chat" };

    const next = chatReducer(createInitialState("he"), { type: "RESTORE", state: restored });

    expect(next).toEqual(restored);
  });

  it("RESET_CONVERSATION returns to the initial state, preserving the current language", () => {
    const userMessage: TextMessage = { id: "u1", type: "text", role: "user", content: "hi" };
    const state: ChatState = {
      ...createInitialState("en"),
      phase: "chat",
      messages: [...createInitialState("en").messages, userMessage],
      prevRecs: [SAMPLE_CARD],
    };

    const next = chatReducer(state, { type: "RESET_CONVERSATION" });

    expect(next.phase).toBe("intro");
    expect(next.lang).toBe("en");
    expect(next.prevRecs).toBeNull();
    expect(next.onboardingStepIndex).toBe(-1);
    expect(next.messages).toHaveLength(1);
    expect(next.messages[0].type).toBe("choice");
  });
});
