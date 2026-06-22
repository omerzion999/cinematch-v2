import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useChatState } from "./useChatState";
import * as api from "@/lib/api";
import { chatReducer, createInitialState, type TextMessage } from "@/lib/chatReducer";
import { ONBOARDING_QUESTIONS } from "@/lib/onboarding";
import { savePersistedState, STORAGE_KEY } from "@/lib/storage";

vi.mock("@/lib/api");

describe("useChatState", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.resetAllMocks();
  });

  it("starts from createInitialState when nothing is persisted", () => {
    const { result } = renderHook(() => useChatState("he"));

    expect(result.current.state.phase).toBe("intro");
    expect(result.current.state.messages).toHaveLength(1);
  });

  it("restores persisted state on mount", () => {
    const persisted = chatReducer(createInitialState("en"), { type: "SKIP_TO_CHAT" });
    savePersistedState(persisted);

    const { result } = renderHook(() => useChatState("he"));

    expect(result.current.state.phase).toBe("chat");
    expect(result.current.state.lang).toBe("en");
  });

  it("persists state to localStorage after each dispatch", () => {
    const { result } = renderHook(() => useChatState("he"));

    act(() => {
      result.current.dispatch({ type: "TOGGLE_LANG" });
    });

    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY)!);
    expect(stored.lang).toBe("en");
  });

  it("calls postRecommend once onboarding completes and applies the result", async () => {
    vi.mocked(api.postRecommend).mockResolvedValue({
      intro: "אתה בטעם של: דרמות מתח",
      outro: "מקווה שאהבת!",
      cluster_id: 2,
      recommendations: [
        {
          title: "Severance",
          genres: "Drama, Sci-Fi & Fantasy",
          rating: 8.7,
          overview: "...",
          poster_path: "/abc.jpg",
          decade_str: "2020s",
          num_seasons: 2,
          binge_fit_score: 0.82,
          explanation: "...",
        },
      ],
    });

    const { result } = renderHook(() => useChatState("he"));

    act(() => {
      result.current.dispatch({ type: "START_ONBOARDING" });
    });
    for (const question of ONBOARDING_QUESTIONS) {
      act(() => {
        result.current.dispatch({
          type: "ANSWER_ONBOARDING_QUESTION",
          questionId: question.id,
          value: "any",
        });
      });
    }

    expect(result.current.state.phase).toBe("loading_recommend");

    await waitFor(() => {
      expect(result.current.state.phase).toBe("chat");
    });

    expect(api.postRecommend).toHaveBeenCalledWith({
      answers: { genre: "any", era: "any", popularity: "any" },
      seeds: [],
      lang: "he",
    });
    expect(result.current.state.prevRecs).toHaveLength(1);
  });

  it("fetches seeds when a concrete genre is chosen and carries picks into postRecommend", async () => {
    const seedCard = {
      title: "Breaking Bad",
      genres: "Crime, Drama",
      rating: 9.5,
      overview: "...",
      poster_path: null,
      decade_str: "2000s",
      num_seasons: 5,
    };
    vi.mocked(api.getSeeds).mockResolvedValue({ genre: "crime", seeds: [seedCard] });
    vi.mocked(api.postRecommend).mockResolvedValue({
      intro: "x", outro: "y", cluster_id: 1,
      recommendations: [{ ...seedCard, title: "Ozark", binge_fit_score: 0.8, explanation: "..." }],
    });

    const { result } = renderHook(() => useChatState("en"));

    act(() => {
      result.current.dispatch({ type: "START_ONBOARDING" });
    });
    act(() => {
      result.current.dispatch({ type: "ANSWER_ONBOARDING_QUESTION", questionId: "genre", value: "crime" });
    });
    expect(result.current.state.phase).toBe("seed_pick");

    await waitFor(() => {
      expect(api.getSeeds).toHaveBeenCalledWith("crime", "en");
      const seedMsg = result.current.state.messages.find((m) => m.type === "seedpick");
      expect(seedMsg && seedMsg.type === "seedpick" && seedMsg.cards).toHaveLength(1);
    });

    act(() => {
      result.current.dispatch({ type: "TOGGLE_SEED", title: "Breaking Bad" });
    });
    act(() => {
      result.current.dispatch({ type: "CONFIRM_SEEDS" });
    });
    act(() => {
      result.current.dispatch({ type: "ANSWER_ONBOARDING_QUESTION", questionId: "era", value: "any" });
    });
    act(() => {
      result.current.dispatch({ type: "ANSWER_ONBOARDING_QUESTION", questionId: "popularity", value: "any" });
    });

    await waitFor(() => {
      expect(result.current.state.phase).toBe("chat");
    });
    expect(api.postRecommend).toHaveBeenCalledWith({
      answers: { genre: "crime", era: "any", popularity: "any" },
      seeds: ["Breaking Bad"],
      lang: "en",
    });
  });

  it("calls postChat when the user sends a message and applies the reply", async () => {
    vi.mocked(api.postChat).mockResolvedValue({
      reply: "הנה כמה הצעות:",
      recommendations: null,
      explanation: null,
    });

    const { result } = renderHook(() => useChatState("he"));

    act(() => {
      result.current.dispatch({ type: "SKIP_TO_CHAT" });
    });
    act(() => {
      result.current.dispatch({ type: "SEND_USER_MESSAGE", content: "תמליץ לי על קומדיה" });
    });

    expect(result.current.state.phase).toBe("loading_chat");

    await waitFor(() => {
      expect(result.current.state.phase).toBe("chat");
    });

    expect(api.postChat).toHaveBeenCalled();
    const lastMessage = result.current.state.messages[
      result.current.state.messages.length - 1
    ] as TextMessage;
    expect(lastMessage.content).toBe("הנה כמה הצעות:");
  });

  it("dispatches RECOMMEND_ERROR with a generic message when postRecommend rejects", async () => {
    vi.mocked(api.postRecommend).mockRejectedValue(new Error("network error"));

    const { result } = renderHook(() => useChatState("he"));

    act(() => {
      result.current.dispatch({ type: "START_ONBOARDING" });
    });
    for (const question of ONBOARDING_QUESTIONS) {
      act(() => {
        result.current.dispatch({
          type: "ANSWER_ONBOARDING_QUESTION",
          questionId: question.id,
          value: "any",
        });
      });
    }

    await waitFor(() => {
      expect(result.current.state.phase).toBe("chat");
    });

    const lastMessage = result.current.state.messages[
      result.current.state.messages.length - 1
    ] as TextMessage;
    expect(lastMessage.content.length).toBeGreaterThan(0);
  });
});
