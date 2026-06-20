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
      answers: {
        genre: "any",
        length: "any",
        era: "any",
        popularity: "any",
      },
      lang: "he",
    });
    expect(result.current.state.prevRecs).toHaveLength(1);
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
