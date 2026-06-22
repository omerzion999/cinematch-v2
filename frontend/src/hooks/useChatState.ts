import { useEffect, useReducer } from "react";
import { getSeeds, postChat, postRecommend } from "@/lib/api";
import {
  chatReducer,
  createInitialState,
  type ChatAction,
  type ChatState,
  type TextMessage,
} from "@/lib/chatReducer";
import { UI_STRINGS } from "@/lib/i18n";
import { loadPersistedState, savePersistedState } from "@/lib/storage";
import type { Lang } from "@/lib/types";

export interface UseChatStateResult {
  state: ChatState;
  dispatch: React.Dispatch<ChatAction>;
}

export function useChatState(defaultLang: Lang): UseChatStateResult {
  const [state, dispatch] = useReducer(
    chatReducer,
    undefined,
    () => loadPersistedState() ?? createInitialState(defaultLang)
  );

  useEffect(() => {
    savePersistedState(state);
  }, [state]);

  // Entered the seed-picker step -> fetch recognizable shows for the chosen genre.
  useEffect(() => {
    if (state.phase !== "seed_pick") return;

    let cancelled = false;
    getSeeds(state.onboardingAnswers.genre, state.lang)
      .then((response) => {
        if (cancelled) return;
        if (response.seeds.length === 0) {
          dispatch({ type: "SKIP_SEEDS" });
        } else {
          dispatch({ type: "SEEDS_LOADED", cards: response.seeds });
        }
      })
      .catch(() => {
        // Never strand onboarding: skip the seed step on any failure.
        if (!cancelled) dispatch({ type: "SKIP_SEEDS" });
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.phase]);

  // Onboarding finished -> ask the backend for personalized recommendations.
  useEffect(() => {
    if (state.phase !== "loading_recommend") return;

    let cancelled = false;
    postRecommend({ answers: state.onboardingAnswers, seeds: state.seeds, lang: state.lang })
      .then((response) => {
        if (cancelled) return;
        if (response.recommendations.length === 0) {
          dispatch({ type: "RECOMMEND_ERROR", message: response.intro });
        } else {
          dispatch({
            type: "RECOMMEND_SUCCESS",
            intro: response.intro,
            outro: response.outro,
            cards: response.recommendations,
          });
        }
      })
      .catch(() => {
        if (!cancelled) {
          dispatch({ type: "RECOMMEND_ERROR", message: UI_STRINGS[state.lang].genericError });
        }
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.phase]);

  // User sent a free-text message -> hand the full conversation to chat_turn.
  useEffect(() => {
    if (state.phase !== "loading_chat") return;

    let cancelled = false;
    const conversation = state.messages
      .filter((message): message is TextMessage => message.type === "text")
      .map((message) => ({ role: message.role, content: message.content }));

    postChat({ conversation, prev_recs: state.prevRecs, lang: state.lang })
      .then((response) => {
        if (cancelled) return;
        dispatch({
          type: "CHAT_SUCCESS",
          reply: response.reply,
          cards: response.recommendations,
          explanation: response.explanation,
        });
      })
      .catch(() => {
        if (!cancelled) {
          dispatch({ type: "CHAT_ERROR", message: UI_STRINGS[state.lang].genericError });
        }
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.phase]);

  return { state, dispatch };
}
