import type { ChatState } from "./chatReducer";

export const STORAGE_KEY = "cinematch:chat-state";

function isChatState(value: unknown): value is ChatState {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.phase === "string" &&
    typeof candidate.lang === "string" &&
    typeof candidate.onboardingStepIndex === "number" &&
    typeof candidate.onboardingAnswers === "object" &&
    candidate.onboardingAnswers !== null &&
    Array.isArray(candidate.messages) &&
    (candidate.prevRecs === null || Array.isArray(candidate.prevRecs))
  );
}

export function loadPersistedState(): ChatState | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (raw === null) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }

  return isChatState(parsed) ? parsed : null;
}

export function savePersistedState(state: ChatState): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}
