import { describe, it, expect, beforeEach } from "vitest";
import { loadPersistedState, savePersistedState, STORAGE_KEY } from "./storage";
import { createInitialState, chatReducer } from "./chatReducer";

describe("storage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("loadPersistedState returns null when nothing has been saved", () => {
    expect(loadPersistedState()).toBeNull();
  });

  it("savePersistedState then loadPersistedState round-trips the state", () => {
    const state = chatReducer(createInitialState("he"), { type: "START_ONBOARDING" });

    savePersistedState(state);
    const loaded = loadPersistedState();

    expect(loaded).toEqual(state);
  });

  it("loadPersistedState returns null if the stored JSON is corrupted", () => {
    localStorage.setItem(STORAGE_KEY, "{not valid json");

    expect(loadPersistedState()).toBeNull();
  });

  it("loadPersistedState returns null if the stored state has an unrecognized shape", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ foo: "bar" }));

    expect(loadPersistedState()).toBeNull();
  });
});
