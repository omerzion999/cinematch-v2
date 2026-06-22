import { describe, it, expect } from "vitest";
import { ONBOARDING_QUESTIONS, resolveTypedAnswer } from "./onboarding";
import type { OnboardingAnswers } from "./types";

describe("ONBOARDING_QUESTIONS", () => {
  it("has exactly 3 questions, in the order genre, era, popularity", () => {
    expect(ONBOARDING_QUESTIONS.map((q) => q.id)).toEqual([
      "genre",
      "era",
      "popularity",
    ]);
  });

  it("every question has a Hebrew and English prompt", () => {
    for (const question of ONBOARDING_QUESTIONS) {
      expect(question.prompt.he.length).toBeGreaterThan(0);
      expect(question.prompt.en.length).toBeGreaterThan(0);
    }
  });

  it("option values for each question match the OnboardingAnswers vocabulary expected by the backend", () => {
    const byId = Object.fromEntries(
      ONBOARDING_QUESTIONS.map((q) => [q.id, q.options.map((o) => o.value)])
    );

    const expected: { [K in keyof OnboardingAnswers]: OnboardingAnswers[K][] } = {
      genre: [
        "drama",
        "comedy",
        "crime",
        "scifi_fantasy",
        "action_adventure",
        "animation",
        "any",
      ],
      era: ["recent", "modern", "classic", "any"],
      popularity: ["well_known", "hidden_gem", "any"],
    };

    for (const [id, values] of Object.entries(expected)) {
      expect(byId[id]).toEqual(values);
    }
  });

  it("every option has a Hebrew and English label", () => {
    for (const question of ONBOARDING_QUESTIONS) {
      for (const option of question.options) {
        expect(option.label.he.length).toBeGreaterThan(0);
        expect(option.label.en.length).toBeGreaterThan(0);
      }
    }
  });
});

describe("resolveTypedAnswer", () => {
  const genreQ = ONBOARDING_QUESTIONS.find((q) => q.id === "genre")!;
  const eraQ = ONBOARDING_QUESTIONS.find((q) => q.id === "era")!;
  const popularityQ = ONBOARDING_QUESTIONS.find((q) => q.id === "popularity")!;

  it("matches an exact option value", () => {
    expect(resolveTypedAnswer(genreQ, "comedy", "en")).toBe("comedy");
  });

  it("matches an English label case-insensitively", () => {
    expect(resolveTypedAnswer(genreQ, "Sci-Fi & Fantasy", "en")).toBe("scifi_fantasy");
  });

  it("matches a Hebrew synonym", () => {
    expect(resolveTypedAnswer(genreQ, "מצחיק", "he")).toBe("comedy");
  });

  it("maps English free text to era and popularity", () => {
    expect(resolveTypedAnswer(eraQ, "the newest shows", "en")).toBe("recent");
    expect(resolveTypedAnswer(eraQ, "modern stuff", "en")).toBe("modern");
    expect(resolveTypedAnswer(popularityQ, "a hidden gem please", "en")).toBe("hidden_gem");
  });

  it("returns null when nothing matches", () => {
    expect(resolveTypedAnswer(genreQ, "qwerty zzz", "en")).toBeNull();
    expect(resolveTypedAnswer(genreQ, "", "en")).toBeNull();
  });
});
