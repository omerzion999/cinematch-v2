import { describe, it, expect } from "vitest";
import { ONBOARDING_QUESTIONS } from "./onboarding";
import type { OnboardingAnswers } from "./types";

describe("ONBOARDING_QUESTIONS", () => {
  it("has exactly 5 questions, in the order genre, length, era, tone, popularity", () => {
    expect(ONBOARDING_QUESTIONS.map((q) => q.id)).toEqual([
      "genre",
      "length",
      "era",
      "tone",
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
        "action_adventure",
        "scifi_fantasy",
        "crime",
        "animation",
        "romance",
        "any",
      ],
      length: ["short", "medium", "long", "any"],
      era: ["recent", "classic", "any"],
      tone: ["light_fun", "serious_drama", "thriller_action", "any"],
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
