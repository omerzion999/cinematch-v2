import { describe, it, expect } from "vitest";
import type {
  OnboardingAnswers,
  RecommendRequest,
  RecommendResponse,
  ShowSummary,
  ChatRequest,
  ChatResponse,
  RecCard,
  ShowDetails,
  ConversationMessage,
  Lang,
} from "./types";

describe("shared types", () => {
  it("OnboardingAnswers / RecommendRequest / RecommendResponse / ShowSummary round-trip", () => {
    const lang: Lang = "en";
    expect(lang).toBe("en");

    const answers: OnboardingAnswers = {
      genre: "drama",
      length: "medium",
      era: "recent",
      popularity: "any",
    };

    const req: RecommendRequest = { answers, lang: "en" };
    expect(req.answers.genre).toBe("drama");
    expect(req.lang).toBe("en");

    const show: ShowSummary = {
      title: "Severance",
      genres: "Drama, Sci-Fi",
      rating: 8.7,
      overview: "A man's work and personal life become dangerously...",
      poster_path: "/abc.jpg",
      decade_str: "2020s",
      num_seasons: 2,
      binge_fit_score: 0.91,
      explanation: "Matches your taste for serious drama.",
    };

    const res: RecommendResponse = {
      intro: "Based on your answers, here's what we found:",
      outro: "Want more like this?",
      cluster_id: 4,
      recommendations: [show],
    };

    expect(res.recommendations[0].title).toBe("Severance");
    expect(res.recommendations[0].binge_fit_score).toBeCloseTo(0.91);
  });

  it("ChatRequest / ChatResponse / RecCard round-trip", () => {
    const conversation: ConversationMessage[] = [
      { role: "user", content: "Show me something funny" },
      { role: "assistant", content: "How about a comedy?" },
    ];

    const prevRec: RecCard = {
      title: "Brooklyn Nine-Nine",
      genres: "Comedy",
      rating: 8.4,
      overview: "A New York police precinct.",
      poster_path: null,
      decade_str: "2010s",
      num_seasons: 8,
    };

    const req: ChatRequest = {
      conversation,
      prev_recs: [prevRec],
      lang: "en",
    };
    expect(req.prev_recs?.[0].title).toBe("Brooklyn Nine-Nine");

    const res: ChatResponse = {
      reply: "Here's another great comedy!",
      recommendations: [prevRec],
      explanation: null,
    };
    expect(res.recommendations?.[0].rating).toBe(8.4);
    expect(res.explanation).toBeNull();
  });

  it("ShowDetails has all expected fields", () => {
    const details: ShowDetails = {
      title: "Severance",
      genres: "Drama, Sci-Fi",
      rating: 8.7,
      overview: "A man's work and personal life become dangerously...",
      poster_path: "/abc.jpg",
      decade_str: "2020s",
      start_year: 2022,
      end_year: null,
      num_seasons: 2,
      num_episodes: 19,
      language: "en",
      votes: 250000,
      popularity: 88.5,
      binge_fit_score: 0.91,
      trailer_url: "https://www.youtube.com/watch?v=xyz",
      trailer_key: "xyz",
      cast: ["Adam Scott", "Britt Lower"],
      watch_providers: ["Apple TV+"],
    };

    expect(details.cast).toContain("Adam Scott");
    expect(details.watch_providers).toContain("Apple TV+");
  });
});
