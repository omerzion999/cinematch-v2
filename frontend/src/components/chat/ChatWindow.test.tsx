import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChatWindow } from "./ChatWindow";
import * as api from "@/lib/api";
import { ONBOARDING_QUESTIONS } from "@/lib/onboarding";
import type { RecommendResponse, ShowDetails } from "@/lib/types";

vi.mock("@/lib/api");

const SAMPLE_RECOMMENDATION: RecommendResponse = {
  intro: "אתה בטעם של: דרמות מתח",
  outro: "מקווה שאהבת!",
  cluster_id: 2,
  recommendations: [
    {
      title: "Severance",
      genres: "Drama, Sci-Fi & Fantasy",
      rating: 8.7,
      overview: "A team at Lumon Industries...",
      poster_path: "/abc.jpg",
      decade_str: "2020s",
      num_seasons: 2,
      binge_fit_score: 0.82,
      explanation: "מתאים לך כי...",
    },
  ],
};

const SAMPLE_SHOW_DETAILS: ShowDetails = {
  title: "Severance",
  genres: "Drama, Sci-Fi & Fantasy",
  rating: 8.7,
  overview: "A team at Lumon Industries...",
  poster_path: "/abc.jpg",
  decade_str: "2020s",
  start_year: 2022,
  end_year: null,
  num_seasons: 2,
  num_episodes: 19,
  language: "en",
  votes: 12000,
  popularity: 95.3,
  binge_fit_score: 0.82,
  trailer_url: null,
  cast: [],
  watch_providers: [],
};

async function completeOnboarding() {
  await userEvent.click(screen.getByRole("button", { name: "כן, בוא נתחיל" }));
  for (const question of ONBOARDING_QUESTIONS) {
    const anyOption = question.options.find((o) => o.value === "any")!;
    const button = await screen.findByRole("button", { name: anyOption.label.he });
    await userEvent.click(button);
  }
}

describe("ChatWindow", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.resetAllMocks();
  });

  it("shows the opening message with start/skip buttons", () => {
    render(<ChatWindow />);

    expect(screen.getByRole("button", { name: "כן, בוא נתחיל" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "לא תודה, אני רוצה להתקדם לצ'אט הרגיל" })
    ).toBeInTheDocument();
  });

  it("walks through onboarding and displays the recommendations", async () => {
    vi.mocked(api.postRecommend).mockResolvedValue(SAMPLE_RECOMMENDATION);

    render(<ChatWindow />);
    await completeOnboarding();

    await waitFor(() => {
      expect(screen.getByText("Severance")).toBeInTheDocument();
    });
    expect(screen.getByText("אתה בטעם של: דרמות מתח")).toBeInTheDocument();
    expect(screen.getByText("מקווה שאהבת!")).toBeInTheDocument();
  });

  it("skipping onboarding enables the chat input and a sent message gets a reply", async () => {
    vi.mocked(api.postChat).mockResolvedValue({
      reply: "בטח, הנה כמה הצעות:",
      recommendations: null,
      explanation: null,
    });

    render(<ChatWindow />);

    await userEvent.click(
      screen.getByRole("button", { name: "לא תודה, אני רוצה להתקדם לצ'אט הרגיל" })
    );

    const input = screen.getByPlaceholderText("כתוב הודעה...");
    expect(input).toBeEnabled();

    await userEvent.type(input, "תמליץ לי על קומדיה");
    await userEvent.click(screen.getByRole("button", { name: "שלח" }));

    await waitFor(() => {
      expect(screen.getByText("בטח, הנה כמה הצעות:")).toBeInTheDocument();
    });
  });

  it("clicking a recommendation card opens the details modal", async () => {
    vi.mocked(api.postRecommend).mockResolvedValue(SAMPLE_RECOMMENDATION);
    vi.mocked(api.getShow).mockResolvedValue(SAMPLE_SHOW_DETAILS);

    render(<ChatWindow />);
    await completeOnboarding();

    await waitFor(() => {
      expect(screen.getByText("Severance")).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("img", { name: "Severance" }));

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
  });

  it("toggles the UI language when the language button is clicked", async () => {
    render(<ChatWindow />);

    await userEvent.click(screen.getByRole("button", { name: "English" }));

    expect(screen.getByPlaceholderText("Type a message...")).toBeInTheDocument();
  });
});
