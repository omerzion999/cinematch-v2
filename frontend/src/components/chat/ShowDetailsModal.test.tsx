import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ShowDetailsModal } from "./ShowDetailsModal";
import * as api from "@/lib/api";
import type { RecCard, ShowDetails } from "@/lib/types";

vi.mock("@/lib/api");

const show: RecCard = {
  title: "Severance",
  genres: "Drama, Sci-Fi & Fantasy",
  rating: 8.7,
  overview: "A team at Lumon Industries...",
  poster_path: "/abc.jpg",
  decade_str: "2020s",
  num_seasons: 2,
  binge_fit_score: 0.82,
  explanation: "מתאים לך כי אתה אוהב דרמות מתח.",
};

const fullDetails: ShowDetails = {
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
  trailer_url: "https://www.youtube.com/watch?v=abc123",
  trailer_key: "abc123",
  cast: ["Adam Scott", "Britt Lower"],
  watch_providers: ["Apple TV+"],
};

const detailsWithoutExtras: ShowDetails = {
  ...fullDetails,
  trailer_url: null,
  trailer_key: null,
  cast: [],
  watch_providers: [],
};

describe("ShowDetailsModal", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("renders nothing when show is null", () => {
    render(<ShowDetailsModal show={null} lang="he" onClose={vi.fn()} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("shows catalog details immediately and TMDB extras once loaded", async () => {
    vi.mocked(api.getShow).mockResolvedValue(fullDetails);

    render(<ShowDetailsModal show={show} lang="he" onClose={vi.fn()} />);

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Severance")).toBeInTheDocument();
    expect(screen.getByText("A team at Lumon Industries...")).toBeInTheDocument();
    expect(screen.getByText("מתאים לך כי אתה אוהב דרמות מתח.")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("Adam Scott")).toBeInTheDocument();
    });
    expect(screen.getByText("Apple TV+")).toBeInTheDocument();
    expect(screen.getByTitle("Severance trailer")).toHaveAttribute(
      "src",
      "https://www.youtube.com/embed/abc123"
    );
  });

  it("shows only catalog details, with no error, when the lookup fails", async () => {
    vi.mocked(api.getShow).mockRejectedValue(new Error("network error"));

    render(<ShowDetailsModal show={show} lang="he" onClose={vi.fn()} />);

    expect(screen.getByText("Severance")).toBeInTheDocument();

    await waitFor(() => {
      expect(api.getShow).toHaveBeenCalledWith("Severance", "he");
    });

    expect(screen.queryByText("Adam Scott")).not.toBeInTheDocument();
    expect(screen.queryByText(/network error/)).not.toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", async () => {
    vi.mocked(api.getShow).mockResolvedValue(detailsWithoutExtras);
    const onClose = vi.fn();

    render(<ShowDetailsModal show={show} lang="he" onClose={onClose} />);

    await userEvent.click(screen.getByRole("button", { name: "סגור" }));
    expect(onClose).toHaveBeenCalled();
  });
});
