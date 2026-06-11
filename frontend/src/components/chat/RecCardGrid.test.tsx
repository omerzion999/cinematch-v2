import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RecCardGrid } from "./RecCardGrid";
import type { RecommendationsMessage } from "@/lib/chatReducer";

const message: RecommendationsMessage = {
  id: "1",
  type: "recommendations",
  role: "assistant",
  cards: [
    {
      title: "Severance",
      genres: "Drama, Sci-Fi & Fantasy",
      rating: 8.7,
      overview: "...",
      poster_path: "/abc.jpg",
      decade_str: "2020s",
      num_seasons: 2,
    },
    {
      title: "Barry",
      genres: "Comedy, Crime",
      rating: 8.4,
      overview: "...",
      poster_path: "/barry.jpg",
      decade_str: "2010s",
      num_seasons: 4,
    },
  ],
};

describe("RecCardGrid", () => {
  it("renders one RecCard per card and forwards clicks with the card's title", async () => {
    const onSelectShow = vi.fn();
    render(<RecCardGrid message={message} lang="en" onSelectShow={onSelectShow} />);

    expect(screen.getByText("Severance")).toBeInTheDocument();
    expect(screen.getByText("Barry")).toBeInTheDocument();

    const buttons = screen.getAllByRole("button");
    expect(buttons).toHaveLength(2);

    await userEvent.click(buttons[1]);
    expect(onSelectShow).toHaveBeenCalledWith("Barry");
  });
});
