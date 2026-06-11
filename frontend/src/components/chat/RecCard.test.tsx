import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RecCard } from "./RecCard";
import type { RecCard as RecCardData } from "@/lib/types";

const showWithPoster: RecCardData = {
  title: "Severance",
  genres: "Drama, Sci-Fi & Fantasy",
  rating: 8.7,
  overview: "A team at Lumon Industries...",
  poster_path: "/abc.jpg",
  decade_str: "2020s",
  num_seasons: 2,
};

const showWithoutPoster: RecCardData = {
  ...showWithPoster,
  title: "Some Obscure Show",
  poster_path: null,
};

describe("RecCard", () => {
  it("renders the poster, title, genres, and rating, and calls onClick with the title", async () => {
    const onClick = vi.fn();
    render(<RecCard show={showWithPoster} lang="en" onClick={onClick} />);

    const img = screen.getByRole("img", { name: "Severance" });
    expect(img).toHaveAttribute("src", "https://image.tmdb.org/t/p/w342/abc.jpg");
    expect(screen.getByText("Severance")).toBeInTheDocument();
    expect(screen.getByText("Drama, Sci-Fi & Fantasy")).toBeInTheDocument();
    expect(screen.getByText(/8\.7/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalledWith("Severance");
  });

  it("renders a text placeholder instead of an image when poster_path is null", () => {
    render(<RecCard show={showWithoutPoster} lang="en" onClick={vi.fn()} />);

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getAllByText("Some Obscure Show").length).toBeGreaterThan(0);
  });
});
