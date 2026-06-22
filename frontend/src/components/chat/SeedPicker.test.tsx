import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SeedPicker } from "./SeedPicker";
import type { SeedPickMessage } from "@/lib/chatReducer";
import type { RecCard } from "@/lib/types";
import { UI_STRINGS } from "@/lib/i18n";

const card = (title: string): RecCard => ({
  title,
  genres: "Crime, Drama",
  rating: 9.0,
  overview: "...",
  poster_path: "/x.jpg",
  decade_str: "2010s",
  num_seasons: 3,
});

function message(partial: Partial<SeedPickMessage> = {}): SeedPickMessage {
  return {
    id: "s1",
    type: "seedpick",
    role: "assistant",
    cards: [card("Breaking Bad"), card("Narcos"), card("Sherlock")],
    selectedTitles: [],
    done: false,
    ...partial,
  };
}

describe("SeedPicker", () => {
  it("shows a loading line while cards are null", () => {
    render(
      <SeedPicker message={message({ cards: null })} lang="en" onToggle={vi.fn()} onConfirm={vi.fn()} onSkip={vi.fn()} />
    );
    expect(screen.getByText(UI_STRINGS.en.seedLoading)).toBeInTheDocument();
  });

  it("renders the prompt and a selectable card per seed, toggling on click", async () => {
    const onToggle = vi.fn();
    render(
      <SeedPicker message={message()} lang="en" onToggle={onToggle} onConfirm={vi.fn()} onSkip={vi.fn()} />
    );
    expect(screen.getByText(UI_STRINGS.en.seedPrompt)).toBeInTheDocument();

    await userEvent.click(screen.getByText("Narcos"));
    expect(onToggle).toHaveBeenCalledWith("Narcos");
  });

  it("disables Continue until at least one seed is picked, and fires confirm/skip", async () => {
    const onConfirm = vi.fn();
    const onSkip = vi.fn();
    const { rerender } = render(
      <SeedPicker message={message()} lang="en" onToggle={vi.fn()} onConfirm={onConfirm} onSkip={onSkip} />
    );

    const continueBtn = screen.getByRole("button", { name: UI_STRINGS.en.seedContinue });
    expect(continueBtn).toBeDisabled();

    await userEvent.click(screen.getByRole("button", { name: UI_STRINGS.en.seedSkip }));
    expect(onSkip).toHaveBeenCalled();

    rerender(
      <SeedPicker
        message={message({ selectedTitles: ["Breaking Bad"] })}
        lang="en"
        onToggle={vi.fn()}
        onConfirm={onConfirm}
        onSkip={onSkip}
      />
    );
    const enabledContinue = screen.getByRole("button", { name: UI_STRINGS.en.seedContinue });
    expect(enabledContinue).toBeEnabled();
    await userEvent.click(enabledContinue);
    expect(onConfirm).toHaveBeenCalled();
  });

  it("hides the action buttons once the step is done", () => {
    render(
      <SeedPicker
        message={message({ selectedTitles: ["Breaking Bad"], done: true })}
        lang="en"
        onToggle={vi.fn()}
        onConfirm={vi.fn()}
        onSkip={vi.fn()}
      />
    );
    expect(screen.queryByRole("button", { name: UI_STRINGS.en.seedContinue })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: UI_STRINGS.en.seedSkip })).not.toBeInTheDocument();
  });
});
