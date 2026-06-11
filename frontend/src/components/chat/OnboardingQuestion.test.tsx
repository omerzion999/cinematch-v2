import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { OnboardingQuestion } from "./OnboardingQuestion";
import type { ChoiceMessage } from "@/lib/chatReducer";

const unanswered: ChoiceMessage = {
  id: "1",
  type: "choice",
  role: "assistant",
  prompt: "איזה ז'אנר הכי מדבר אליך?",
  options: [
    { value: "drama", label: "דרמה" },
    { value: "comedy", label: "קומדיה" },
  ],
};

const answered: ChoiceMessage = {
  ...unanswered,
  selectedValue: "drama",
  selectedLabel: "דרמה",
};

describe("OnboardingQuestion", () => {
  it("renders the prompt and one button per option, and calls onSelect when clicked", async () => {
    const onSelect = vi.fn();
    render(<OnboardingQuestion message={unanswered} lang="he" onSelect={onSelect} />);

    expect(screen.getByText("איזה ז'אנר הכי מדבר אליך?")).toBeInTheDocument();
    const dramaButton = screen.getByRole("button", { name: "דרמה" });
    expect(screen.getByRole("button", { name: "קומדיה" })).toBeInTheDocument();

    await userEvent.click(dramaButton);
    expect(onSelect).toHaveBeenCalledWith("drama");
  });

  it("renders only the selected label and no buttons once answered", () => {
    const onSelect = vi.fn();
    render(<OnboardingQuestion message={answered} lang="he" onSelect={onSelect} />);

    expect(screen.getByText("דרמה")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "קומדיה" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});
