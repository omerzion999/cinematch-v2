import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MessageBubble } from "./MessageBubble";
import type { TextMessage } from "@/lib/chatReducer";

const userMessage: TextMessage = {
  id: "1",
  type: "text",
  role: "user",
  content: "תמליץ לי על קומדיה",
};

const assistantMessage: TextMessage = {
  id: "2",
  type: "text",
  role: "assistant",
  content: "Here are some comedies you might like.",
};

describe("MessageBubble", () => {
  it("renders a user message left-aligned with rtl text direction for Hebrew", () => {
    render(<MessageBubble message={userMessage} lang="he" />);

    const bubble = screen.getByText("תמליץ לי על קומדיה");
    expect(bubble).toHaveAttribute("dir", "rtl");
    expect(bubble.parentElement).toHaveClass("justify-start");
  });

  it("renders an assistant message right-aligned for Hebrew", () => {
    render(<MessageBubble message={assistantMessage} lang="he" />);

    const bubble = screen.getByText("Here are some comedies you might like.");
    expect(bubble.parentElement).toHaveClass("justify-end");
  });

  it("renders an assistant message left-aligned with ltr direction for English", () => {
    render(<MessageBubble message={assistantMessage} lang="en" />);

    const bubble = screen.getByText("Here are some comedies you might like.");
    expect(bubble).toHaveAttribute("dir", "ltr");
    expect(bubble.parentElement).toHaveClass("justify-start");
  });
});
