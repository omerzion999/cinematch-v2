import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Button } from "./button";
import { Card, CardContent } from "./card";
import { Badge } from "./badge";
import { Input } from "./input";

describe("shadcn/ui primitives", () => {
  it("Button renders its children and responds to the variant prop", () => {
    render(<Button variant="outline">Click me</Button>);
    expect(screen.getByRole("button", { name: "Click me" })).toBeInTheDocument();
  });

  it("Card renders nested content", () => {
    render(
      <Card>
        <CardContent>Hello card</CardContent>
      </Card>
    );
    expect(screen.getByText("Hello card")).toBeInTheDocument();
  });

  it("Badge renders its children", () => {
    render(<Badge>Drama</Badge>);
    expect(screen.getByText("Drama")).toBeInTheDocument();
  });

  it("Input accepts a placeholder and value", () => {
    render(<Input placeholder="Type a message..." readOnly value="hi" />);
    expect(screen.getByPlaceholderText("Type a message...")).toHaveValue("hi");
  });
});
