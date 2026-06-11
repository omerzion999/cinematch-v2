import { describe, it, expect } from "vitest";
import { UI_STRINGS } from "./i18n";

describe("UI_STRINGS", () => {
  it("has identical key sets for he and en", () => {
    const heKeys = Object.keys(UI_STRINGS.he).sort();
    const enKeys = Object.keys(UI_STRINGS.en).sort();
    expect(heKeys).toEqual(enKeys);
  });

  it("every string is non-empty", () => {
    for (const lang of ["he", "en"] as const) {
      for (const [key, value] of Object.entries(UI_STRINGS[lang])) {
        expect(value.length, `${lang}.${key}`).toBeGreaterThan(0);
      }
    }
  });
});
