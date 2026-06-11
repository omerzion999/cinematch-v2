import "@testing-library/jest-dom/vitest";

// jsdom does not implement ResizeObserver, which Radix UI's ScrollArea relies on.
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}
