# CineMatch AI v2 - Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the React/TypeScript chat UI for CineMatch AI v2 - a WhatsApp-style single-page app that walks the user through 5 onboarding questions, shows 3 recommended shows with a details popup, and then continues as an open chat - calling the 3 backend endpoints from `docs/superpowers/plans/2026-06-10-cinematch-v2-backend.md` (`/api/recommend`, `/api/chat`, `/api/show/{title}`).

**Architecture:** Vite + React 18 + TypeScript, Tailwind CSS + shadcn/ui primitives. A single reducer (`lib/chatReducer.ts`) drives the whole conversation as a sequence of typed `ChatMessage` items (text bubbles, an onboarding-question bubble, a recommendation-cards bubble). `hooks/useChatState.ts` wraps the reducer, calls the backend API client (`lib/api.ts`), and persists state to `localStorage` (`lib/storage.ts`) per the spec's "no server-side chat storage" requirement. `ChatWindow` renders the message list + input box + language toggle; `ShowDetailsModal` lazily fetches `/api/show/{title}` only when a card is clicked. In production, this builds to `frontend/dist`, which `backend/app/main.py` (Task 19 of the backend plan) serves as static files - no CORS, one Render service. In development, Vite's dev server proxies `/api` to a local `uvicorn` process.

**Tech Stack:** React 18, TypeScript 5, Vite 6, Tailwind CSS 3, shadcn/ui (Radix primitives), Vitest + React Testing Library for unit tests, manual browser verification for the full chat flow (per project guidance: UI features must be exercised in a real browser before being called done).

---

## File Structure

```
cinematch-ai-v2/                          (repo root)
├── backend/                              (see backend implementation plan)
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── components.json                  # shadcn/ui config
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css                    # Tailwind directives + base styles
│       ├── test/
│       │   └── setup.ts                 # Vitest + jest-dom setup
│       ├── lib/
│       │   ├── utils.ts                 # shadcn's cn() helper
│       │   ├── types.ts                 # TS types matching backend Pydantic models
│       │   ├── onboarding.ts            # 5 onboarding questions + bilingual i18n strings
│       │   ├── api.ts                   # fetch wrappers for /api/recommend, /api/chat, /api/show
│       │   ├── chatReducer.ts           # core conversation state machine
│       │   └── storage.ts               # localStorage persistence
│       ├── hooks/
│       │   └── useChatState.ts          # wraps reducer + api + storage
│       └── components/
│           ├── ui/                      # shadcn primitives: button, card, dialog, scroll-area, badge, input
│           └── chat/
│               ├── MessageBubble.tsx
│               ├── OnboardingQuestion.tsx
│               ├── RecCard.tsx
│               ├── RecCardGrid.tsx
│               ├── ShowDetailsModal.tsx
│               └── ChatWindow.tsx
└── render.yaml                           (see backend implementation plan, Task 19)
```

**Onboarding answer vocabulary** (must match `app/clustering/onboarding_map.py` from the backend plan exactly):

```typescript
genre:      "drama" | "comedy" | "action_adventure" | "scifi_fantasy" | "crime" | "animation" | "romance" | "any"
length:     "short" | "medium" | "long" | "any"
era:        "recent" | "classic" | "any"
tone:       "light_fun" | "serious_drama" | "thriller_action" | "any"
popularity: "well_known" | "hidden_gem" | "any"
```

---

## Task 1: Project Scaffolding (Vite + React + TS + Tailwind + Vitest)

**Files:**
- Create: `frontend/` (via Vite scaffold)
- Modify: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tsconfig.node.json`
- Create: `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/src/index.css`, `frontend/src/test/setup.ts`
- Create: `frontend/src/lib/utils.ts`, `frontend/components.json`

- [ ] **Step 1: Scaffold the Vite project**

Run (from the repo root):
```powershell
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```
Expected: `frontend/` now contains `package.json`, `index.html`, `src/main.tsx`, `src/App.tsx`, `vite.config.ts`, `tsconfig.json`, `tsconfig.node.json`, etc.

- [ ] **Step 2: Install Tailwind, shadcn/ui dependencies, and test tooling**

Run (from `frontend/`):
```powershell
npm install -D tailwindcss@3.4.13 postcss@8.4.47 autoprefixer@10.4.20 tailwindcss-animate@1.0.7
npm install -D vitest@2.1.4 jsdom@25.0.1 @testing-library/react@16.0.1 @testing-library/jest-dom@6.5.0 @testing-library/user-event@14.5.2
npm install class-variance-authority@0.7.0 clsx@2.1.1 tailwind-merge@2.5.4 lucide-react@0.460.0 @radix-ui/react-dialog@1.1.2
npx tailwindcss init -p
```
Expected: `tailwind.config.js` and `postcss.config.js` are created; `package.json`'s
`dependencies`/`devDependencies` now include the packages above.

- [ ] **Step 3: Configure path alias (`@/*` -> `src/*`)**

Edit `frontend/tsconfig.json` - add `compilerOptions.baseUrl` and `paths`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

Edit `frontend/tsconfig.node.json` - add the same `baseUrl`/`paths` so editor
tooling resolves `@/*` inside `vite.config.ts` too:

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 4: Configure Vite (path alias, dev proxy, Vitest)**

Replace `frontend/vite.config.ts` with:

```typescript
import path from "path";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
  },
});
```

`server.proxy` only matters for `npm run dev` (so the Vite dev server forwards
`/api/*` to a locally-running `uvicorn app.main:app --reload` on port 8000).
The production build (`npm run build` -> `frontend/dist`) is served directly
by FastAPI, same-origin, so no proxy/CORS is needed there.

- [ ] **Step 5: Configure Tailwind**

Replace `frontend/tailwind.config.js` with:

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
```

Replace `frontend/src/index.css` with:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222 47% 11%;
    --card: 0 0% 100%;
    --card-foreground: 222 47% 11%;
    --primary: 142 71% 45%;
    --primary-foreground: 0 0% 100%;
    --secondary: 210 40% 96%;
    --secondary-foreground: 222 47% 11%;
    --muted: 210 40% 96%;
    --muted-foreground: 215 16% 47%;
    --border: 214 32% 91%;
    --input: 214 32% 91%;
    --ring: 142 71% 45%;
    --radius: 0.75rem;
  }

  * {
    @apply border-border;
  }

  body {
    @apply bg-secondary text-foreground;
  }
}
```

- [ ] **Step 6: Add the shadcn `cn()` helper and `components.json`**

Create `frontend/src/lib/utils.ts`:

```typescript
import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

Create `frontend/components.json` (so `npx shadcn@latest add <component>` in
Task 7 knows where to put generated files, without running the interactive
`init` wizard):

```json
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "default",
  "rsc": false,
  "tsx": true,
  "tailwind": {
    "config": "tailwind.config.js",
    "css": "src/index.css",
    "baseColor": "slate",
    "cssVariables": true,
    "prefix": ""
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils",
    "ui": "@/components/ui"
  }
}
```

- [ ] **Step 7: Add Vitest setup file**

Create `frontend/src/test/setup.ts`:

```typescript
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 8: Add a test script and run a smoke test**

Edit `frontend/package.json` - add to `"scripts"`:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "lint": "eslint .",
    "preview": "vite preview",
    "test": "vitest run"
  }
}
```

Create `frontend/src/lib/utils.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import { cn } from "./utils";

describe("cn", () => {
  it("merges class names and resolves Tailwind conflicts", () => {
    expect(cn("px-2", "px-4")).toBe("px-4");
    expect(cn("text-sm", "font-bold")).toBe("text-sm font-bold");
  });
});
```

Run: `cd frontend && npm test`
Expected: PASS (1 test)

- [ ] **Step 9: Commit**

```bash
git add frontend
git commit -m "Scaffold frontend project (Vite, React, TS, Tailwind, shadcn config, Vitest)"
```

---

## Task 2: Shared Types (`lib/types.ts`)

**Files:**
- Create: `frontend/src/lib/types.ts`
- Test: `frontend/src/lib/types.test.ts`

These types mirror the Pydantic models from the backend plan exactly:
`OnboardingAnswers`/`RecommendRequest`/`RecommendResponse`/`ShowSummary` (Task
16), `ChatMessage`/`RecCard`/`ChatRequest`/`ChatResponse` (Task 17), and
`ShowDetails` (Task 18).

- [ ] **Step 1: Write the failing test**

A pure types file has no runtime behavior, but we still write one test that
exercises the types end-to-end (construct a value of each shape and assert
its fields) - this catches typos in field names and gives the engineer a
concrete usage example. Create `frontend/src/lib/types.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import type {
  OnboardingAnswers,
  RecommendRequest,
  RecommendResponse,
  ShowSummary,
  RecCard,
  ConversationMessage,
  ChatRequest,
  ChatResponse,
  ShowDetails,
} from "./types";

describe("types", () => {
  it("OnboardingAnswers/RecommendRequest/RecommendResponse round-trip the /api/recommend shape", () => {
    const answers: OnboardingAnswers = {
      genre: "drama",
      length: "any",
      era: "recent",
      tone: "serious_drama",
      popularity: "any",
    };
    const request: RecommendRequest = { answers, lang: "he" };

    const pick: ShowSummary = {
      title: "Severance",
      genres: "Drama, Sci-Fi & Fantasy",
      rating: 8.7,
      overview: "A team at Lumon Industries...",
      poster_path: "/abc.jpg",
      decade_str: "2020s",
      num_seasons: 2,
      binge_fit_score: 0.82,
      explanation: "מתאים לך כי...",
    };
    const response: RecommendResponse = {
      intro: "אתה בטעם של: דרמות מתח",
      outro: "מקווה שאהבת!",
      cluster_id: 3,
      recommendations: [pick],
    };

    expect(request.answers.genre).toBe("drama");
    expect(response.recommendations[0].title).toBe("Severance");
  });

  it("ChatRequest/ChatResponse/RecCard round-trip the /api/chat shape", () => {
    const conversation: ConversationMessage[] = [
      { role: "user", content: "משהו עם הומור שחור" },
    ];
    const card: RecCard = {
      title: "Barry",
      genres: "Comedy, Crime",
      rating: 8.4,
      overview: "A hitman takes an acting class.",
      poster_path: "/barry.jpg",
      decade_str: "2010s",
      num_seasons: 4,
    };
    const request: ChatRequest = { conversation, prev_recs: null, lang: "he" };
    const response: ChatResponse = {
      reply: "הנה כמה הצעות:",
      recommendations: [card],
      explanation: "שתי הסדרות משלבות הומור שחור...",
    };

    expect(request.conversation[0].role).toBe("user");
    expect(response.recommendations?.[0].title).toBe("Barry");
  });

  it("ShowDetails covers the /api/show/{title} shape", () => {
    const details: ShowDetails = {
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
      cast: ["Adam Scott", "Britt Lower"],
      watch_providers: ["Apple TV+"],
    };

    expect(details.cast).toHaveLength(2);
    expect(details.trailer_url).toContain("youtube.com");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './types'` (or similar - the file doesn't exist yet)

- [ ] **Step 3: Implement `lib/types.ts`**

Create `frontend/src/lib/types.ts`:

```typescript
export type Lang = "he" | "en";

export interface OnboardingAnswers {
  genre:
    | "drama"
    | "comedy"
    | "action_adventure"
    | "scifi_fantasy"
    | "crime"
    | "animation"
    | "romance"
    | "any";
  length: "short" | "medium" | "long" | "any";
  era: "recent" | "classic" | "any";
  tone: "light_fun" | "serious_drama" | "thriller_action" | "any";
  popularity: "well_known" | "hidden_gem" | "any";
}

export interface RecommendRequest {
  answers: OnboardingAnswers;
  lang: Lang;
}

/** A recommendation card as returned by /api/chat (no per-card explanation). */
export interface RecCard {
  title: string;
  genres: string;
  rating: number;
  overview: string;
  poster_path: string | null;
  decade_str: string;
  num_seasons: number | null;
  binge_fit_score?: number;
  explanation?: string;
}

/** A recommendation card as returned by /api/recommend (always has its own explanation). */
export interface ShowSummary extends RecCard {
  binge_fit_score: number;
  explanation: string;
}

export interface RecommendResponse {
  intro: string;
  outro: string;
  cluster_id: number;
  recommendations: ShowSummary[];
}

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  conversation: ConversationMessage[];
  prev_recs: RecCard[] | null;
  lang: Lang;
}

export interface ChatResponse {
  reply: string;
  recommendations: RecCard[] | null;
  explanation: string | null;
}

export interface ShowDetails {
  title: string;
  genres: string;
  rating: number;
  overview: string;
  poster_path: string | null;
  decade_str: string;
  start_year: number | null;
  end_year: number | null;
  num_seasons: number | null;
  num_episodes: number | null;
  language: string | null;
  votes: number | null;
  popularity: number | null;
  binge_fit_score: number;
  trailer_url: string | null;
  cast: string[];
  watch_providers: string[];
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (3 new tests, 4 total)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/types.ts frontend/src/lib/types.test.ts
git commit -m "Add shared TypeScript types matching backend API contracts"
```

---

## Task 3: Onboarding Questions & UI Strings (`lib/onboarding.ts`, `lib/i18n.ts`)

**Files:**
- Create: `frontend/src/lib/onboarding.ts`
- Create: `frontend/src/lib/i18n.ts`
- Test: `frontend/src/lib/onboarding.test.ts`
- Test: `frontend/src/lib/i18n.test.ts`

`onboarding.ts` defines the 5 fixed onboarding questions (text + button
options, in Hebrew and English) shown one-at-a-time as bot bubbles - this is
the "deterministic, fixed buttons, no LLM" UI described in the spec's UX step
2. `i18n.ts` holds the small set of static UI strings the frontend owns
(opening message, button labels, input placeholder, etc.) - this is separate
from (and much smaller than) the backend's `app/i18n.py`, which generates the
recommendation intro/outro text.

- [ ] **Step 1: Write the failing test for `onboarding.ts`**

Create `frontend/src/lib/onboarding.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import { ONBOARDING_QUESTIONS } from "./onboarding";
import type { OnboardingAnswers } from "./types";

describe("ONBOARDING_QUESTIONS", () => {
  it("has exactly 5 questions, in the order genre, length, era, tone, popularity", () => {
    expect(ONBOARDING_QUESTIONS.map((q) => q.id)).toEqual([
      "genre",
      "length",
      "era",
      "tone",
      "popularity",
    ]);
  });

  it("every question has a Hebrew and English prompt", () => {
    for (const question of ONBOARDING_QUESTIONS) {
      expect(question.prompt.he.length).toBeGreaterThan(0);
      expect(question.prompt.en.length).toBeGreaterThan(0);
    }
  });

  it("option values for each question match the OnboardingAnswers vocabulary expected by the backend", () => {
    const byId = Object.fromEntries(
      ONBOARDING_QUESTIONS.map((q) => [q.id, q.options.map((o) => o.value)])
    );

    const expected: { [K in keyof OnboardingAnswers]: OnboardingAnswers[K][] } = {
      genre: [
        "drama",
        "comedy",
        "action_adventure",
        "scifi_fantasy",
        "crime",
        "animation",
        "romance",
        "any",
      ],
      length: ["short", "medium", "long", "any"],
      era: ["recent", "classic", "any"],
      tone: ["light_fun", "serious_drama", "thriller_action", "any"],
      popularity: ["well_known", "hidden_gem", "any"],
    };

    for (const [id, values] of Object.entries(expected)) {
      expect(byId[id]).toEqual(values);
    }
  });

  it("every option has a Hebrew and English label", () => {
    for (const question of ONBOARDING_QUESTIONS) {
      for (const option of question.options) {
        expect(option.label.he.length).toBeGreaterThan(0);
        expect(option.label.en.length).toBeGreaterThan(0);
      }
    }
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './onboarding'`

- [ ] **Step 3: Implement `lib/onboarding.ts`**

Create `frontend/src/lib/onboarding.ts`:

```typescript
import type { OnboardingAnswers } from "./types";

export interface OnboardingOption {
  value: string;
  label: { he: string; en: string };
}

export interface OnboardingQuestion {
  id: keyof OnboardingAnswers;
  prompt: { he: string; en: string };
  options: OnboardingOption[];
}

export const ONBOARDING_QUESTIONS: OnboardingQuestion[] = [
  {
    id: "genre",
    prompt: {
      he: "איזה ז'אנר הכי מדבר אליך?",
      en: "Which genre speaks to you the most?",
    },
    options: [
      { value: "drama", label: { he: "דרמה", en: "Drama" } },
      { value: "comedy", label: { he: "קומדיה", en: "Comedy" } },
      {
        value: "action_adventure",
        label: { he: "אקשן והרפתקאות", en: "Action & Adventure" },
      },
      {
        value: "scifi_fantasy",
        label: { he: "מד\"ב ופנטזיה", en: "Sci-Fi & Fantasy" },
      },
      { value: "crime", label: { he: "פשע", en: "Crime" } },
      { value: "animation", label: { he: "אנימציה", en: "Animation" } },
      { value: "romance", label: { he: "רומנטיקה", en: "Romance" } },
      { value: "any", label: { he: "לא משנה", en: "Doesn't matter" } },
    ],
  },
  {
    id: "length",
    prompt: {
      he: "כמה עונות אתה מעדיף?",
      en: "How many seasons do you prefer?",
    },
    options: [
      {
        value: "short",
        label: { he: "קצר / מיני (1-2 עונות)", en: "Short / mini-series (1-2 seasons)" },
      },
      {
        value: "medium",
        label: { he: "בינוני (3-5 עונות)", en: "Medium (3-5 seasons)" },
      },
      { value: "long", label: { he: "ארוך (6+ עונות)", en: "Long (6+ seasons)" } },
      { value: "any", label: { he: "לא משנה", en: "Doesn't matter" } },
    ],
  },
  {
    id: "era",
    prompt: {
      he: "איזה עידן מעניין אותך?",
      en: "Which era interests you?",
    },
    options: [
      { value: "recent", label: { he: "חדש (2020 ואילך)", en: "Recent (2020+)" } },
      { value: "classic", label: { he: "קלאסי", en: "Classic" } },
      { value: "any", label: { he: "לא משנה", en: "Doesn't matter" } },
    ],
  },
  {
    id: "tone",
    prompt: {
      he: "איזה טון אתה מחפש?",
      en: "What tone are you in the mood for?",
    },
    options: [
      { value: "light_fun", label: { he: "קליל ומשעשע", en: "Light & fun" } },
      {
        value: "serious_drama",
        label: { he: "רציני ודרמטי", en: "Serious & dramatic" },
      },
      {
        value: "thriller_action",
        label: { he: "מותח ואקשן", en: "Thriller & action" },
      },
      { value: "any", label: { he: "לא משנה", en: "Doesn't matter" } },
    ],
  },
  {
    id: "popularity",
    prompt: {
      he: "מה אתה מעדיף?",
      en: "What do you prefer?",
    },
    options: [
      {
        value: "well_known",
        label: { he: "להיטים מוכרים", en: "Well-known hits" },
      },
      {
        value: "hidden_gem",
        label: { he: "פנינים נסתרות", en: "Hidden gems" },
      },
      { value: "any", label: { he: "לא משנה", en: "Doesn't matter" } },
    ],
  },
];
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (4 new tests, 8 total)

- [ ] **Step 5: Write the failing test for `i18n.ts`**

Create `frontend/src/lib/i18n.test.ts`:

```typescript
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
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './i18n'`

- [ ] **Step 7: Implement `lib/i18n.ts`**

Create `frontend/src/lib/i18n.ts`:

```typescript
import type { Lang } from "./types";

export const UI_STRINGS: Record<Lang, Record<string, string>> = {
  he: {
    openingMessage:
      "היי! אני CineMatch 🎬 - הבוט שעוזר למצוא את הסדרה הבאה שתאהב. רוצה לענות על כמה שאלות קצרות כדי שאכיר את הטעם שלך?",
    startOnboarding: "כן, בוא נתחיל",
    skipToChat: "לא תודה, אני רוצה להתקדם לצ'אט הרגיל",
    skipGreeting:
      "בסדר גמור! אפשר לבקש ממני המלצות, לשאול על סדרה ספציפית, או סתם לפטפט.",
    processing: "רגע אחד, מחפש את ההמלצות הכי מתאימות לך...",
    inputPlaceholder: "כתוב הודעה...",
    send: "שלח",
    languageToggleLabel: "English",
    ratingLabel: "דירוג",
    seasonsLabel: "עונות",
    castLabel: "שחקנים ראשיים",
    watchProvidersLabel: "איפה לצפות",
    trailerLabel: "טריילר",
    closeModal: "סגור",
    detailsLoading: "טוען פרטים נוספים...",
    genericError: "מצטערים, קרתה תקלה. נסו שוב מאוחר יותר.",
  },
  en: {
    openingMessage:
      "Hi! I'm CineMatch 🎬 - here to help you find your next favorite show. Want to answer a few quick questions so I can learn your taste?",
    startOnboarding: "Yes, let's start",
    skipToChat: "No thanks, take me to the regular chat",
    skipGreeting:
      "Sure thing! Ask me for recommendations, ask about a specific show, or just chat.",
    processing: "One sec, finding the best picks for you...",
    inputPlaceholder: "Type a message...",
    send: "Send",
    languageToggleLabel: "עברית",
    ratingLabel: "Rating",
    seasonsLabel: "Seasons",
    castLabel: "Cast",
    watchProvidersLabel: "Where to watch",
    trailerLabel: "Trailer",
    closeModal: "Close",
    detailsLoading: "Loading more details...",
  },
};
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (2 new tests, 10 total)

- [ ] **Step 9: Commit**

```bash
git add frontend/src/lib/onboarding.ts frontend/src/lib/onboarding.test.ts frontend/src/lib/i18n.ts frontend/src/lib/i18n.test.ts
git commit -m "Add onboarding question data and frontend UI strings (he/en)"
```

---

## Task 4: API Client (`lib/api.ts`)

**Files:**
- Create: `frontend/src/lib/api.ts`
- Test: `frontend/src/lib/api.test.ts`

Thin fetch wrappers around the 3 backend endpoints. All requests are
same-origin (`/api/...`) - in dev, Vite's proxy (Task 1, Step 4) forwards
these to `http://localhost:8000`; in production, FastAPI serves both the
static frontend and `/api/*` from the same origin, so no `fetch` base URL
configuration is needed.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/lib/api.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { postRecommend, postChat, getShow, ApiError } from "./api";
import type { RecommendRequest, ChatRequest } from "./types";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("api client", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("postRecommend POSTs to /api/recommend and returns the parsed response", async () => {
    const request: RecommendRequest = {
      answers: { genre: "drama", length: "any", era: "any", tone: "any", popularity: "any" },
      lang: "he",
    };
    const responseBody = { intro: "...", outro: "...", cluster_id: 1, recommendations: [] };
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(responseBody));

    const result = await postRecommend(request);

    expect(fetch).toHaveBeenCalledWith(
      "/api/recommend",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      })
    );
    expect(result).toEqual(responseBody);
  });

  it("postChat POSTs to /api/chat and returns the parsed response", async () => {
    const request: ChatRequest = {
      conversation: [{ role: "user", content: "hi" }],
      prev_recs: null,
      lang: "he",
    };
    const responseBody = { reply: "שלום!", recommendations: null, explanation: null };
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(responseBody));

    const result = await postChat(request);

    expect(fetch).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      })
    );
    expect(result).toEqual(responseBody);
  });

  it("getShow GETs /api/show/{title} with the lang query param and returns the parsed response", async () => {
    const responseBody = {
      title: "Severance",
      genres: "Drama",
      rating: 8.7,
      overview: "A team at Lumon Industries...",
      poster_path: null,
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
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(responseBody));

    const result = await getShow("Severance", "he");

    expect(fetch).toHaveBeenCalledWith("/api/show/Severance?lang=he");
    expect(result).toEqual(responseBody);
  });

  it("getShow URL-encodes the title", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({}));

    await getShow("Brooklyn Nine-Nine: S1", "en");

    expect(fetch).toHaveBeenCalledWith("/api/show/Brooklyn%20Nine-Nine%3A%20S1?lang=en");
  });

  it("getShow throws ApiError with the backend's detail message on 404", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ detail: "הסדרה לא נמצאה" }, 404));

    await expect(getShow("Unknown Show", "he")).rejects.toMatchObject({
      name: "ApiError",
      message: "הסדרה לא נמצאה",
      status: 404,
    });
  });

  it("postRecommend throws ApiError on a non-2xx response", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({}, 500));
    const request: RecommendRequest = {
      answers: { genre: "any", length: "any", era: "any", tone: "any", popularity: "any" },
      lang: "he",
    };

    await expect(postRecommend(request)).rejects.toMatchObject({
      name: "ApiError",
      status: 500,
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './api'`

- [ ] **Step 3: Implement `lib/api.ts`**

Create `frontend/src/lib/api.ts`:

```typescript
import type {
  ChatRequest,
  ChatResponse,
  Lang,
  RecommendRequest,
  RecommendResponse,
  ShowDetails,
} from "./types";

const API_BASE = "/api";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function postJson<TResponse>(path: string, body: unknown): Promise<TResponse> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new ApiError(`POST ${path} failed with status ${res.status}`, res.status);
  }
  return res.json() as Promise<TResponse>;
}

export function postRecommend(request: RecommendRequest): Promise<RecommendResponse> {
  return postJson<RecommendResponse>("/recommend", request);
}

export function postChat(request: ChatRequest): Promise<ChatResponse> {
  return postJson<ChatResponse>("/chat", request);
}

export async function getShow(title: string, lang: Lang): Promise<ShowDetails> {
  const url = `${API_BASE}/show/${encodeURIComponent(title)}?lang=${lang}`;
  const res = await fetch(url);
  if (!res.ok) {
    let detail = `GET ${url} failed with status ${res.status}`;
    const body = await res.json().catch(() => null);
    if (body && typeof body.detail === "string") {
      detail = body.detail;
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<ShowDetails>;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (6 new tests, 16 total)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/lib/api.test.ts
git commit -m "Add API client for /api/recommend, /api/chat, /api/show"
```

---

## Task 5: Conversation State Machine (`lib/chatReducer.ts`)

**Files:**
- Create: `frontend/src/lib/chatReducer.ts`
- Test: `frontend/src/lib/chatReducer.test.ts`

This is the heart of the frontend: a pure reducer that turns the 7-step UX
flow from the spec into a sequence of typed `ChatMessage` items. The opening
message (spec step 1) and each of the 5 onboarding questions (spec step 2)
are modeled as the same `ChoiceMessage` type - a bot bubble with buttons that
"closes" (records `selectedValue`/`selectedLabel`) once answered, exactly
like the WhatsApp-style behavior described in the spec. `hooks/useChatState.ts`
(Task 12) wraps this reducer and triggers the actual `/api/recommend` and
`/api/chat` calls in response to phase changes.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/lib/chatReducer.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import {
  chatReducer,
  createInitialState,
  type ChatState,
  type ChoiceMessage,
  type RecommendationsMessage,
  type TextMessage,
} from "./chatReducer";
import { ONBOARDING_QUESTIONS } from "./onboarding";
import type { RecCard } from "./types";

const SAMPLE_CARD: RecCard = {
  title: "Severance",
  genres: "Drama, Sci-Fi & Fantasy",
  rating: 8.7,
  overview: "A team at Lumon Industries...",
  poster_path: "/abc.jpg",
  decade_str: "2020s",
  num_seasons: 2,
};

describe("createInitialState", () => {
  it("starts in the intro phase with one choice message offering onboarding or skip", () => {
    const state = createInitialState("he");

    expect(state.phase).toBe("intro");
    expect(state.lang).toBe("he");
    expect(state.messages).toHaveLength(1);

    const opening = state.messages[0] as ChoiceMessage;
    expect(opening.type).toBe("choice");
    expect(opening.options.map((o) => o.value)).toEqual(["start", "skip"]);
    expect(opening.selectedValue).toBeUndefined();
  });
});

describe("chatReducer", () => {
  it("START_ONBOARDING closes the opening message and shows the first onboarding question", () => {
    const state = createInitialState("he");

    const next = chatReducer(state, { type: "START_ONBOARDING" });

    expect(next.phase).toBe("onboarding");
    expect(next.onboardingStepIndex).toBe(0);
    expect(next.messages).toHaveLength(2);

    const opening = next.messages[0] as ChoiceMessage;
    expect(opening.selectedValue).toBe("start");
    expect(opening.selectedLabel).toBeTruthy();

    const firstQuestion = next.messages[1] as ChoiceMessage;
    expect(firstQuestion.type).toBe("choice");
    expect(firstQuestion.options.map((o) => o.value)).toEqual(
      ONBOARDING_QUESTIONS[0].options.map((o) => o.value)
    );
  });

  it("SKIP_TO_CHAT closes the opening message and goes straight to chat with a greeting", () => {
    const state = createInitialState("he");

    const next = chatReducer(state, { type: "SKIP_TO_CHAT" });

    expect(next.phase).toBe("chat");
    expect(next.messages).toHaveLength(2);

    const opening = next.messages[0] as ChoiceMessage;
    expect(opening.selectedValue).toBe("skip");

    const greeting = next.messages[1] as TextMessage;
    expect(greeting.type).toBe("text");
    expect(greeting.role).toBe("assistant");
    expect(greeting.content.length).toBeGreaterThan(0);
  });

  it("walks through all 5 onboarding questions and ends in loading_recommend with all answers recorded", () => {
    let state = chatReducer(createInitialState("he"), { type: "START_ONBOARDING" });

    const answeredValues = ["drama", "medium", "recent", "serious_drama", "hidden_gem"];
    for (let i = 0; i < ONBOARDING_QUESTIONS.length; i++) {
      const question = ONBOARDING_QUESTIONS[i];
      state = chatReducer(state, {
        type: "ANSWER_ONBOARDING_QUESTION",
        questionId: question.id,
        value: answeredValues[i],
      });
    }

    expect(state.phase).toBe("loading_recommend");
    expect(state.onboardingAnswers).toEqual({
      genre: "drama",
      length: "medium",
      era: "recent",
      tone: "serious_drama",
      popularity: "hidden_gem",
    });

    // opening message + 5 questions, all closed (answered)
    const choiceMessages = state.messages.filter(
      (m): m is ChoiceMessage => m.type === "choice"
    );
    expect(choiceMessages).toHaveLength(6);
    for (const message of choiceMessages) {
      expect(message.selectedValue).toBeTruthy();
    }
  });

  it("RECOMMEND_SUCCESS appends intro, recommendation cards, and outro, and returns to chat phase", () => {
    let state = chatReducer(createInitialState("he"), { type: "START_ONBOARDING" });
    state = { ...state, phase: "loading_recommend" };

    const next = chatReducer(state, {
      type: "RECOMMEND_SUCCESS",
      intro: "אתה בטעם של: דרמות מתח",
      outro: "מקווה שאהבת!",
      cards: [SAMPLE_CARD],
    });

    expect(next.phase).toBe("chat");
    expect(next.prevRecs).toEqual([SAMPLE_CARD]);

    const last3 = next.messages.slice(-3);
    expect((last3[0] as TextMessage).content).toBe("אתה בטעם של: דרמות מתח");
    expect((last3[1] as RecommendationsMessage).type).toBe("recommendations");
    expect((last3[1] as RecommendationsMessage).cards).toEqual([SAMPLE_CARD]);
    expect((last3[2] as TextMessage).content).toBe("מקווה שאהבת!");
  });

  it("RECOMMEND_ERROR appends an assistant message and returns to chat phase", () => {
    const state: ChatState = { ...createInitialState("he"), phase: "loading_recommend" };

    const next = chatReducer(state, { type: "RECOMMEND_ERROR", message: "שגיאה זמנית" });

    expect(next.phase).toBe("chat");
    const last = next.messages[next.messages.length - 1] as TextMessage;
    expect(last.content).toBe("שגיאה זמנית");
    expect(last.role).toBe("assistant");
  });

  it("SEND_USER_MESSAGE appends a user message and moves to loading_chat", () => {
    const state: ChatState = { ...createInitialState("he"), phase: "chat" };

    const next = chatReducer(state, { type: "SEND_USER_MESSAGE", content: "תמליץ לי על קומדיה" });

    expect(next.phase).toBe("loading_chat");
    const last = next.messages[next.messages.length - 1] as TextMessage;
    expect(last.role).toBe("user");
    expect(last.content).toBe("תמליץ לי על קומדיה");
  });

  it("CHAT_SUCCESS with recommendations appends reply, cards, and explanation, and updates prevRecs", () => {
    const state: ChatState = { ...createInitialState("he"), phase: "loading_chat" };

    const next = chatReducer(state, {
      type: "CHAT_SUCCESS",
      reply: "הנה כמה הצעות:",
      cards: [SAMPLE_CARD],
      explanation: "שתיהן דרמות מתח עכשוויות.",
    });

    expect(next.phase).toBe("chat");
    expect(next.prevRecs).toEqual([SAMPLE_CARD]);

    const last3 = next.messages.slice(-3);
    expect((last3[0] as TextMessage).content).toBe("הנה כמה הצעות:");
    expect((last3[1] as RecommendationsMessage).cards).toEqual([SAMPLE_CARD]);
    expect((last3[2] as TextMessage).content).toBe("שתיהן דרמות מתח עכשוויות.");
  });

  it("CHAT_SUCCESS without recommendations only appends the reply", () => {
    const state: ChatState = { ...createInitialState("he"), phase: "loading_chat" };

    const next = chatReducer(state, { type: "CHAT_SUCCESS", reply: "שלום! איך אפשר לעזור?" });

    expect(next.phase).toBe("chat");
    expect(next.prevRecs).toBeNull();
    const last = next.messages[next.messages.length - 1] as TextMessage;
    expect(last.content).toBe("שלום! איך אפשר לעזור?");
  });

  it("CHAT_ERROR appends an assistant message and returns to chat phase", () => {
    const state: ChatState = { ...createInitialState("he"), phase: "loading_chat" };

    const next = chatReducer(state, { type: "CHAT_ERROR", message: "שגיאה זמנית" });

    expect(next.phase).toBe("chat");
    const last = next.messages[next.messages.length - 1] as TextMessage;
    expect(last.content).toBe("שגיאה זמנית");
  });

  it("TOGGLE_LANG flips between he and en", () => {
    const state = createInitialState("he");

    const next = chatReducer(state, { type: "TOGGLE_LANG" });
    expect(next.lang).toBe("en");

    const back = chatReducer(next, { type: "TOGGLE_LANG" });
    expect(back.lang).toBe("he");
  });

  it("RESTORE replaces the entire state (used to load persisted state)", () => {
    const restored: ChatState = { ...createInitialState("en"), phase: "chat" };

    const next = chatReducer(createInitialState("he"), { type: "RESTORE", state: restored });

    expect(next).toEqual(restored);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './chatReducer'`

- [ ] **Step 3: Implement `lib/chatReducer.ts`**

Create `frontend/src/lib/chatReducer.ts`:

```typescript
import { UI_STRINGS } from "./i18n";
import { ONBOARDING_QUESTIONS, type OnboardingQuestion } from "./onboarding";
import type { Lang, OnboardingAnswers, RecCard } from "./types";

export type Phase = "intro" | "onboarding" | "loading_recommend" | "loading_chat" | "chat";

export interface TextMessage {
  id: string;
  type: "text";
  role: "user" | "assistant";
  content: string;
}

export interface ChoiceOption {
  value: string;
  label: string;
}

export interface ChoiceMessage {
  id: string;
  type: "choice";
  role: "assistant";
  prompt: string;
  options: ChoiceOption[];
  selectedValue?: string;
  selectedLabel?: string;
}

export interface RecommendationsMessage {
  id: string;
  type: "recommendations";
  role: "assistant";
  cards: RecCard[];
}

export type ChatMessage = TextMessage | ChoiceMessage | RecommendationsMessage;

export interface ChatState {
  phase: Phase;
  lang: Lang;
  onboardingStepIndex: number;
  onboardingAnswers: OnboardingAnswers;
  messages: ChatMessage[];
  prevRecs: RecCard[] | null;
}

export type ChatAction =
  | { type: "START_ONBOARDING" }
  | { type: "SKIP_TO_CHAT" }
  | { type: "ANSWER_ONBOARDING_QUESTION"; questionId: keyof OnboardingAnswers; value: string }
  | { type: "RECOMMEND_SUCCESS"; intro: string; outro: string; cards: RecCard[] }
  | { type: "RECOMMEND_ERROR"; message: string }
  | { type: "SEND_USER_MESSAGE"; content: string }
  | {
      type: "CHAT_SUCCESS";
      reply: string;
      cards?: RecCard[] | null;
      explanation?: string | null;
    }
  | { type: "CHAT_ERROR"; message: string }
  | { type: "TOGGLE_LANG" }
  | { type: "RESTORE"; state: ChatState };

const DEFAULT_ONBOARDING_ANSWERS: OnboardingAnswers = {
  genre: "any",
  length: "any",
  era: "any",
  tone: "any",
  popularity: "any",
};

export function createInitialState(lang: Lang): ChatState {
  const strings = UI_STRINGS[lang];
  return {
    phase: "intro",
    lang,
    onboardingStepIndex: -1,
    onboardingAnswers: { ...DEFAULT_ONBOARDING_ANSWERS },
    messages: [
      {
        id: crypto.randomUUID(),
        type: "choice",
        role: "assistant",
        prompt: strings.openingMessage,
        options: [
          { value: "start", label: strings.startOnboarding },
          { value: "skip", label: strings.skipToChat },
        ],
      },
    ],
    prevRecs: null,
  };
}

function buildQuestionMessage(question: OnboardingQuestion, lang: Lang): ChoiceMessage {
  return {
    id: crypto.randomUUID(),
    type: "choice",
    role: "assistant",
    prompt: question.prompt[lang],
    options: question.options.map((o) => ({ value: o.value, label: o.label[lang] })),
  };
}

function closeLastChoice(
  messages: ChatMessage[],
  selectedValue: string,
  selectedLabel: string
): ChatMessage[] {
  const last = messages[messages.length - 1];
  if (last.type !== "choice") return messages;
  const updated: ChoiceMessage = { ...last, selectedValue, selectedLabel };
  return [...messages.slice(0, -1), updated];
}

function textMessage(role: "user" | "assistant", content: string): TextMessage {
  return { id: crypto.randomUUID(), type: "text", role, content };
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "START_ONBOARDING": {
      const strings = UI_STRINGS[state.lang];
      const firstQuestion = ONBOARDING_QUESTIONS[0];
      return {
        ...state,
        phase: "onboarding",
        onboardingStepIndex: 0,
        messages: [
          ...closeLastChoice(state.messages, "start", strings.startOnboarding),
          buildQuestionMessage(firstQuestion, state.lang),
        ],
      };
    }

    case "SKIP_TO_CHAT": {
      const strings = UI_STRINGS[state.lang];
      return {
        ...state,
        phase: "chat",
        messages: [
          ...closeLastChoice(state.messages, "skip", strings.skipToChat),
          textMessage("assistant", strings.skipGreeting),
        ],
      };
    }

    case "ANSWER_ONBOARDING_QUESTION": {
      const question = ONBOARDING_QUESTIONS[state.onboardingStepIndex];
      const option = question.options.find((o) => o.value === action.value);
      const label = option ? option.label[state.lang] : action.value;

      const updatedAnswers: OnboardingAnswers = {
        ...state.onboardingAnswers,
        [action.questionId]: action.value,
      };
      const messages = closeLastChoice(state.messages, action.value, label);

      const nextIndex = state.onboardingStepIndex + 1;
      if (nextIndex < ONBOARDING_QUESTIONS.length) {
        return {
          ...state,
          onboardingAnswers: updatedAnswers,
          onboardingStepIndex: nextIndex,
          messages: [...messages, buildQuestionMessage(ONBOARDING_QUESTIONS[nextIndex], state.lang)],
        };
      }
      return {
        ...state,
        onboardingAnswers: updatedAnswers,
        phase: "loading_recommend",
        messages,
      };
    }

    case "RECOMMEND_SUCCESS": {
      const recsMessage: RecommendationsMessage = {
        id: crypto.randomUUID(),
        type: "recommendations",
        role: "assistant",
        cards: action.cards,
      };
      return {
        ...state,
        phase: "chat",
        prevRecs: action.cards,
        messages: [
          ...state.messages,
          textMessage("assistant", action.intro),
          recsMessage,
          textMessage("assistant", action.outro),
        ],
      };
    }

    case "RECOMMEND_ERROR":
      return {
        ...state,
        phase: "chat",
        messages: [...state.messages, textMessage("assistant", action.message)],
      };

    case "SEND_USER_MESSAGE":
      return {
        ...state,
        phase: "loading_chat",
        messages: [...state.messages, textMessage("user", action.content)],
      };

    case "CHAT_SUCCESS": {
      const messages: ChatMessage[] = [...state.messages, textMessage("assistant", action.reply)];
      let prevRecs = state.prevRecs;

      if (action.cards && action.cards.length > 0) {
        messages.push({
          id: crypto.randomUUID(),
          type: "recommendations",
          role: "assistant",
          cards: action.cards,
        });
        prevRecs = action.cards;

        if (action.explanation) {
          messages.push(textMessage("assistant", action.explanation));
        }
      }

      return { ...state, phase: "chat", messages, prevRecs };
    }

    case "CHAT_ERROR":
      return {
        ...state,
        phase: "chat",
        messages: [...state.messages, textMessage("assistant", action.message)],
      };

    case "TOGGLE_LANG":
      return { ...state, lang: state.lang === "he" ? "en" : "he" };

    case "RESTORE":
      return action.state;

    default:
      return state;
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (12 new tests, 28 total)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/chatReducer.ts frontend/src/lib/chatReducer.test.ts
git commit -m "Add conversation state machine reducer"
```

---

## Task 6: localStorage Persistence (`lib/storage.ts`)

**Files:**
- Create: `frontend/src/lib/storage.ts`
- Test: `frontend/src/lib/storage.test.ts`

Implements the spec's step 7 ("conversation history is saved in the
browser's `localStorage`, not on the server; a page refresh does not reset
it; a different browser/computer starts a new chat"). `useChatState` (Task
12) calls `loadPersistedState()` once on mount and `savePersistedState(state)`
after every reducer dispatch.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/lib/storage.test.ts`:

```typescript
import { describe, it, expect, beforeEach } from "vitest";
import { loadPersistedState, savePersistedState, STORAGE_KEY } from "./storage";
import { createInitialState, chatReducer } from "./chatReducer";

describe("storage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("loadPersistedState returns null when nothing has been saved", () => {
    expect(loadPersistedState()).toBeNull();
  });

  it("savePersistedState then loadPersistedState round-trips the state", () => {
    const state = chatReducer(createInitialState("he"), { type: "START_ONBOARDING" });

    savePersistedState(state);
    const loaded = loadPersistedState();

    expect(loaded).toEqual(state);
  });

  it("loadPersistedState returns null if the stored JSON is corrupted", () => {
    localStorage.setItem(STORAGE_KEY, "{not valid json");

    expect(loadPersistedState()).toBeNull();
  });

  it("loadPersistedState returns null if the stored state has an unrecognized shape", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ foo: "bar" }));

    expect(loadPersistedState()).toBeNull();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './storage'`

- [ ] **Step 3: Implement `lib/storage.ts`**

Create `frontend/src/lib/storage.ts`:

```typescript
import type { ChatState } from "./chatReducer";

export const STORAGE_KEY = "cinematch:chat-state";

function isChatState(value: unknown): value is ChatState {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.phase === "string" &&
    typeof candidate.lang === "string" &&
    typeof candidate.onboardingStepIndex === "number" &&
    typeof candidate.onboardingAnswers === "object" &&
    Array.isArray(candidate.messages)
  );
}

export function loadPersistedState(): ChatState | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (raw === null) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }

  return isChatState(parsed) ? parsed : null;
}

export function savePersistedState(state: ChatState): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (4 new tests, 32 total)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/storage.ts frontend/src/lib/storage.test.ts
git commit -m "Add localStorage persistence for conversation state"
```

---

## Task 7: shadcn/ui Primitives (`components/ui/`)

**Files:**
- Create: `frontend/src/components/ui/button.tsx`, `card.tsx`, `dialog.tsx`, `scroll-area.tsx`, `badge.tsx`, `input.tsx` (generated)
- Test: `frontend/src/components/ui/smoke.test.tsx`

Generates the shadcn/ui primitives listed in the spec's file structure
(`components/ui/`: button, card, dialog, input - plus `scroll-area` and
`badge`, which the chat components from Task 8 onward need for the
auto-scrolling message list and rating/genre tags). `components.json` from
Task 1 already configures the `@/components/ui` and `@/lib/utils` aliases, so
the generator runs non-interactively.

- [ ] **Step 1: Generate the primitives**

Run (from `frontend/`):
```powershell
npx shadcn@latest add button card dialog scroll-area badge input -y
```
Expected: creates `src/components/ui/button.tsx`, `card.tsx`, `dialog.tsx`,
`scroll-area.tsx`, `badge.tsx`, `input.tsx`, and installs any missing peer
dependencies (e.g. `@radix-ui/react-scroll-area`) into `package.json`.

- [ ] **Step 2: Write a smoke test**

Create `frontend/src/components/ui/smoke.test.tsx`:

```tsx
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
```

- [ ] **Step 3: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (4 new tests, 36 total). If any primitive failed to generate
correctly, this step will fail with a clear import error naming the missing
file - re-run the `npx shadcn@latest add ...` command for that component.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ui frontend/package.json frontend/package-lock.json frontend/components.json
git commit -m "Add shadcn/ui primitives (button, card, dialog, scroll-area, badge, input)"
```

---

## Task 8: MessageBubble Component

**Files:**
- Create: `frontend/src/components/chat/MessageBubble.tsx`
- Test: `frontend/src/components/chat/MessageBubble.test.tsx`

Renders a single `TextMessage` (Task 5) as a WhatsApp-style bubble: user
messages right-aligned with the primary color, assistant messages
left-aligned in a card. Each bubble sets `dir="rtl"`/`dir="ltr"` from the
current `lang` so Hebrew and English text both render correctly regardless
of the page's overall direction.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/chat/MessageBubble.test.tsx`:

```tsx
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
  it("renders a user message right-aligned with rtl direction for Hebrew", () => {
    render(<MessageBubble message={userMessage} lang="he" />);

    const bubble = screen.getByText("תמליץ לי על קומדיה");
    expect(bubble).toHaveAttribute("dir", "rtl");
    expect(bubble.parentElement).toHaveClass("justify-end");
  });

  it("renders an assistant message left-aligned with ltr direction for English", () => {
    render(<MessageBubble message={assistantMessage} lang="en" />);

    const bubble = screen.getByText("Here are some comedies you might like.");
    expect(bubble).toHaveAttribute("dir", "ltr");
    expect(bubble.parentElement).toHaveClass("justify-start");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './MessageBubble'`

- [ ] **Step 3: Implement `MessageBubble.tsx`**

Create `frontend/src/components/chat/MessageBubble.tsx`:

```tsx
import { cn } from "@/lib/utils";
import type { TextMessage } from "@/lib/chatReducer";
import type { Lang } from "@/lib/types";

interface MessageBubbleProps {
  message: TextMessage;
  lang: Lang;
}

export function MessageBubble({ message, lang }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div
        dir={lang === "he" ? "rtl" : "ltr"}
        className={cn(
          "max-w-[80%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm",
          isUser
            ? "bg-primary text-primary-foreground"
            : "border border-border bg-card text-card-foreground"
        )}
      >
        {message.content}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (2 new tests, 38 total)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/chat/MessageBubble.tsx frontend/src/components/chat/MessageBubble.test.tsx
git commit -m "Add MessageBubble component"
```

---

## Task 9: OnboardingQuestion Component

**Files:**
- Create: `frontend/src/components/chat/OnboardingQuestion.tsx`
- Test: `frontend/src/components/chat/OnboardingQuestion.test.tsx`

Renders a `ChoiceMessage` (Task 5) - used for both the opening message (spec
step 1) and each of the 5 onboarding questions (spec step 2). While
unanswered, shows the prompt with one button per option; once
`selectedValue` is set, the buttons are replaced by a single badge showing
the chosen `selectedLabel` - the "closes after selection, like WhatsApp"
behavior from the spec.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/chat/OnboardingQuestion.test.tsx`:

```tsx
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './OnboardingQuestion'`

- [ ] **Step 3: Implement `OnboardingQuestion.tsx`**

Create `frontend/src/components/chat/OnboardingQuestion.tsx`:

```tsx
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ChoiceMessage } from "@/lib/chatReducer";
import type { Lang } from "@/lib/types";

interface OnboardingQuestionProps {
  message: ChoiceMessage;
  lang: Lang;
  onSelect: (value: string) => void;
}

export function OnboardingQuestion({ message, lang, onSelect }: OnboardingQuestionProps) {
  return (
    <div className="flex w-full justify-start">
      <div
        dir={lang === "he" ? "rtl" : "ltr"}
        className="max-w-[80%] rounded-lg border border-border bg-card px-3 py-2 text-sm text-card-foreground"
      >
        <p className="mb-2">{message.prompt}</p>
        {message.selectedValue ? (
          <Badge>{message.selectedLabel}</Badge>
        ) : (
          <div className="flex flex-wrap gap-2">
            {message.options.map((option) => (
              <Button
                key={option.value}
                type="button"
                size="sm"
                variant="secondary"
                onClick={() => onSelect(option.value)}
              >
                {option.label}
              </Button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (2 new tests, 40 total)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/chat/OnboardingQuestion.tsx frontend/src/components/chat/OnboardingQuestion.test.tsx
git commit -m "Add OnboardingQuestion component"
```

---

## Task 10: RecCard & RecCardGrid Components

**Files:**
- Create: `frontend/src/components/chat/RecCard.tsx`
- Create: `frontend/src/components/chat/RecCardGrid.tsx`
- Test: `frontend/src/components/chat/RecCard.test.tsx`
- Test: `frontend/src/components/chat/RecCardGrid.test.tsx`

`RecCard` renders one recommendation as a small poster card (poster, title,
genres, rating) per spec step 4. Posters use the catalog's `poster_path` via
`https://image.tmdb.org/t/p/w342{poster_path}` (spec line 102); if
`poster_path` is `null`, a text placeholder with the title is shown instead -
no error, per the spec's edge-case notes. `RecCardGrid` renders a
`RecommendationsMessage` (Task 5) as a wrapped row of `RecCard`s inside a bot
bubble. Clicking a card calls `onSelectShow(title)`, which `ChatWindow` (Task
12) wires up to open `ShowDetailsModal` (Task 11).

- [ ] **Step 1: Write the failing test for `RecCard`**

Create `frontend/src/components/chat/RecCard.test.tsx`:

```tsx
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './RecCard'`

- [ ] **Step 3: Implement `RecCard.tsx`**

Create `frontend/src/components/chat/RecCard.tsx`:

```tsx
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { UI_STRINGS } from "@/lib/i18n";
import type { Lang, RecCard as RecCardData } from "@/lib/types";

const TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w342";

interface RecCardProps {
  show: RecCardData;
  lang: Lang;
  onClick: (title: string) => void;
}

export function RecCard({ show, lang, onClick }: RecCardProps) {
  const strings = UI_STRINGS[lang];

  return (
    <Card
      role="button"
      tabIndex={0}
      onClick={() => onClick(show.title)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") onClick(show.title);
      }}
      className="w-40 shrink-0 cursor-pointer overflow-hidden transition hover:shadow-md"
    >
      {show.poster_path ? (
        <img
          src={`${TMDB_POSTER_BASE}${show.poster_path}`}
          alt={show.title}
          className="h-56 w-full object-cover"
        />
      ) : (
        <div className="flex h-56 w-full items-center justify-center bg-muted p-2 text-center text-sm text-muted-foreground">
          {show.title}
        </div>
      )}
      <CardContent className="space-y-1 p-2" dir={lang === "he" ? "rtl" : "ltr"}>
        <p className="truncate text-sm font-semibold">{show.title}</p>
        <p className="truncate text-xs text-muted-foreground">{show.genres}</p>
        <Badge variant="secondary">
          {strings.ratingLabel}: {show.rating.toFixed(1)}
        </Badge>
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (2 new tests, 42 total)

- [ ] **Step 5: Write the failing test for `RecCardGrid`**

Create `frontend/src/components/chat/RecCardGrid.test.tsx`:

```tsx
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
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './RecCardGrid'`

- [ ] **Step 7: Implement `RecCardGrid.tsx`**

Create `frontend/src/components/chat/RecCardGrid.tsx`:

```tsx
import { RecCard } from "./RecCard";
import type { RecommendationsMessage } from "@/lib/chatReducer";
import type { Lang } from "@/lib/types";

interface RecCardGridProps {
  message: RecommendationsMessage;
  lang: Lang;
  onSelectShow: (title: string) => void;
}

export function RecCardGrid({ message, lang, onSelectShow }: RecCardGridProps) {
  return (
    <div className="flex w-full justify-start">
      <div className="flex max-w-full flex-wrap gap-3">
        {message.cards.map((show) => (
          <RecCard key={show.title} show={show} lang={lang} onClick={onSelectShow} />
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (1 new test, 43 total)

- [ ] **Step 9: Commit**

```bash
git add frontend/src/components/chat/RecCard.tsx frontend/src/components/chat/RecCard.test.tsx frontend/src/components/chat/RecCardGrid.tsx frontend/src/components/chat/RecCardGrid.test.tsx
git commit -m "Add RecCard and RecCardGrid components"
```

---

## Task 11: ShowDetailsModal Component

**Files:**
- Create: `frontend/src/components/chat/ShowDetailsModal.tsx`
- Test: `frontend/src/components/chat/ShowDetailsModal.test.tsx`

Implements spec step 4's popup: clicking a `RecCard` opens a modal that
immediately shows the catalog data already in hand (poster, overview,
rating, genres, decade, seasons, and - when available - the per-show LLM
`explanation` from `/api/recommend`'s `ShowSummary`), then fetches
`GET /api/show/{title}` (Task 18 of the backend plan) in the background to
add the trailer link, cast, and watch providers. Per the spec's edge-case
notes, if that lookup fails or TMDB has no match, the modal silently keeps
showing only the catalog data - no error is shown to the user.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/chat/ShowDetailsModal.test.tsx`:

```tsx
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
  cast: ["Adam Scott", "Britt Lower"],
  watch_providers: ["Apple TV+"],
};

const detailsWithoutExtras: ShowDetails = {
  ...fullDetails,
  trailer_url: null,
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
    expect(screen.getByRole("link", { name: "טריילר" })).toHaveAttribute(
      "href",
      "https://www.youtube.com/watch?v=abc123"
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './ShowDetailsModal'`

- [ ] **Step 3: Implement `ShowDetailsModal.tsx`**

Create `frontend/src/components/chat/ShowDetailsModal.tsx`:

```tsx
import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { getShow } from "@/lib/api";
import { UI_STRINGS } from "@/lib/i18n";
import type { Lang, RecCard, ShowDetails } from "@/lib/types";

const TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w342";

interface ShowDetailsModalProps {
  show: RecCard | null;
  lang: Lang;
  onClose: () => void;
}

export function ShowDetailsModal({ show, lang, onClose }: ShowDetailsModalProps) {
  const strings = UI_STRINGS[lang];
  const [details, setDetails] = useState<ShowDetails | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setDetails(null);
    if (!show) return;

    let cancelled = false;
    setLoading(true);

    getShow(show.title, lang)
      .then((result) => {
        if (!cancelled) setDetails(result);
      })
      .catch(() => {
        // TMDB lookup failed or had no match - keep showing catalog data only.
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [show, lang]);

  if (!show) return null;

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent dir={lang === "he" ? "rtl" : "ltr"} className="max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{show.title}</DialogTitle>
          <DialogDescription>
            {show.genres} · {show.decade_str}
            {show.num_seasons != null ? ` · ${strings.seasonsLabel}: ${show.num_seasons}` : ""}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 text-sm">
          {show.poster_path && (
            <img
              src={`${TMDB_POSTER_BASE}${show.poster_path}`}
              alt={show.title}
              className="mx-auto h-64 rounded-md object-cover"
            />
          )}

          <Badge variant="secondary">
            {strings.ratingLabel}: {show.rating.toFixed(1)}
          </Badge>

          <p>{show.overview}</p>

          {show.explanation && (
            <p className="rounded-md bg-muted p-2 italic text-muted-foreground">
              {show.explanation}
            </p>
          )}

          {loading && <p className="text-muted-foreground">{strings.detailsLoading}</p>}

          {details?.trailer_url && (
            <a
              href={details.trailer_url}
              target="_blank"
              rel="noreferrer"
              className="text-primary underline"
            >
              {strings.trailerLabel}
            </a>
          )}

          {details && details.cast.length > 0 && (
            <p>
              <span className="font-semibold">{strings.castLabel}: </span>
              {details.cast.join(", ")}
            </p>
          )}

          {details && details.watch_providers.length > 0 && (
            <p>
              <span className="font-semibold">{strings.watchProvidersLabel}: </span>
              {details.watch_providers.join(", ")}
            </p>
          )}
        </div>

        <DialogFooter>
          <Button variant="secondary" onClick={onClose}>
            {strings.closeModal}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (4 new tests, 47 total)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/chat/ShowDetailsModal.tsx frontend/src/components/chat/ShowDetailsModal.test.tsx
git commit -m "Add ShowDetailsModal component"
```

---

## Task 12: `useChatState` Hook

**Files:**
- Create: `frontend/src/hooks/useChatState.ts`
- Test: `frontend/src/hooks/useChatState.test.ts`

This hook wires the pure `chatReducer` (Task 5) to the outside world: it
seeds state from `localStorage` (Task 6) on first mount, persists every
change back to `localStorage`, and - whenever the reducer puts the
conversation into `loading_recommend` or `loading_chat` - calls
`postRecommend`/`postChat` (Task 4) and feeds the result back in via
`RECOMMEND_SUCCESS`/`RECOMMEND_ERROR`/`CHAT_SUCCESS`/`CHAT_ERROR`. This is the
only place in the frontend that talks to the network.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/hooks/useChatState.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useChatState } from "./useChatState";
import * as api from "@/lib/api";
import { chatReducer, createInitialState, type TextMessage } from "@/lib/chatReducer";
import { ONBOARDING_QUESTIONS } from "@/lib/onboarding";
import { savePersistedState, STORAGE_KEY } from "@/lib/storage";

vi.mock("@/lib/api");

describe("useChatState", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.resetAllMocks();
  });

  it("starts from createInitialState when nothing is persisted", () => {
    const { result } = renderHook(() => useChatState("he"));

    expect(result.current.state.phase).toBe("intro");
    expect(result.current.state.messages).toHaveLength(1);
  });

  it("restores persisted state on mount", () => {
    const persisted = chatReducer(createInitialState("en"), { type: "SKIP_TO_CHAT" });
    savePersistedState(persisted);

    const { result } = renderHook(() => useChatState("he"));

    expect(result.current.state.phase).toBe("chat");
    expect(result.current.state.lang).toBe("en");
  });

  it("persists state to localStorage after each dispatch", () => {
    const { result } = renderHook(() => useChatState("he"));

    act(() => {
      result.current.dispatch({ type: "TOGGLE_LANG" });
    });

    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY)!);
    expect(stored.lang).toBe("en");
  });

  it("calls postRecommend once onboarding completes and applies the result", async () => {
    vi.mocked(api.postRecommend).mockResolvedValue({
      intro: "אתה בטעם של: דרמות מתח",
      outro: "מקווה שאהבת!",
      cluster_id: 2,
      recommendations: [
        {
          title: "Severance",
          genres: "Drama, Sci-Fi & Fantasy",
          rating: 8.7,
          overview: "...",
          poster_path: "/abc.jpg",
          decade_str: "2020s",
          num_seasons: 2,
          binge_fit_score: 0.82,
          explanation: "...",
        },
      ],
    });

    const { result } = renderHook(() => useChatState("he"));

    act(() => {
      result.current.dispatch({ type: "START_ONBOARDING" });
    });
    for (const question of ONBOARDING_QUESTIONS) {
      act(() => {
        result.current.dispatch({
          type: "ANSWER_ONBOARDING_QUESTION",
          questionId: question.id,
          value: "any",
        });
      });
    }

    expect(result.current.state.phase).toBe("loading_recommend");

    await waitFor(() => {
      expect(result.current.state.phase).toBe("chat");
    });

    expect(api.postRecommend).toHaveBeenCalledWith({
      answers: {
        genre: "any",
        length: "any",
        era: "any",
        tone: "any",
        popularity: "any",
      },
      lang: "he",
    });
    expect(result.current.state.prevRecs).toHaveLength(1);
  });

  it("calls postChat when the user sends a message and applies the reply", async () => {
    vi.mocked(api.postChat).mockResolvedValue({
      reply: "הנה כמה הצעות:",
      recommendations: null,
      explanation: null,
    });

    const { result } = renderHook(() => useChatState("he"));

    act(() => {
      result.current.dispatch({ type: "SKIP_TO_CHAT" });
    });
    act(() => {
      result.current.dispatch({ type: "SEND_USER_MESSAGE", content: "תמליץ לי על קומדיה" });
    });

    expect(result.current.state.phase).toBe("loading_chat");

    await waitFor(() => {
      expect(result.current.state.phase).toBe("chat");
    });

    expect(api.postChat).toHaveBeenCalled();
    const lastMessage = result.current.state.messages[
      result.current.state.messages.length - 1
    ] as TextMessage;
    expect(lastMessage.content).toBe("הנה כמה הצעות:");
  });

  it("dispatches RECOMMEND_ERROR with a generic message when postRecommend rejects", async () => {
    vi.mocked(api.postRecommend).mockRejectedValue(new Error("network error"));

    const { result } = renderHook(() => useChatState("he"));

    act(() => {
      result.current.dispatch({ type: "START_ONBOARDING" });
    });
    for (const question of ONBOARDING_QUESTIONS) {
      act(() => {
        result.current.dispatch({
          type: "ANSWER_ONBOARDING_QUESTION",
          questionId: question.id,
          value: "any",
        });
      });
    }

    await waitFor(() => {
      expect(result.current.state.phase).toBe("chat");
    });

    const lastMessage = result.current.state.messages[
      result.current.state.messages.length - 1
    ] as TextMessage;
    expect(lastMessage.content.length).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './useChatState'`

- [ ] **Step 3: Implement `useChatState.ts`**

Create `frontend/src/hooks/useChatState.ts`:

```typescript
import { useEffect, useReducer } from "react";
import { postChat, postRecommend } from "@/lib/api";
import {
  chatReducer,
  createInitialState,
  type ChatAction,
  type ChatState,
  type TextMessage,
} from "@/lib/chatReducer";
import { UI_STRINGS } from "@/lib/i18n";
import { loadPersistedState, savePersistedState } from "@/lib/storage";
import type { Lang } from "@/lib/types";

export interface UseChatStateResult {
  state: ChatState;
  dispatch: React.Dispatch<ChatAction>;
}

export function useChatState(defaultLang: Lang): UseChatStateResult {
  const [state, dispatch] = useReducer(
    chatReducer,
    undefined,
    () => loadPersistedState() ?? createInitialState(defaultLang)
  );

  useEffect(() => {
    savePersistedState(state);
  }, [state]);

  // Onboarding finished -> ask the backend for cluster-based recommendations.
  useEffect(() => {
    if (state.phase !== "loading_recommend") return;

    let cancelled = false;
    postRecommend({ answers: state.onboardingAnswers, lang: state.lang })
      .then((response) => {
        if (cancelled) return;
        if (response.recommendations.length === 0) {
          dispatch({ type: "RECOMMEND_ERROR", message: response.intro });
        } else {
          dispatch({
            type: "RECOMMEND_SUCCESS",
            intro: response.intro,
            outro: response.outro,
            cards: response.recommendations,
          });
        }
      })
      .catch(() => {
        if (!cancelled) {
          dispatch({ type: "RECOMMEND_ERROR", message: UI_STRINGS[state.lang].genericError });
        }
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.phase]);

  // User sent a free-text message -> hand the full conversation to chat_turn.
  useEffect(() => {
    if (state.phase !== "loading_chat") return;

    let cancelled = false;
    const conversation = state.messages
      .filter((message): message is TextMessage => message.type === "text")
      .map((message) => ({ role: message.role, content: message.content }));

    postChat({ conversation, prev_recs: state.prevRecs, lang: state.lang })
      .then((response) => {
        if (cancelled) return;
        dispatch({
          type: "CHAT_SUCCESS",
          reply: response.reply,
          cards: response.recommendations,
          explanation: response.explanation,
        });
      })
      .catch(() => {
        if (!cancelled) {
          dispatch({ type: "CHAT_ERROR", message: UI_STRINGS[state.lang].genericError });
        }
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.phase]);

  return { state, dispatch };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (6 new tests, 53 total)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useChatState.ts frontend/src/hooks/useChatState.test.ts
git commit -m "Add useChatState hook (reducer + API calls + persistence)"
```

---

## Task 13: ChatWindow, App Wiring, and Final Build/E2E Verification

**Files:**
- Create: `frontend/src/components/chat/ChatWindow.tsx`
- Test: `frontend/src/components/chat/ChatWindow.test.tsx`
- Modify: `frontend/src/App.tsx`, `frontend/index.html`, `frontend/src/main.tsx`

`ChatWindow` is the top-level component: it calls `useChatState` (Task 12),
renders the message list (auto-scrolling, one component per message type from
Task 5), the input bar (enabled only in `chat` phase), the language toggle,
and `ShowDetailsModal` (Task 11). `App.tsx` becomes a one-line wrapper around
it. This task finishes with a production build and a manual browser
walkthrough of the full UX flow from the spec.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/chat/ChatWindow.test.tsx`:

```tsx
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL with `Cannot find module './ChatWindow'`

- [ ] **Step 3: Implement `ChatWindow.tsx`**

Create `frontend/src/components/chat/ChatWindow.tsx`:

```tsx
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChatState } from "@/hooks/useChatState";
import { UI_STRINGS } from "@/lib/i18n";
import { ONBOARDING_QUESTIONS } from "@/lib/onboarding";
import type { RecCard as RecCardData } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";
import { OnboardingQuestion } from "./OnboardingQuestion";
import { RecCardGrid } from "./RecCardGrid";
import { ShowDetailsModal } from "./ShowDetailsModal";

export function ChatWindow() {
  const { state, dispatch } = useChatState("he");
  const strings = UI_STRINGS[state.lang];
  const [inputValue, setInputValue] = useState("");
  const [selectedShow, setSelectedShow] = useState<RecCardData | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [state.messages]);

  const isLoading = state.phase === "loading_recommend" || state.phase === "loading_chat";
  const canSendMessage = state.phase === "chat";
  const dir = state.lang === "he" ? "rtl" : "ltr";

  function handleChoiceSelect(value: string) {
    if (state.phase === "intro") {
      if (value === "start") {
        dispatch({ type: "START_ONBOARDING" });
      } else {
        dispatch({ type: "SKIP_TO_CHAT" });
      }
      return;
    }
    if (state.phase === "onboarding") {
      const question = ONBOARDING_QUESTIONS[state.onboardingStepIndex];
      dispatch({ type: "ANSWER_ONBOARDING_QUESTION", questionId: question.id, value });
    }
  }

  function handleSelectShow(title: string) {
    for (const message of state.messages) {
      if (message.type === "recommendations") {
        const found = message.cards.find((card) => card.title === title);
        if (found) {
          setSelectedShow(found);
          return;
        }
      }
    }
  }

  function handleSend() {
    const trimmed = inputValue.trim();
    if (!trimmed) return;
    dispatch({ type: "SEND_USER_MESSAGE", content: trimmed });
    setInputValue("");
  }

  return (
    <div dir={dir} className="mx-auto flex h-screen max-w-2xl flex-col bg-background">
      <header className="flex items-center justify-between border-b border-border p-3">
        <h1 className="text-lg font-bold">CineMatch AI</h1>
        <Button variant="ghost" size="sm" onClick={() => dispatch({ type: "TOGGLE_LANG" })}>
          {strings.languageToggleLabel}
        </Button>
      </header>

      <ScrollArea className="flex-1 p-3">
        <div className="flex flex-col gap-3">
          {state.messages.map((message) => {
            switch (message.type) {
              case "text":
                return <MessageBubble key={message.id} message={message} lang={state.lang} />;
              case "choice":
                return (
                  <OnboardingQuestion
                    key={message.id}
                    message={message}
                    lang={state.lang}
                    onSelect={handleChoiceSelect}
                  />
                );
              case "recommendations":
                return (
                  <RecCardGrid
                    key={message.id}
                    message={message}
                    lang={state.lang}
                    onSelectShow={handleSelectShow}
                  />
                );
              default:
                return null;
            }
          })}
          {isLoading && (
            <div className="flex w-full justify-start">
              <div className="rounded-lg border border-border bg-card px-3 py-2 text-sm text-muted-foreground">
                {strings.processing}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <form
        className="flex items-center gap-2 border-t border-border p-3"
        onSubmit={(event) => {
          event.preventDefault();
          handleSend();
        }}
      >
        <Input
          value={inputValue}
          onChange={(event) => setInputValue(event.target.value)}
          placeholder={strings.inputPlaceholder}
          disabled={!canSendMessage}
        />
        <Button type="submit" disabled={!canSendMessage || inputValue.trim().length === 0}>
          {strings.send}
        </Button>
      </form>

      <ShowDetailsModal
        show={selectedShow}
        lang={state.lang}
        onClose={() => setSelectedShow(null)}
      />
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (5 new tests, 58 total)

- [ ] **Step 5: Wire up `App.tsx`**

Replace `frontend/src/App.tsx` with:

```tsx
import { ChatWindow } from "@/components/chat/ChatWindow";

export default function App() {
  return <ChatWindow />;
}
```

Delete `frontend/src/App.css` if the Vite scaffold created one (no longer
referenced - all styling is Tailwind utility classes).

- [ ] **Step 6: Update `index.html`**

Replace `frontend/index.html` with:

```html
<!doctype html>
<html lang="he" dir="rtl">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>CineMatch AI</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 7: Verify `main.tsx` imports the stylesheet**

Open `frontend/src/main.tsx` and confirm it contains `import "./index.css"`
(the default Vite + React + TS template already includes this - if it
imports `"./index.css"` under a different name, rename the import to match
Task 1's `src/index.css`). The file should look like:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

- [ ] **Step 8: Run the full test suite and production build**

Run:
```powershell
cd frontend
npm test
npm run build
```
Expected: all tests pass (58 total), and `npm run build` completes
successfully, producing `frontend/dist/index.html` and `frontend/dist/assets/*`.

- [ ] **Step 9: Manual browser walkthrough (requires the backend running)**

This step exercises the full UX flow described in the spec end-to-end,
against a real backend. It assumes the backend implementation plan
(`docs/superpowers/plans/2026-06-10-cinematch-v2-backend.md`) has already
been executed.

1. In one terminal, start the backend:
   ```powershell
   cd backend
   Copy-Item .env.example .env   # only if not already done
   uvicorn app.main:app --reload
   ```
   Expected: server starts on `http://localhost:8000`; `GET /api/health`
   returns `{"status": "ok"}`.

2. In a second terminal, start the frontend dev server:
   ```powershell
   cd frontend
   npm run dev
   ```
   Expected: Vite prints a local URL, typically `http://localhost:5173`.

3. Open `http://localhost:5173` in a browser and walk through:
   - The opening message appears automatically with two buttons (spec step 1).
   - Click "כן, בוא נתחיל" - the 5 onboarding questions appear one at a time,
     each closing into a badge after you pick an option (spec step 2).
   - After the 5th answer, a "processing" bubble appears, then an intro
     message, 3 recommendation cards with posters/titles/genres/ratings, and
     an outro message (spec steps 3-4).
   - Click a recommendation card - a modal opens showing the full overview,
     rating, the per-show LLM explanation, and (if `TMDB_API_KEY` is set in
     `backend/.env`) a trailer link, cast, and watch providers (spec step 4).
     Close the modal.
   - In the chat input, type a free-text message (e.g., "תמליץ לי על עוד
     משהו דומה") and send it - the bot replies, and may show new
     recommendation cards (spec steps 5-6).
   - Click the language toggle button in the header - the UI strings (input
     placeholder, button labels) switch to English; send another message in
     English and confirm the bot replies in English (spec i18n section).
   - Reload the page (F5) - the conversation persists exactly as it was
     (spec step 7, `localStorage`).
   - Open the site in a different browser (or an incognito window) - it
     starts a fresh conversation from the opening message.

   If every bullet above works, the application is complete and runnable
   end-to-end.

- [ ] **Step 10: Commit**

```bash
git add frontend/src/components/chat/ChatWindow.tsx frontend/src/components/chat/ChatWindow.test.tsx frontend/src/App.tsx frontend/index.html frontend/src/main.tsx
git commit -m "Add ChatWindow, wire up App, and verify full chat flow end-to-end"
```

---
