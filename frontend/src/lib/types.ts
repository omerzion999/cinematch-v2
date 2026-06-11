export type Lang = "he" | "en";

export interface OnboardingAnswers {
  genre: "drama" | "comedy" | "action_adventure" | "scifi_fantasy" | "crime" | "animation" | "romance" | "any";
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
