import type {
  ChatRequest,
  ChatResponse,
  Lang,
  RecommendRequest,
  RecommendResponse,
  ShowDetails,
  TranslateRequest,
  TranslateResponse,
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

export function postTranslate(request: TranslateRequest): Promise<TranslateResponse> {
  return postJson<TranslateResponse>("/translate", request);
}

export async function getShow(title: string, lang: Lang): Promise<ShowDetails> {
  const url = `${API_BASE}/show/${encodeURIComponent(title)}?lang=${lang}`;
  const res = await fetch(url);
  if (!res.ok) {
    let detail = `GET ${url} failed with status ${res.status}`;
    const body: unknown = await res.json().catch(() => null);
    if (body !== null && typeof body === "object" && "detail" in body) {
      const detailValue = (body as { detail: unknown }).detail;
      if (typeof detailValue === "string") {
        detail = detailValue;
      }
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<ShowDetails>;
}
