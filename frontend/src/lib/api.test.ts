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
      answers: { genre: "drama", length: "any", era: "any", popularity: "any" },
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

    await expect(getShow("Unknown Show", "he")).rejects.toBeInstanceOf(ApiError);
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
      answers: { genre: "any", length: "any", era: "any", popularity: "any" },
      lang: "he",
    };

    await expect(postRecommend(request)).rejects.toBeInstanceOf(ApiError);
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({}, 500));
    await expect(postRecommend(request)).rejects.toMatchObject({
      name: "ApiError",
      status: 500,
    });
  });

  it("propagates a network-level fetch rejection", async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new TypeError("Failed to fetch"));
    const request: RecommendRequest = {
      answers: { genre: "any", length: "any", era: "any", popularity: "any" },
      lang: "he",
    };

    await expect(postRecommend(request)).rejects.toThrow("Failed to fetch");
  });
});
