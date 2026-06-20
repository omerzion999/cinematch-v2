"""GET /api/show/{title} - full catalog details + best-effort TMDB lookup."""

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.agent.tmdb import get_tv_show_details, search_tv_show
from app.catalog_lookup import find_catalog_index
from app.i18n import t

router = APIRouter()


class ShowDetails(BaseModel):
    title: str
    genres: str
    rating: float
    overview: str
    poster_path: str | None = None
    decade_str: str
    start_year: int | None = None
    end_year: int | None = None
    num_seasons: float | None = None
    num_episodes: float | None = None
    language: str | None = None
    votes: float | None = None
    popularity: float | None = None
    binge_fit_score: float
    trailer_url: str | None = None
    cast: list[str] = []
    watch_providers: list[str] = []


def _nan_to_none(value):
    return None if pd.isna(value) else value


def _extract_trailer_url(details: dict) -> str | None:
    for video in details.get("videos", {}).get("results", []):
        if video.get("site") == "YouTube" and video.get("type") == "Trailer":
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None


def _extract_cast(details: dict, limit: int = 5) -> list[str]:
    cast = details.get("credits", {}).get("cast", [])
    return [c["name"] for c in cast[:limit]]


def _extract_watch_providers(details: dict, region: str = "US") -> list[str]:
    providers = details.get("watch/providers", {}).get("results", {}).get(region, {})
    return [p["provider_name"] for p in providers.get("flatrate", [])]


@router.get("/api/show/{title}", response_model=ShowDetails)
def get_show(title: str, request: Request, lang: str = "he") -> ShowDetails:
    state = request.app.state.cinematch
    catalog = state["catalog"]

    idx = find_catalog_index(catalog, title)
    if idx is None:
        raise HTTPException(status_code=404, detail=t("show_not_found", lang))

    row = catalog.iloc[idx]
    details = ShowDetails(
        title=row["title"],
        genres=row["genres"],
        rating=0.0 if pd.isna(row["rating"]) else float(row["rating"]),
        overview=row["overview"],
        poster_path=_nan_to_none(row.get("poster_path")),
        decade_str=row["decade_str"],
        start_year=_nan_to_none(row.get("start_year")),
        end_year=_nan_to_none(row.get("end_year")),
        num_seasons=_nan_to_none(row.get("num_seasons")),
        num_episodes=_nan_to_none(row.get("num_episodes")),
        language=_nan_to_none(row.get("language")),
        votes=_nan_to_none(row.get("votes")),
        popularity=_nan_to_none(row.get("popularity")),
        binge_fit_score=float(row["binge_fit_score"]),
    )

    year = _nan_to_none(row.get("start_year"))
    tmdb_result = search_tv_show(row["title"], year=int(year) if year else None)
    if tmdb_result:
        tmdb_details = get_tv_show_details(tmdb_result["id"])
        if tmdb_details:
            details.trailer_url = _extract_trailer_url(tmdb_details)
            details.cast = _extract_cast(tmdb_details)
            details.watch_providers = _extract_watch_providers(tmdb_details)

    return details
