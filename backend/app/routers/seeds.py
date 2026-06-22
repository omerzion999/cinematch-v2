"""GET /api/seeds - recognizable seed shows for the onboarding seed-picker step.

For the chosen genre we return a small set of curated, globally recognizable
shows (backend/app/data/genre_seeds.json) so the user can pick the ones they
love. Picks then drive multi-seed similarity in the recommend router.

The seed cards are returned in the same shape as recommendation cards so the
frontend reuses the RecCard component. Most curated seeds are IMDb-sourced rows
with an empty catalog poster_path, so posters are resolved via the TMDB fallback
(app/poster.py) and the built response is cached per genre to avoid repeat
lookups.
"""

import json
import os
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.catalog_lookup import find_catalog_index
from app.poster import resolve_poster_path

router = APIRouter()

_SEEDS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "genre_seeds.json")

Genre = Literal[
    "drama", "comedy", "action_adventure", "scifi_fantasy", "crime", "animation"
]


class SeedCard(BaseModel):
    title: str
    genres: str
    rating: float
    overview: str
    poster_path: str | None = None
    decade_str: str
    num_seasons: float | None = None


class SeedsResponse(BaseModel):
    genre: str
    seeds: list[SeedCard]


def _load_seed_titles() -> dict[str, list[str]]:
    with open(_SEEDS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _nan_to_none(value):
    return None if pd.isna(value) else value


# Per-genre cache of the built seed cards (catalog + posters are static).
_CACHE: dict[str, list[SeedCard]] = {}


def _build_seed_cards(catalog: pd.DataFrame, genre: str) -> list[SeedCard]:
    if genre in _CACHE:
        return _CACHE[genre]

    titles = _load_seed_titles().get(genre, [])
    cards: list[SeedCard] = []
    for title in titles:
        idx = find_catalog_index(catalog, title)
        if idx is None:
            continue
        row = catalog.iloc[idx]
        pick = row.to_dict()
        cards.append(
            SeedCard(
                title=row["title"],
                genres=row["genres"],
                rating=0.0 if pd.isna(row["rating"]) else float(row["rating"]),
                overview=str(row["overview"] or ""),
                poster_path=resolve_poster_path(pick),
                decade_str=row["decade_str"],
                num_seasons=_nan_to_none(row.get("num_seasons")),
            )
        )
    _CACHE[genre] = cards
    return cards


@router.get("/api/seeds", response_model=SeedsResponse)
def seeds(request: Request, genre: Genre, lang: Literal["he", "en"] = "he") -> SeedsResponse:
    state = request.app.state.cinematch
    return SeedsResponse(genre=genre, seeds=_build_seed_cards(state["catalog"], genre))
