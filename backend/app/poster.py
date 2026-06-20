"""
Poster path resolution, shared by the recommend and chat routers.

Most catalog rows carry a TMDB poster_path, but about a quarter do not. For
those we do a best-effort TMDB title lookup so cards still show artwork. Missing
keys / failed lookups degrade silently to None (the UI shows a title tile).
"""

import pandas as pd

from app.agent.tmdb import search_tv_show


def clean_poster_path(value) -> str | None:
    """Normalize a catalog/TMDB poster_path: NaN and empty strings become None."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def resolve_poster_path(pick: dict) -> str | None:
    """Use the catalog's poster_path if present, otherwise look it up on TMDB."""
    poster_path = clean_poster_path(pick.get("poster_path"))
    if poster_path:
        return poster_path

    year = pick.get("start_year")
    year = None if pd.isna(year) else year
    result = search_tv_show(pick["title"], year=int(year) if year else None)
    if result:
        return clean_poster_path(result.get("poster_path"))
    return None
