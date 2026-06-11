"""
Live TMDB API v3 lookups for the show-details popup (routers/show.py).

Requires TMDB_API_KEY in the environment. If the key is missing, or any
request fails, both functions return None and the caller falls back to
catalog-only data (no error surfaced to the user, per the design spec).
"""

import os

import requests

TMDB_BASE_URL = "https://api.themoviedb.org/3"
_TIMEOUT = 5


def search_tv_show(title: str, year: int | None = None) -> dict | None:
    """
    Searches TMDB for a TV show by title (and optional first-air-date year).
    Returns the best-matching result dict from /search/tv, or None if no
    API key is configured, the request fails, or there are no results.
    """
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        return None

    params = {"api_key": api_key, "query": title}
    if year:
        params["first_air_date_year"] = year

    try:
        resp = requests.get(f"{TMDB_BASE_URL}/search/tv", params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception:
        return None

    if not results:
        return None
    return results[0]


def get_tv_show_details(tmdb_id: int) -> dict | None:
    """
    Fetches full details for a TMDB TV show id, including videos, credits,
    and watch providers via append_to_response. Returns None on any failure
    or if no API key is configured.
    """
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        return None

    params = {"api_key": api_key, "append_to_response": "videos,credits,watch/providers"}
    try:
        resp = requests.get(f"{TMDB_BASE_URL}/tv/{tmdb_id}", params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None
