"""
Backfill EMPTY catalog overviews from OUR primary TMDB source (tvs.csv).

Dev-only, run once at prep time:
    python backend/app/data_prep/enrich_catalog.py [path/to/tvs.csv]

About 2,855 of the 11,013 catalog rows have an empty `overview`, including every
famous show (Breaking Bad, Game of Thrones, Stranger Things, The Office,
Euphoria, BoJack Horseman). Their 384-dim text embeddings are therefore
meaningless. tvs.csv (our Assignment 1 primary source, 152,970 rows) carries
overviews for ~85k titles and can fill ~80% of the gap incl. every famous show.

This fills ONLY empty overviews, IN PLACE, preserving the catalog row count, row
ORDER, and the `title` column byte-for-byte. That invariant is critical:
embeddings.npy is positional (row i <-> embeddings[i]) and cluster_labels.parquet
joins on `title`, so any reorder/rename would silently break both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CATALOG_PATH = DATA_DIR / "catalog.parquet"

# tvs.csv lives outside the repo (our Assignment 1 source data).
DEFAULT_TVS_PATH = (
    Path(__file__).resolve().parents[4] / "תרגיל 1" / "tvs.csv"
)

# Famous shows we expect to gain a synopsis (sanity spot-check).
_SPOT_CHECK = [
    "Breaking Bad",
    "Game of Thrones",
    "Stranger Things",
    "The Office",
    "Euphoria",
    "BoJack Horseman",
]


def _norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _is_empty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip() == ""


def build_overview_map(tvs_path: Path) -> dict[str, str]:
    """title (lowercased, stripped) -> overview, from tvs.csv.

    Keys come from BOTH `name` and `original_name`. When a title maps to several
    rows (e.g. multiple shows named "The Office"), keep the overview from the row
    with the highest vote_count, i.e. the canonical/most-watched show.
    """
    tvs = pd.read_csv(
        tvs_path,
        usecols=["name", "original_name", "overview", "vote_count"],
        low_memory=False,
    )
    tvs = tvs[~_is_empty(tvs["overview"])].copy()
    tvs["vote_count"] = pd.to_numeric(tvs["vote_count"], errors="coerce").fillna(0.0)
    tvs["overview"] = tvs["overview"].astype(str).str.strip()

    # Long form: one (key, overview, vote_count) row per (name / original_name).
    name_rows = tvs[["name", "overview", "vote_count"]].rename(columns={"name": "key"})
    orig_rows = tvs[["original_name", "overview", "vote_count"]].rename(
        columns={"original_name": "key"}
    )
    long = pd.concat([name_rows, orig_rows], ignore_index=True)
    long["key"] = _norm(long["key"])
    long = long[long["key"] != ""]

    # Highest vote_count wins per key.
    long = long.sort_values("vote_count", ascending=False).drop_duplicates(
        "key", keep="first"
    )
    return dict(zip(long["key"], long["overview"]))


def enrich(tvs_path: Path) -> None:
    catalog = pd.read_parquet(CATALOG_PATH)
    original_titles = catalog["title"].copy()
    original_cols = list(catalog.columns)
    n_rows = len(catalog)

    before_empty = int(_is_empty(catalog["overview"]).sum())
    print(f"catalog rows: {n_rows}")
    print(f"empty overviews before: {before_empty}")

    overview_map = build_overview_map(tvs_path)
    print(f"tvs overview map size: {len(overview_map)}")

    empty_mask = _is_empty(catalog["overview"])
    keys = _norm(catalog.loc[empty_mask, "title"])
    filled = keys.map(overview_map)
    fill_count = int(filled.notna().sum())

    # Write only where we found a match; leave the rest blank.
    catalog.loc[empty_mask, "overview"] = filled.where(filled.notna(), catalog.loc[empty_mask, "overview"])

    after_empty = int(_is_empty(catalog["overview"]).sum())
    coverage = 100.0 * (n_rows - after_empty) / n_rows
    print(f"filled from tvs: {fill_count}")
    print(f"empty overviews after: {after_empty}")
    print(f"overview coverage: {coverage:.1f}%")

    # Spot-check famous shows.
    for title in _SPOT_CHECK:
        row = catalog[catalog["title"] == title]
        ok = (not row.empty) and not bool(_is_empty(row["overview"]).iloc[0])
        print(f"  spot-check {title!r:22} overview filled: {ok}")

    # ---- Invariants (fail loudly before writing) ----------------------------
    assert len(catalog) == n_rows, "row count changed"
    assert list(catalog.columns) == original_cols, "column order changed"
    assert catalog["title"].equals(original_titles), "title column changed"
    assert after_empty <= before_empty, "overview emptiness increased"

    catalog.to_parquet(CATALOG_PATH, index=False)
    print(f"wrote {CATALOG_PATH}")


def main() -> None:
    tvs_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TVS_PATH
    if not tvs_path.exists():
        raise SystemExit(f"tvs.csv not found at {tvs_path}; pass the path as argv[1]")
    enrich(tvs_path)


if __name__ == "__main__":
    main()
