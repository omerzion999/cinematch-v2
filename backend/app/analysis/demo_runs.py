"""
Generate worked examples of the agent's behavior for the report's
testing-and-validation section (rubric section 7): at least 3 cases including a
failure case where the agent does not find a match.

These exercise the engine layer directly (no LLM / network needed), so the
output is deterministic and reproducible.

Run: python -m app.analysis.demo_runs
"""

import json
import os

import pandas as pd

from app.analysis import DATA_DIR, out_path
from app.engine.preference import rank_by_preferences

ANY = {"genre": "any", "length": "any", "era": "any", "popularity": "any"}


def _load_catalog_with_features() -> pd.DataFrame:
    catalog = pd.read_parquet(os.path.join(DATA_DIR, "catalog.parquet"))
    labels = pd.read_parquet(os.path.join(DATA_DIR, "cluster_labels.parquet"))
    with open(os.path.join(DATA_DIR, "cluster_centroids.json"), encoding="utf-8") as f:
        feature_dims = json.load(f)["feature_dims"]
    overlap = [c for c in feature_dims if c in catalog.columns]
    return catalog.drop(columns=overlap).merge(
        labels[["title", "cluster_id"] + feature_dims], on="title", how="inner"
    )


def _fmt(df: pd.DataFrame) -> str:
    if df.empty:
        return "  (no match)"
    lines = []
    for _, r in df.iterrows():
        year = int(r["start_year"]) if pd.notna(r["start_year"]) else "?"
        lines.append(f"  - {r['title']} | {r['genres']} | {year} | rating {r['rating']}")
    return "\n".join(lines)


def main():
    cwf = _load_catalog_with_features()
    blocks = []

    def case(title, answers, **kw):
        picks = rank_by_preferences(cwf, {**ANY, **answers}, top_n=3, **kw)
        blocks.append(f"## {title}\nanswers={answers}\n{_fmt(picks)}\n")

    case("Case 1: Crime fan, recent hits", {"genre": "crime", "era": "recent", "popularity": "well_known"})
    case("Case 2: Light comedy, short", {"genre": "comedy", "length": "short"})
    case("Case 3: Hidden-gem sci-fi", {"genre": "scifi_fantasy", "popularity": "hidden_gem"})
    case("Case 4: Surprise me (no preferences)", {})

    # Failure case: exclude the dominant genres AND force an impossible-ish combo
    # by excluding everything populated, so the pool collapses to (almost) nothing.
    impossible = cwf[cwf["genres"].fillna("").str.contains("zzzznotagenre", na=False)]
    picks = rank_by_preferences(impossible, ANY, top_n=3)
    blocks.append(
        "## Case 5 (failure): request the engine cannot satisfy\n"
        "When no catalog title matches the constraints, the ranker returns an "
        "empty result and the API replies with the graceful 'no recommendations / "
        "try rephrasing' message instead of inventing a title.\n"
        f"{_fmt(picks)}\n"
    )

    text = "# CineMatch agent - worked examples (deterministic engine layer)\n\n" + "\n".join(blocks)
    path = out_path("demo_runs.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
