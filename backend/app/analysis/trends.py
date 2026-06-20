"""
Trends and patterns in the catalog, for the report:
  - genre share by decade (rising / declining genres)
  - rating by decade
  - Shannon entropy of the genre mix, overall and per decade

Run: python -m app.analysis.trends
"""

import os
from collections import Counter

import numpy as np
import pandas as pd

from app.analysis import DATA_DIR, out_path

TOP_GENRES = [
    "Drama", "Comedy", "Animation", "Crime", "Action & Adventure",
    "Sci-Fi & Fantasy", "Mystery", "Documentary", "Family", "Romance",
]
DECADES = ["1980s", "1990s", "2000s", "2010s", "2020s"]


def _genre_set(s) -> set:
    if not s:
        return set()
    return {g.strip() for g in str(s).split(",") if g.strip()}


def _shannon_entropy(counts) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts if c > 0])
    return float(-(probs * np.log2(probs)).sum())


def genre_share_by_decade(catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for decade in DECADES:
        sub = catalog[catalog["decade_str"] == decade]
        n = len(sub)
        if n == 0:
            continue
        gsets = sub["genres"].fillna("").apply(_genre_set)
        row = {"decade": decade, "titles": n}
        for g in TOP_GENRES:
            row[g] = round(100.0 * gsets.apply(lambda s, g=g: g in s).sum() / n, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def rating_by_decade(catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for decade in DECADES:
        sub = catalog[catalog["decade_str"] == decade]
        if sub.empty:
            continue
        rows.append({
            "decade": decade,
            "titles": len(sub),
            "avg_rating": round(float(sub["rating"].mean()), 2),
            "median_rating": round(float(sub["rating"].median()), 2),
            "avg_binge_fit": round(float(sub["binge_fit_score"].mean()), 2),
        })
    return pd.DataFrame(rows)


def genre_entropy(catalog: pd.DataFrame) -> pd.DataFrame:
    def entropy_for(sub: pd.DataFrame) -> tuple[float, int]:
        c = Counter()
        for s in sub["genres"].fillna(""):
            for g in _genre_set(s):
                c[g] += 1
        return _shannon_entropy(list(c.values())), len(c)

    rows = []
    h_all, k_all = entropy_for(catalog)
    rows.append({"scope": "all", "distinct_genres": k_all, "shannon_entropy_bits": round(h_all, 3)})
    for decade in DECADES:
        sub = catalog[catalog["decade_str"] == decade]
        if sub.empty:
            continue
        h, k = entropy_for(sub)
        rows.append({"scope": decade, "distinct_genres": k, "shannon_entropy_bits": round(h, 3)})
    return pd.DataFrame(rows)


def main():
    catalog = pd.read_parquet(os.path.join(DATA_DIR, "catalog.parquet"))

    share = genre_share_by_decade(catalog)
    rating = rating_by_decade(catalog)
    entropy = genre_entropy(catalog)

    share.to_csv(out_path("genre_share_by_decade.csv"), index=False)
    rating.to_csv(out_path("rating_by_decade.csv"), index=False)
    entropy.to_csv(out_path("genre_entropy.csv"), index=False)

    print("genre share by decade (% of titles tagged):")
    print(share.to_string(index=False))
    print("\nrating by decade:")
    print(rating.to_string(index=False))
    print("\ngenre entropy:")
    print(entropy.to_string(index=False))

    # Rising / declining: change in share from 2000s to 2020s.
    if {"2000s", "2020s"}.issubset(set(share["decade"])):
        a = share[share["decade"] == "2000s"].iloc[0]
        b = share[share["decade"] == "2020s"].iloc[0]
        delta = {g: round(b[g] - a[g], 1) for g in TOP_GENRES}
        trend = pd.DataFrame(
            sorted(delta.items(), key=lambda x: x[1], reverse=True),
            columns=["genre", "share_change_2000s_to_2020s_pct"],
        )
        trend.to_csv(out_path("genre_trend_2000s_to_2020s.csv"), index=False)
        print("\nrising / declining genres (2000s -> 2020s, percentage points):")
        print(trend.to_string(index=False))


if __name__ == "__main__":
    main()
