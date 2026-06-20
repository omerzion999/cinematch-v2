"""
Model-comparison evidence for the report (sections 5-6):

  1. Clustering collapse: per-cluster top genres + sizes (shows 5/8 K-Means
     buckets are the same Drama/Comedy blend) and the silhouette scores.
  2. Similarity methods: Spearman correlation between Jaccard (genres/era/lang),
     Cosine-numeric (rating/votes/year/popularity), and Cosine-text (multilingual
     embeddings), over a random sample of title pairs. The low Jaccard-vs-Cosine
     correlation is what justifies the hybrid scorer (matches Assignment 2).

Run: python -m app.analysis.model_eval
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.analysis import DATA_DIR, out_path
from app.engine.cosine import build_numeric_matrix
from app.engine.jaccard import _build_feature_set, jaccard

RANDOM_STATE = 42
N_PAIRS = 5000


def clustering_report() -> pd.DataFrame:
    with open(os.path.join(DATA_DIR, "cluster_profiles.json"), encoding="utf-8") as f:
        profiles = json.load(f)
    rows = []
    for cid, p in sorted(profiles.items(), key=lambda kv: int(kv[0])):
        rows.append({
            "cluster_id": int(cid),
            "size": p["size"],
            "top_genres": " + ".join(p["top_genres"]),
            "avg_rating": p["avg_rating"],
            "avg_start_year": p["avg_start_year"],
            "avg_binge_fit": p["avg_binge_fit_score"],
        })
    return pd.DataFrame(rows)


def silhouette_report() -> pd.DataFrame:
    with open(os.path.join(DATA_DIR, "cluster_centroids.json"), encoding="utf-8") as f:
        cc = json.load(f)
    scores = cc.get("silhouette_scores", {})
    df = pd.DataFrame(
        [{"k": int(k), "silhouette": round(v, 4)} for k, v in scores.items()]
    ).sort_values("k")
    df["chosen"] = df["k"] == cc.get("k")
    return df


def similarity_correlations(catalog: pd.DataFrame) -> pd.DataFrame:
    embeddings = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
    numeric = build_numeric_matrix(catalog).astype(np.float32)
    # L2-normalize numeric rows so a dot product is cosine.
    norms = np.linalg.norm(numeric, axis=1, keepdims=True)
    numeric_n = numeric / np.where(norms == 0, 1, norms)

    feature_sets = [_build_feature_set(row) for _, row in catalog.iterrows()]

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(catalog)
    i = rng.integers(0, n, N_PAIRS)
    j = rng.integers(0, n, N_PAIRS)
    keep = i != j
    i, j = i[keep], j[keep]

    jac = np.array([jaccard(feature_sets[a], feature_sets[b]) for a, b in zip(i, j)])
    cos_text = np.array([float(embeddings[a] @ embeddings[b]) for a, b in zip(i, j)])
    cos_num = np.array([float(numeric_n[a] @ numeric_n[b]) for a, b in zip(i, j)])

    def sp(x, y):
        return round(float(spearmanr(x, y).statistic), 4)

    return pd.DataFrame([
        {"pair": "Jaccard vs Cosine-text", "spearman": sp(jac, cos_text)},
        {"pair": "Jaccard vs Cosine-numeric", "spearman": sp(jac, cos_num)},
        {"pair": "Cosine-text vs Cosine-numeric", "spearman": sp(cos_text, cos_num)},
    ])


def main():
    catalog = pd.read_parquet(os.path.join(DATA_DIR, "catalog.parquet"))

    clusters = clustering_report()
    silhouette = silhouette_report()
    corr = similarity_correlations(catalog)

    clusters.to_csv(out_path("clustering_profiles.csv"), index=False)
    silhouette.to_csv(out_path("clustering_silhouette.csv"), index=False)
    corr.to_csv(out_path("similarity_correlations.csv"), index=False)

    print("K-Means cluster profiles (collapse evidence):")
    print(clusters.to_string(index=False))
    print("\nsilhouette by k:")
    print(silhouette.to_string(index=False))
    print(f"\nsimilarity correlations (n={N_PAIRS} random pairs):")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
