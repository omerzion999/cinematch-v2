"""
Offline K-Means training script for the "taste profile" clustering layer.

Run with:  python -m app.clustering.train_clusters

Reads app/data/catalog.parquet, builds the 14-dim feature matrix
(see clustering/features.py), fits K-Means for several candidate k values,
picks the k with the highest silhouette score, and writes three artifacts
into app/data/:

  - cluster_labels.parquet : title, cluster_id, + the 14 feature columns
                              (the runtime recommender uses this directly,
                              so it never has to recompute the feature vector)
  - cluster_centroids.json : k, feature_dims (ordered), centroids (k x 14)
  - cluster_profiles.json  : per-cluster bilingual description used in bot
                              messages ("your taste profile is: ...")
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.clustering.features import FEATURE_DIMS, GENRE_DIMS, build_cluster_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CANDIDATE_KS = range(6, 13)
SILHOUETTE_SAMPLE_SIZE = 2000
RANDOM_STATE = 42

GENRE_LABELS = {
    "Drama": {"he": "דרמה", "en": "Drama"},
    "Comedy": {"he": "קומדיה", "en": "Comedy"},
    "Animation": {"he": "אנימציה", "en": "Animation"},
    "Crime": {"he": "פשע", "en": "Crime"},
    "Action & Adventure": {"he": "אקשן והרפתקאות", "en": "Action & Adventure"},
    "Sci-Fi & Fantasy": {"he": 'מד"ב ופנטזיה', "en": "Sci-Fi & Fantasy"},
    "Mystery": {"he": "מתח ותעלומות", "en": "Mystery"},
    "Documentary": {"he": "דוקומנטרי", "en": "Documentary"},
    "Family": {"he": "משפחה", "en": "Family"},
    "Romance": {"he": "רומנטיקה", "en": "Romance"},
}


def choose_k(X: np.ndarray, k_range=CANDIDATE_KS) -> tuple[int, dict]:
    """Fit K-Means for each k in k_range, return (best_k, {k: silhouette_score})."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(
            X, labels,
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(X)),
            random_state=RANDOM_STATE,
        )
        scores[k] = float(score)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)


def build_cluster_profiles(features: pd.DataFrame, catalog: pd.DataFrame, labels: np.ndarray) -> dict:
    """
    features: output of build_cluster_features (title + FEATURE_DIMS), row order
              must match `catalog` and `labels`.
    catalog:  full catalog DataFrame (for start_year, rating, binge_fit_score)
    labels:   cluster_id per row, same order as features/catalog
    """
    df = features.copy()
    df["cluster_id"] = labels
    df["start_year"] = catalog["start_year"].values
    df["rating"] = catalog["rating"].values
    df["binge_fit_score"] = catalog["binge_fit_score"].values

    profiles = {}
    for cluster_id, group in df.groupby("cluster_id"):
        genre_means = {g: float(group[f"genre:{g}"].mean()) for g in GENRE_DIMS}
        top_genres = sorted(genre_means, key=genre_means.get, reverse=True)[:2]

        avg_year = float(group["start_year"].mean())
        if avg_year >= 2017:
            era_he, era_en = "מהשנים האחרונות", "from recent years"
        elif avg_year < 2005:
            era_he, era_en = "קלאסיות", "classic"
        else:
            era_he, era_en = "", ""

        label_he = " ו".join(GENRE_LABELS[g]["he"] for g in top_genres)
        label_en = " & ".join(GENRE_LABELS[g]["en"] for g in top_genres)
        if era_he:
            label_he = f"{label_he} {era_he}"
        if era_en:
            label_en = f"{label_en} {era_en}"

        profiles[str(int(cluster_id))] = {
            "label_he": label_he.strip(),
            "label_en": label_en.strip(),
            "size": int(len(group)),
            "top_genres": top_genres,
            "avg_rating": round(float(group["rating"].mean()), 2),
            "avg_start_year": round(avg_year, 1),
            "avg_binge_fit_score": round(float(group["binge_fit_score"].mean()), 2),
        }
    return profiles


def main():
    catalog = pd.read_parquet(os.path.join(DATA_DIR, "catalog.parquet"))
    features = build_cluster_features(catalog)
    X = features[FEATURE_DIMS].values.astype(np.float32)

    best_k, silhouette_scores = choose_k(X)
    print(f"Silhouette scores by k: {silhouette_scores}")
    print(f"Chosen k = {best_k}")

    km = fit_kmeans(X, best_k)
    labels = km.labels_

    labels_out = features.copy()
    labels_out["cluster_id"] = labels
    labels_out.to_parquet(os.path.join(DATA_DIR, "cluster_labels.parquet"), index=False)

    centroids_out = {
        "k": int(best_k),
        "feature_dims": FEATURE_DIMS,
        "centroids": km.cluster_centers_.tolist(),
        "silhouette_scores": {str(k): v for k, v in silhouette_scores.items()},
    }
    with open(os.path.join(DATA_DIR, "cluster_centroids.json"), "w", encoding="utf-8") as f:
        json.dump(centroids_out, f, ensure_ascii=False, indent=2)

    profiles = build_cluster_profiles(features, catalog, labels)
    with open(os.path.join(DATA_DIR, "cluster_profiles.json"), "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    print(f"Wrote cluster_labels.parquet, cluster_centroids.json, cluster_profiles.json (k={best_k})")


if __name__ == "__main__":
    main()
