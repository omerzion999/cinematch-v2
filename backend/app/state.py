"""
Loads all shared data/model artifacts once at FastAPI startup.

app/main.py's lifespan handler calls load_state() and stores the result on
app.state.cinematch. Routers read from request.app.state.cinematch.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from app.engine.anomaly import calibrate
from app.engine.cosine import build_numeric_matrix
from app.engine.hybrid import ALPHA, BETA

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_state() -> dict:
    catalog = pd.read_parquet(os.path.join(DATA_DIR, "catalog.parquet"))
    embeddings = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
    numeric_matrix = build_numeric_matrix(catalog)

    cluster_labels = pd.read_parquet(os.path.join(DATA_DIR, "cluster_labels.parquet"))
    with open(os.path.join(DATA_DIR, "cluster_centroids.json"), encoding="utf-8") as f:
        cluster_centroids = json.load(f)
    with open(os.path.join(DATA_DIR, "cluster_profiles.json"), encoding="utf-8") as f:
        cluster_profiles = json.load(f)

    feature_dims = cluster_centroids["feature_dims"]
    overlap = [c for c in feature_dims if c in catalog.columns]
    catalog_with_features = catalog.drop(columns=overlap).merge(
        cluster_labels[["title", "cluster_id"] + feature_dims],
        on="title",
        how="inner",
    )

    anomaly_threshold = _calibrate_anomaly_threshold(catalog, embeddings, numeric_matrix)

    return {
        "catalog": catalog,
        "embeddings": embeddings,
        "numeric_matrix": numeric_matrix,
        "catalog_with_features": catalog_with_features,
        "cluster_centroids": cluster_centroids["centroids"],
        "cluster_profiles": cluster_profiles,
        "feature_dims": feature_dims,
        "anomaly_threshold": anomaly_threshold,
    }


def _calibrate_anomaly_threshold(
    catalog: pd.DataFrame, embeddings: np.ndarray, numeric_matrix: np.ndarray
) -> float:
    """
    Sample 200 catalog rows; for each, compute its best numeric+text hybrid
    match score against the rest of the catalog. Calibrate the anomaly
    threshold at the 8th percentile of that distribution (ALPHA/BETA from
    engine/hybrid.py; GAMMA = 1 - ALPHA - BETA).
    """
    gamma = 1.0 - ALPHA - BETA
    rng = np.random.default_rng(42)
    n = len(catalog)
    sample_size = min(200, n)
    sample_idx = rng.choice(n, sample_size, replace=False)

    best_hybrid_scores = []
    for src_pos in sample_idx:
        n_scores = sk_cosine(numeric_matrix[src_pos:src_pos + 1], numeric_matrix)[0]
        t_scores = embeddings @ embeddings[src_pos]
        full_scores = BETA * n_scores + gamma * t_scores
        full_scores[src_pos] = -1  # exclude self
        best_hybrid_scores.append(float(full_scores.max()))

    return calibrate(np.array(best_hybrid_scores), percentile=8)
