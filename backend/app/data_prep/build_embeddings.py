"""
Regenerate embeddings.npy from the (enriched) catalog overviews.

Dev-only, run once at prep time AFTER enrich_catalog.py:
    python backend/app/data_prep/build_embeddings.py

sentence-transformers is a DEV-ONLY dependency: it is installed into the local
venv but is NOT in backend/requirements.txt and must never reach Render. The app
only ever reads the committed embeddings.npy at runtime.

Embeds each row's `overview` (fallback to `title` when overview is still empty)
with the 384-dim multilingual model paraphrase-multilingual-MiniLM-L12-v2 (the
same model/dims as the existing file, and Hebrew-capable). Output is L2-normalized
float32 with row order identical to catalog.parquet, so the positional mapping
(catalog row i <-> embeddings[i]) used across the engine stays intact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CATALOG_PATH = DATA_DIR / "catalog.parquet"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384


def _embed_text(catalog: pd.DataFrame) -> list[str]:
    overview = catalog["overview"].fillna("").astype(str).str.strip()
    title = catalog["title"].fillna("").astype(str).str.strip()
    # Fallback to title for the ~550 rows still without a synopsis, so no row is
    # embedded as an empty string (which would collapse to a near-zero vector).
    return overview.where(overview != "", title).tolist()


def build() -> None:
    from sentence_transformers import SentenceTransformer

    catalog = pd.read_parquet(CATALOG_PATH)
    texts = _embed_text(catalog)
    print(f"catalog rows: {len(catalog)}")
    print(f"loading model {MODEL_NAME} ...")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    # ---- Invariants (must match tests/test_data_files.py) -------------------
    assert embeddings.shape == (len(catalog), EMBED_DIM), embeddings.shape
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (float(norms.min()), float(norms.max()))

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"wrote {EMBEDDINGS_PATH}  shape={embeddings.shape}  dtype={embeddings.dtype}")
    print(f"norms min/max: {norms.min():.4f}/{norms.max():.4f}")


if __name__ == "__main__":
    build()
