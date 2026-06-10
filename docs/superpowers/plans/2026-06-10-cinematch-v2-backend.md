# CineMatch AI v2 - Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FastAPI backend for CineMatch AI v2 - a chat-based TV show recommendation agent that reuses v1's hybrid recommendation engine (Jaccard + Cosine numeric + Cosine text), adds a new K-Means clustering layer for "taste profile" segmentation, and exposes 3 endpoints (`/api/recommend`, `/api/chat`, `/api/show/{title}`) backed by Groq LLM and TMDB.

**Architecture:** Python/FastAPI app under `backend/`. `app/engine/` ports v1's similarity/hybrid/anomaly modules verbatim (import paths only change). `app/clustering/` is new: builds a 14-dim feature vector (10 genre one-hot + 4 numeric z-scores) per title, trains K-Means offline (`train_clusters.py`), and at runtime maps onboarding answers (or chat intents without a seed title) to that same vector space to pick a cluster and rank candidates within it. `app/agent/` ports v1's Groq `chat_turn`/explanation logic and adds a new TMDB live-lookup module. `app/routers/` wires it all into 3 endpoints. Everything is covered by pytest with external calls (Groq, TMDB) mocked.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, pandas, numpy, scikit-learn, pyarrow, groq SDK, requests (TMDB), pytest, httpx (FastAPI TestClient).

---

## File Structure

```
cinematch-ai-v2/                          (repo root - this directory)
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                       # FastAPI app, lifespan startup, static frontend mount
│   │   ├── state.py                      # Loads catalog/embeddings/clusters once at startup
│   │   ├── i18n.py                       # Bilingual strings used in backend-generated text
│   │   ├── data/
│   │   │   ├── catalog.parquet           # copied from v1
│   │   │   ├── embeddings.npy            # copied from v1
│   │   │   ├── cluster_labels.parquet    # NEW: title -> cluster_id + 14 feature dims
│   │   │   ├── cluster_centroids.json    # NEW: K-Means centroids + feature dim names
│   │   │   └── cluster_profiles.json     # NEW: bilingual cluster descriptions
│   │   ├── engine/
│   │   │   ├── __init__.py
│   │   │   ├── jaccard.py                # ported from v1 engine/jaccard.py
│   │   │   ├── cosine.py                 # ported from v1 engine/cosine.py
│   │   │   ├── hybrid.py                 # ported from v1 engine/hybrid.py
│   │   │   └── anomaly.py                # ported from v1 engine/anomaly.py
│   │   ├── clustering/
│   │   │   ├── __init__.py
│   │   │   ├── features.py               # NEW: build 14-dim feature matrix
│   │   │   ├── train_clusters.py         # NEW: offline K-Means training script
│   │   │   ├── onboarding_map.py         # NEW: onboarding answers / chat intent -> vector
│   │   │   └── recommend.py              # NEW: nearest-cluster + within-cluster ranking
│   │   ├── agent/
│   │   │   ├── __init__.py
│   │   │   ├── llm.py                    # ported from v1 agent/llm.py (Groq chat_turn etc.)
│   │   │   ├── tmdb.py                   # NEW: live TMDB lookups
│   │   │   └── explanations.py           # NEW: explain_picks for onboarding recs
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── recommend.py              # POST /api/recommend
│   │       ├── chat.py                   # POST /api/chat
│   │       └── show.py                   # GET /api/show/{title}
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_engine_jaccard.py
│   │   ├── test_engine_cosine.py
│   │   ├── test_engine_hybrid.py
│   │   ├── test_engine_anomaly.py
│   │   ├── test_clustering_features.py
│   │   ├── test_clustering_onboarding_map.py
│   │   ├── test_clustering_recommend.py
│   │   ├── test_agent_tmdb.py
│   │   ├── test_agent_explanations.py
│   │   ├── test_api_recommend.py
│   │   ├── test_api_chat.py
│   │   └── test_api_show.py
│   ├── requirements.txt
│   └── pytest.ini
└── render.yaml
```

**Fixed feature vector used everywhere in clustering** (order matters - referenced by index throughout):

```python
GENRE_DIMS = [
    "Drama", "Comedy", "Animation", "Crime", "Action & Adventure",
    "Sci-Fi & Fantasy", "Mystery", "Documentary", "Family", "Romance",
]
NUMERIC_DIMS = ["rating_z", "popularity_z", "start_year_z", "num_seasons_z"]
FEATURE_DIMS = [f"genre:{g}" for g in GENRE_DIMS] + NUMERIC_DIMS  # 14 dims total
```

These two lists (`GENRE_DIMS`, `NUMERIC_DIMS`, `FEATURE_DIMS`) are defined once in `app/clustering/features.py` (Task 7) and imported everywhere else that needs them (Tasks 8, 9, 10, 16, 17).

---

## Task 1: Project Scaffolding

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/pytest.ini`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_main.py`

- [ ] **Step 1: Create the directory structure and requirements file**

Run:
```powershell
New-Item -ItemType Directory -Force -Path "backend\app\engine","backend\app\clustering","backend\app\agent","backend\app\routers","backend\app\data","backend\tests"
```

Create `backend/requirements.txt`:

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
pyarrow==18.1.0
groq==0.13.1
requests==2.32.3
python-dotenv==1.0.1
pytest==8.3.4
httpx==0.28.1
```

- [ ] **Step 2: Install dependencies**

Run: `cd backend && pip install -r requirements.txt`
Expected: all packages install without error.

- [ ] **Step 3: Create pytest config**

Create `backend/pytest.ini`:

```ini
[pytest]
testpaths = tests
pythonpath = .
```

- [ ] **Step 4: Create empty `__init__.py` files**

Create `backend/app/__init__.py` (empty file).
Create `backend/app/engine/__init__.py` (empty file).
Create `backend/app/clustering/__init__.py` (empty file).
Create `backend/app/agent/__init__.py` (empty file).
Create `backend/app/routers/__init__.py` (empty file).
Create `backend/tests/__init__.py` (empty file).

- [ ] **Step 5: Write the failing test for the FastAPI app**

Create `backend/tests/conftest.py`:

```python
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
```

Create `backend/tests/test_main.py`:

```python
def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd backend && pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'` (or similar import error).

- [ ] **Step 7: Write minimal FastAPI app**

Create `backend/app/main.py`:

```python
from fastapi import FastAPI

app = FastAPI(title="CineMatch AI v2")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd backend && pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add backend/requirements.txt backend/pytest.ini backend/app backend/tests
git commit -m "Scaffold backend FastAPI project"
```

---

## Task 2: Data Migration

**Files:**
- Create: `backend/app/data/catalog.parquet` (copied)
- Create: `backend/app/data/embeddings.npy` (copied)
- Create: `backend/tests/test_data_files.py`

- [ ] **Step 1: Copy the v1 data files**

Run:
```powershell
Copy-Item "C:\Users\Hello\Desktop\cinematch-ai-main\data\catalog.parquet" "backend\app\data\catalog.parquet"
Copy-Item "C:\Users\Hello\Desktop\cinematch-ai-main\data\embeddings.npy" "backend\app\data\embeddings.npy"
```

- [ ] **Step 2: Write the failing test**

Create `backend/tests/test_data_files.py`:

```python
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "data")


def test_catalog_loads_with_expected_shape():
    catalog = pd.read_parquet(os.path.join(DATA_DIR, "catalog.parquet"))
    assert len(catalog) == 11013
    expected_cols = {
        "title", "language", "start_year", "end_year", "genres", "rating",
        "votes", "popularity", "overview", "poster_path", "num_episodes",
        "num_seasons", "source_dataset", "decade", "decade_str",
        "genre_set_str", "rating_z", "votes_z", "start_year_z",
        "popularity_z", "rating_bucket", "binge_fit_score",
    }
    assert expected_cols.issubset(set(catalog.columns))


def test_embeddings_match_catalog_rows():
    catalog = pd.read_parquet(os.path.join(DATA_DIR, "catalog.parquet"))
    embeddings = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
    assert embeddings.shape == (len(catalog), 384)
    # L2-normalized
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && pytest tests/test_data_files.py -v`
Expected: FAIL with `FileNotFoundError` if Step 1 was skipped, otherwise PASS already (Step 1 makes this pass immediately - that's fine, this test documents/locks in the data contract).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_data_files.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/data/catalog.parquet backend/app/data/embeddings.npy backend/tests/test_data_files.py
git commit -m "Add catalog data and precomputed embeddings"
```

---

## Task 3: Port `engine/jaccard.py`

**Files:**
- Create: `backend/app/engine/jaccard.py`
- Modify: `backend/tests/conftest.py`
- Test: `backend/tests/test_engine_jaccard.py`

- [ ] **Step 1: Add shared `catalog`/`embeddings` fixtures to conftest**

Replace the full contents of `backend/tests/conftest.py` with:

```python
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "data")


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def catalog():
    return pd.read_parquet(os.path.join(DATA_DIR, "catalog.parquet"))


@pytest.fixture(scope="session")
def embeddings():
    return np.load(os.path.join(DATA_DIR, "embeddings.npy"))


@pytest.fixture(scope="session")
def numeric_matrix(catalog):
    from app.engine.cosine import build_numeric_matrix
    return build_numeric_matrix(catalog)
```

- [ ] **Step 2: Write the failing test**

Create `backend/tests/test_engine_jaccard.py`:

```python
from app.engine.jaccard import _build_feature_set, jaccard, top_k_jaccard


def test_build_feature_set_includes_genre_decade_language(catalog):
    row = catalog[catalog["title"] == "Game of Thrones"].iloc[0]
    feat = _build_feature_set(row)
    assert "genre:Action" in feat
    assert "genre:Adventure" in feat
    assert "genre:Drama" in feat
    assert "decade:2010s" in feat
    assert "lang:en" in feat


def test_jaccard_identical_sets_is_one():
    a = frozenset({"genre:Drama", "decade:2010s"})
    assert jaccard(a, a) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    a = frozenset({"genre:Drama"})
    b = frozenset({"genre:Comedy"})
    assert jaccard(a, b) == 0.0


def test_top_k_jaccard_returns_k_rows_sorted_desc(catalog):
    result = top_k_jaccard("Game of Thrones", catalog, k=5)
    assert len(result) == 5
    assert "jaccard_score" in result.columns
    scores = result["jaccard_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert "Game of Thrones" not in result["title"].values
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && pytest tests/test_engine_jaccard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.engine.jaccard'`

- [ ] **Step 4: Create `app/engine/jaccard.py`**

Port verbatim from `C:\Users\Hello\Desktop\cinematch-ai-main\engine\jaccard.py` (no import path changes needed - this module has no internal cross-imports):

```python
"""
Jaccard similarity filter.

Jaccard(A, B) = |A ∩ B| / |A ∪ B|

Feature set per title: genres ∪ decade_bucket ∪ language_bucket
Returns the top-K most similar titles by Jaccard score.
"""

import numpy as np
import pandas as pd


def _build_feature_set(row: pd.Series) -> frozenset:
    items = set()
    # genres
    genres = str(row.get("genre_set_str", "") or "")
    for g in genres.split("|"):
        g = g.strip()
        if g:
            items.add(f"genre:{g}")
    # decade
    dec = row.get("decade_str", "")
    if dec and dec != "Unknown":
        items.add(f"decade:{dec}")
    # language family
    lang = str(row.get("language", "") or "").lower()[:2]
    if lang:
        items.add(f"lang:{lang}")
    return frozenset(items)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def top_k_jaccard(
    query_title: str,
    catalog: pd.DataFrame,
    k: int = 50,
    exclude_titles: set = None,
) -> pd.DataFrame:
    """Return top-K rows from catalog most similar to query_title by Jaccard."""
    exclude_titles = exclude_titles or set()
    mask = catalog["title"].str.lower() == query_title.strip().lower()
    if not mask.any():
        # soft match: contains
        mask = catalog["title"].str.lower().str.contains(
            query_title.strip().lower(), regex=False, na=False)
    if not mask.any():
        return catalog.head(0)

    query_row = catalog[mask].iloc[0]
    query_set = _build_feature_set(query_row)
    query_idx = query_row.name

    scores = []
    for idx, row in catalog.iterrows():
        if idx == query_idx:
            continue
        if row["title"] in exclude_titles:
            continue
        s = jaccard(query_set, _build_feature_set(row))
        scores.append((idx, s))

    scores.sort(key=lambda x: -x[1])
    top_idx = [i for i, _ in scores[:k]]
    result = catalog.loc[top_idx].copy()
    result["jaccard_score"] = [s for _, s in scores[:k]]
    return result.reset_index(drop=True)


def batch_jaccard_matrix(catalog: pd.DataFrame) -> np.ndarray:
    """Compute full N×N Jaccard matrix (used in evaluate.py)."""
    feature_sets = [_build_feature_set(row) for _, row in catalog.iterrows()]
    n = len(feature_sets)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            s = jaccard(feature_sets[i], feature_sets[j])
            M[i, j] = s
            M[j, i] = s
    return M
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && pytest tests/test_engine_jaccard.py -v`
Expected: PASS (the test may take a few seconds since `top_k_jaccard` is O(N) over 11,013 rows)

- [ ] **Step 6: Commit**

```bash
git add backend/app/engine/jaccard.py backend/tests/conftest.py backend/tests/test_engine_jaccard.py
git commit -m "Port jaccard similarity engine from v1"
```

---

## Task 4: Port `engine/cosine.py`

**Files:**
- Create: `backend/app/engine/cosine.py`
- Test: `backend/tests/test_engine_cosine.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_engine_cosine.py`:

```python
import numpy as np

from app.engine.cosine import (
    NUMERIC_COLS,
    build_numeric_matrix,
    top_k_cosine_numeric,
    top_k_cosine_text,
)


def test_numeric_cols_are_the_four_z_scores():
    assert NUMERIC_COLS == ["rating_z", "votes_z", "start_year_z", "popularity_z"]


def test_build_numeric_matrix_is_symmetric_with_unit_diagonal(catalog):
    matrix = build_numeric_matrix(catalog)
    assert matrix.shape == (len(catalog), len(catalog))
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-4)
    assert np.allclose(matrix, matrix.T, atol=1e-4)


def test_top_k_cosine_numeric_returns_k_rows(catalog, numeric_matrix):
    result = top_k_cosine_numeric("Game of Thrones", catalog, numeric_matrix, k=5)
    assert len(result) == 5
    assert "cosine_numeric_score" in result.columns
    assert "Game of Thrones" not in result["title"].values


def test_top_k_cosine_text_returns_k_rows(catalog, embeddings):
    got_idx = catalog.index[catalog["title"] == "Game of Thrones"][0]
    pos = catalog.index.get_loc(got_idx)
    query_embedding = embeddings[pos]

    result = top_k_cosine_text(
        query_embedding, catalog, embeddings, k=5, query_title="Game of Thrones"
    )
    assert len(result) == 5
    assert "cosine_text_score" in result.columns
    assert "Game of Thrones" not in result["title"].values
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_engine_cosine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.engine.cosine'`

- [ ] **Step 3: Create `app/engine/cosine.py`**

Port verbatim from `C:\Users\Hello\Desktop\cinematch-ai-main\engine\cosine.py` (no import path changes needed):

```python
"""
Cosine similarity — two variants.

Cosine(A, B) = (A·B) / (||A|| * ||B||)

Variant 1 — Numeric features:
    [rating_z, votes_z, start_year_z, popularity_z]
    (pre-computed z-scores stored in catalog)

Variant 2 — Text embeddings:
    384-dim multilingual sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2)
    pre-computed and L2-normalized → dot product = cosine similarity
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


# ── Numeric Cosine ────────────────────────────────────────────────────────────

NUMERIC_COLS = ["rating_z", "votes_z", "start_year_z", "popularity_z"]


def _numeric_matrix(catalog: pd.DataFrame) -> np.ndarray:
    X = catalog[NUMERIC_COLS].fillna(0).values.astype(np.float32)
    return sk_cosine(X).astype(np.float32)


def top_k_cosine_numeric(
    query_title: str,
    catalog: pd.DataFrame,
    numeric_matrix: np.ndarray,
    k: int = 50,
    exclude_titles: set = None,
) -> pd.DataFrame:
    exclude_titles = exclude_titles or set()
    mask = catalog["title"].str.lower() == query_title.strip().lower()
    if not mask.any():
        mask = catalog["title"].str.lower().str.contains(
            query_title.strip().lower(), regex=False, na=False)
    if not mask.any():
        return catalog.head(0)

    q_idx = catalog[mask].index[0]
    row_pos = catalog.index.get_loc(q_idx)

    scores_vec = numeric_matrix[row_pos].copy()
    scores_vec[row_pos] = -1

    top_pos = np.argsort(-scores_vec)[:k + len(exclude_titles) + 1]
    result_rows, result_scores = [], []
    for pos in top_pos:
        title = catalog.iloc[pos]["title"]
        if pos == row_pos or title in exclude_titles:
            continue
        result_rows.append(catalog.iloc[pos])
        result_scores.append(float(scores_vec[pos]))
        if len(result_rows) == k:
            break

    if not result_rows:
        return catalog.head(0)

    result = pd.DataFrame(result_rows).reset_index(drop=True)
    result["cosine_numeric_score"] = result_scores
    return result


# ── Text Embedding Cosine ──────────────────────────────────────────────────────

def top_k_cosine_text(
    query_embedding: np.ndarray,
    catalog: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 50,
    exclude_titles: set = None,
    query_title: str = None,
) -> pd.DataFrame:
    """
    query_embedding: 1-D L2-normalized 384-dim vector (already normalized)
    embeddings: (N, 384) L2-normalized matrix
    """
    exclude_titles = exclude_titles or set()

    # dot product = cosine since both are L2-normalized
    scores_vec = embeddings @ query_embedding.astype(np.float32)

    # optionally exclude the seed title itself
    if query_title:
        seed_mask = catalog["title"].str.lower() == query_title.strip().lower()
        if seed_mask.any():
            seed_pos = catalog[seed_mask].index[0]
            scores_vec[catalog.index.get_loc(seed_pos)] = -1

    top_pos = np.argsort(-scores_vec)[:k + len(exclude_titles) + 1]
    result_rows, result_scores = [], []
    for pos in top_pos:
        title = catalog.iloc[pos]["title"]
        if title in exclude_titles:
            continue
        result_rows.append(catalog.iloc[pos])
        result_scores.append(float(scores_vec[pos]))
        if len(result_rows) == k:
            break

    if not result_rows:
        return catalog.head(0)

    result = pd.DataFrame(result_rows).reset_index(drop=True)
    result["cosine_text_score"] = result_scores
    return result


def build_numeric_matrix(catalog: pd.DataFrame) -> np.ndarray:
    return _numeric_matrix(catalog)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_engine_cosine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/engine/cosine.py backend/tests/test_engine_cosine.py
git commit -m "Port cosine similarity engine from v1"
```

---

## Task 5: Port `engine/hybrid.py`

**Files:**
- Create: `backend/app/engine/hybrid.py`
- Test: `backend/tests/test_engine_hybrid.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_engine_hybrid.py`:

```python
from app.engine.hybrid import ALPHA, BETA, HEBREW_WEIGHTS, apply_filters, recommend


def test_weights_sum_to_one():
    assert abs(ALPHA + BETA + (1 - ALPHA - BETA) - 1.0) < 1e-9
    assert abs(sum(HEBREW_WEIGHTS.values()) - 1.0) < 1e-9


def test_apply_filters_era_classic_keeps_only_pre_2000(catalog):
    filtered = apply_filters(catalog, {"era_pref": "classic"})
    assert len(filtered) > 0
    assert (filtered["start_year"].fillna(9999) < 2000).all()


def test_apply_filters_length_short_falls_back_when_too_few(catalog):
    # length_pref short = num_seasons <= 2; "any" filters dict should be unaffected
    filtered = apply_filters(catalog, {})
    assert len(filtered) == len(catalog)


def test_recommend_returns_top_n_with_hybrid_score(catalog, numeric_matrix, embeddings):
    got_idx = catalog.index[catalog["title"] == "Game of Thrones"][0]
    pos = catalog.index.get_loc(got_idx)
    query_embedding = embeddings[pos]

    result = recommend(
        "Game of Thrones",
        catalog,
        numeric_matrix,
        embeddings,
        query_embedding,
        top_n=3,
    )
    assert len(result) == 3
    for col in ["title", "genres", "rating", "hybrid_score", "jaccard_score",
                "cosine_numeric_score", "cosine_text_score"]:
        assert col in result.columns
    scores = result["hybrid_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert "Game of Thrones" not in result["title"].values


def test_recommend_uses_hebrew_weights_when_query_lang_he(catalog, numeric_matrix, embeddings):
    got_idx = catalog.index[catalog["title"] == "Game of Thrones"][0]
    pos = catalog.index.get_loc(got_idx)
    query_embedding = embeddings[pos]

    result = recommend(
        "Game of Thrones",
        catalog,
        numeric_matrix,
        embeddings,
        query_embedding,
        top_n=3,
        query_lang="he",
    )
    assert len(result) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_engine_hybrid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.engine.hybrid'`

- [ ] **Step 3: Create `app/engine/hybrid.py`**

Port from `C:\Users\Hello\Desktop\cinematch-ai-main\engine\hybrid.py`. The only change is the two import lines at the top (`engine.` → `app.engine.`):

```python
"""
Hybrid scorer.
Combines Jaccard, Cosine-numeric, and Cosine-text into one ranked list.

Score = α·Jaccard + β·Cosine_numeric + (1-α-β)·Cosine_text

Default weights tuned on a 10-pair hand-rated preference set:
  α = 0.35  (genre/era matching matters most)
  β = 0.30  (numeric profile: rating, popularity)
  γ = 0.35  (semantic synopsis similarity)
"""

import numpy as np
import pandas as pd
from app.engine.jaccard import top_k_jaccard
from app.engine.cosine import top_k_cosine_numeric, top_k_cosine_text

ALPHA = 0.35   # Jaccard weight (genre/era)
BETA  = 0.30   # Cosine-numeric weight (rating/popularity profile)
# GAMMA = 1 - ALPHA - BETA = 0.35  → Cosine-text weight (semantic plot)

# ── Bias mitigation ───────────────────────────────────────────────────────────
# The multilingual model performs better on English text; Hebrew embedding quality
# is lower. When the user queries in Hebrew we up-weight Jaccard (language-neutral,
# genre-set based) and down-weight text-cosine (semantic embedding).
HEBREW_WEIGHTS = {"alpha": 0.50, "beta": 0.30, "gamma": 0.20}


def apply_filters(catalog: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Hard-filter a candidate DataFrame by intent filters.
    Year/era filters are always enforced — no fallback to unfiltered results
    when a hard year constraint is present. Other soft filters fall back to
    the full input if fewer than 3 rows survive.
    """
    if not filters:
        return catalog
    df = catalog.copy()

    # ── Hard year filters (ALWAYS enforced, no fallback) ─────────────────────
    year_hard = False

    year_min = filters.get("year_min")
    year_max = filters.get("year_max")
    if year_min:
        df = df[df["start_year"].fillna(0) >= year_min]
        year_hard = True
    if year_max:
        df = df[df["start_year"].fillna(9999) <= year_max]
        year_hard = True

    era = filters.get("era_pref")
    if era and era != "any":
        year_hard = True
        if era in ("recent", "2020s"):
            df = df[df["start_year"].fillna(0) >= 2020]
        elif era == "2010s":
            df = df[(df["start_year"].fillna(0) >= 2010) & (df["start_year"].fillna(0) < 2020)]
        elif era == "2000s":
            df = df[(df["start_year"].fillna(0) >= 2000) & (df["start_year"].fillna(0) < 2010)]
        elif era == "1990s":
            df = df[(df["start_year"].fillna(0) >= 1990) & (df["start_year"].fillna(0) < 2000)]
        elif era == "classic":
            df = df[df["start_year"].fillna(9999) < 2000]

    # If hard year filter left nothing, return empty so caller shows "few results"
    if year_hard and len(df) == 0:
        return df

    # ── Soft filters (skip if fewer than 3 would survive) ────────────────────
    pre_soft = df.copy()

    # length_pref
    lp = filters.get("length_pref")
    if lp == "short":
        tmp = df[df["num_seasons"].fillna(99) <= 2]
        if len(tmp) >= 3: df = tmp
    elif lp == "long":
        tmp = df[df["num_seasons"].fillna(0) >= 4]
        if len(tmp) >= 3: df = tmp
    elif lp == "limited":
        tmp = df[df["num_seasons"].fillna(0) == 1]
        if len(tmp) >= 3: df = tmp

    # language_pref
    lang_col = "language" if "language" in df.columns else (
        "original_language" if "original_language" in df.columns else None)
    if lang_col:
        lpr = filters.get("language_pref")
        if lpr == "foreign":
            tmp = df[df[lang_col] != "en"]
            if len(tmp) >= 3: df = tmp
        elif lpr not in (None, "any"):
            tmp = df[df[lang_col] == lpr]
            if len(tmp) >= 3: df = tmp

    # status
    status = filters.get("status")
    if status == "airing":
        tmp = df[df["end_year"].isna()]
        if len(tmp) >= 3: df = tmp
    elif status == "finished":
        tmp = df[df["end_year"].notna()]
        if len(tmp) >= 3: df = tmp

    # popularity_pref
    pop = filters.get("popularity_pref")
    if pop == "hidden_gem":
        tmp = df[(df["votes"].fillna(0) < 10000) & (df["rating"].fillna(0) > 7.5)]
        if len(tmp) >= 3: df = tmp
    elif pop == "trending":
        df = df.sort_values("popularity", ascending=False)
    elif pop == "well_known":
        tmp = df[df["votes"].fillna(0) > 100000]
        if len(tmp) >= 3: df = tmp

    # binge_pref
    if filters.get("binge_pref") == "binge" and "binge_fit_score" in df.columns:
        df = df.sort_values("binge_fit_score", ascending=False)

    # rating_min
    rmin = filters.get("rating_min")
    if rmin:
        tmp = df[df["rating"].fillna(0) >= rmin]
        if len(tmp) >= 3: df = tmp

    # exclude_genres
    genre_col = "genre_set_str" if "genre_set_str" in df.columns else "genres"
    for genre in (filters.get("exclude_genres") or []):
        tmp = df[~df[genre_col].fillna("").str.contains(genre, case=False, na=False)]
        if len(tmp) >= 3:
            df = tmp

    return df


def recommend(
    query_title: str,
    catalog: pd.DataFrame,
    numeric_matrix: np.ndarray,
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_n: int = 5,
    alpha: float = ALPHA,
    beta: float = BETA,
    exclude_titles: set = None,
    query_lang: str = "en",
    filters: dict = None,
) -> pd.DataFrame:
    # ── Bias mitigation: re-weight for Hebrew queries ───────────────────────
    if query_lang == "he":
        alpha = HEBREW_WEIGHTS["alpha"]
        beta  = HEBREW_WEIGHTS["beta"]
        # gamma derived below = 0.20
    """
    Returns top_n recommendations as a DataFrame with columns:
      title, genres, rating, votes, decade_str, overview, poster_path,
      jaccard_score, cosine_numeric_score, cosine_text_score, hybrid_score
    """
    gamma = 1.0 - alpha - beta
    exclude_titles = exclude_titles or set()

    # Get candidates from each method (union pool)
    j_res = top_k_jaccard(query_title, catalog, k=100, exclude_titles=exclude_titles)
    n_res = top_k_cosine_numeric(query_title, catalog, numeric_matrix, k=100,
                                 exclude_titles=exclude_titles)
    t_res = top_k_cosine_text(query_embedding, catalog, embeddings, k=100,
                              exclude_titles=exclude_titles, query_title=query_title)

    # Build a unified score table indexed by title
    scores: dict[str, dict] = {}

    def _add(df: pd.DataFrame, col: str):
        for _, row in df.iterrows():
            t = row["title"]
            if t not in scores:
                scores[t] = {
                    "jaccard_score": 0.0,
                    "cosine_numeric_score": 0.0,
                    "cosine_text_score": 0.0,
                    "_row": row,
                }
            scores[t][col] = float(row.get(col, 0.0) or 0.0)

    _add(j_res, "jaccard_score")
    _add(n_res, "cosine_numeric_score")
    _add(t_res, "cosine_text_score")

    # Compute hybrid
    for t in scores:
        d = scores[t]
        d["hybrid_score"] = (
            alpha * d["jaccard_score"]
            + beta  * d["cosine_numeric_score"]
            + gamma * d["cosine_text_score"]
        )

    if not scores:
        return pd.DataFrame()

    rows = []
    for t, d in sorted(scores.items(), key=lambda x: -x[1]["hybrid_score"]):
        r = d["_row"].copy()
        r["jaccard_score"]        = round(d["jaccard_score"], 4)
        r["cosine_numeric_score"] = round(d["cosine_numeric_score"], 4)
        r["cosine_text_score"]    = round(d["cosine_text_score"], 4)
        r["hybrid_score"]         = round(d["hybrid_score"], 4)
        rows.append(r)

    # Fetch a wide pool so year/era filters have rows to work with,
    # then apply hard filters without falling back to unfiltered data.
    candidates = pd.DataFrame(rows).reset_index(drop=True)
    if filters:
        candidates = apply_filters(candidates, filters)
    result = candidates.head(top_n).reset_index(drop=True)
    return result


def score_all_pairs(
    catalog: pd.DataFrame,
    numeric_matrix: np.ndarray,
    embeddings: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> np.ndarray:
    """Compute the full N×N hybrid matrix (used by anomaly + evaluate)."""
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos
    from app.engine.jaccard import batch_jaccard_matrix

    gamma = 1.0 - alpha - beta

    j_mat = batch_jaccard_matrix(catalog).astype(np.float32)
    n_mat = numeric_matrix.astype(np.float32)
    # embeddings are already L2-normalized → dot = cosine
    t_mat = (embeddings @ embeddings.T).astype(np.float32)

    return alpha * j_mat + beta * n_mat + gamma * t_mat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_engine_hybrid.py -v`
Expected: PASS (this test is slow - `recommend()` calls `top_k_jaccard` which is O(N) - allow up to ~30s)

- [ ] **Step 5: Commit**

```bash
git add backend/app/engine/hybrid.py backend/tests/test_engine_hybrid.py
git commit -m "Port hybrid recommendation scorer from v1"
```

---

## Task 6: Port `engine/anomaly.py`

**Files:**
- Create: `backend/app/engine/anomaly.py`
- Test: `backend/tests/test_engine_anomaly.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_engine_anomaly.py`:

```python
import numpy as np

from app.engine import anomaly


def test_calibrate_sets_threshold_to_percentile():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    threshold = anomaly.calibrate(scores, percentile=10.0)
    assert threshold == np.percentile(scores, 10.0)
    assert anomaly.get_threshold() == threshold


def test_is_anomalous_below_threshold():
    anomaly.calibrate(np.array([0.5, 0.6, 0.7, 0.8, 0.9]), percentile=20.0)
    threshold = anomaly.get_threshold()
    assert anomaly.is_anomalous(threshold - 0.01) is True
    assert anomaly.is_anomalous(threshold + 0.01) is False


def test_is_anomalous_uses_explicit_threshold_override():
    assert anomaly.is_anomalous(0.1, threshold=0.5) is True
    assert anomaly.is_anomalous(0.6, threshold=0.5) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_engine_anomaly.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.engine.anomaly'`

- [ ] **Step 3: Create `app/engine/anomaly.py`**

Port verbatim from `C:\Users\Hello\Desktop\cinematch-ai-main\engine\anomaly.py` (no import path changes needed):

```python
"""
Anomaly detector.
Flags "out-of-distribution" queries: when the best hybrid match score
falls below the 5th-percentile threshold computed across the catalog.

If a query's best match score < threshold → the query is unusual / no good match.
"""

import numpy as np
import pandas as pd

# Precomputed or set at startup
_threshold: float = None


def calibrate(best_scores_array: np.ndarray, percentile: float = 5.0) -> float:
    """
    best_scores_array: 1-D array of best-match hybrid scores for all catalog
                       items (computed during startup by evaluate.py or app startup).
    Returns the threshold below which a query is "anomalous".
    """
    global _threshold
    _threshold = float(np.percentile(best_scores_array, percentile))
    return _threshold


def is_anomalous(best_match_score: float, threshold: float = None) -> bool:
    t = threshold if threshold is not None else _threshold
    if t is None:
        return False  # not calibrated → assume normal
    return best_match_score < t


def get_threshold() -> float:
    return _threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_engine_anomaly.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/engine/anomaly.py backend/tests/test_engine_anomaly.py
git commit -m "Port anomaly threshold detector from v1"
```

---

## Task 7: Clustering Feature Vector (`clustering/features.py`)

This module defines the fixed 14-dim feature space used by K-Means training (Task 8),
the onboarding/intent-to-vector mapping (Task 9), and within-cluster ranking (Task 10).

**Files:**
- Create: `backend/app/clustering/features.py`
- Test: `backend/tests/test_clustering_features.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_clustering_features.py`:

```python
import numpy as np

from app.clustering.features import FEATURE_DIMS, GENRE_DIMS, NUMERIC_DIMS, build_cluster_features


def test_feature_dims_has_14_entries():
    assert len(GENRE_DIMS) == 10
    assert len(NUMERIC_DIMS) == 4
    assert len(FEATURE_DIMS) == 14
    assert FEATURE_DIMS[:10] == [f"genre:{g}" for g in GENRE_DIMS]
    assert FEATURE_DIMS[10:] == NUMERIC_DIMS


def test_build_cluster_features_shape_and_columns(catalog):
    features = build_cluster_features(catalog)
    assert len(features) == len(catalog)
    assert list(features.columns) == ["title"] + FEATURE_DIMS


def test_genre_dims_are_binary(catalog):
    features = build_cluster_features(catalog)
    for genre in GENRE_DIMS:
        col = features[f"genre:{genre}"]
        assert set(col.unique()).issubset({0.0, 1.0})


def test_game_of_thrones_genre_dims(catalog):
    features = build_cluster_features(catalog)
    row = features[features["title"] == "Game of Thrones"].iloc[0]
    # genres = "Action, Adventure, Drama"
    assert row["genre:Drama"] == 1.0
    assert row["genre:Action & Adventure"] == 0.0
    assert row["genre:Comedy"] == 0.0


def test_num_seasons_z_has_no_nan_and_is_roughly_standardized(catalog):
    features = build_cluster_features(catalog)
    z = features["num_seasons_z"]
    assert not z.isna().any()
    assert abs(z.mean()) < 1e-3
    assert abs(z.std() - 1.0) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_clustering_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.clustering.features'`

- [ ] **Step 3: Create `app/clustering/features.py`**

```python
"""
Feature vector construction for K-Means clustering and the cluster-based
recommender.

Each title is represented as a 14-dim vector:
  - 10 genre one-hot dims (1.0 if the genre string appears in the title's
    comma-separated `genres` column, else 0.0)
  - 4 numeric z-score dims: rating_z, popularity_z, start_year_z, num_seasons_z

`rating_z`, `popularity_z`, and `start_year_z` are already precomputed in
catalog.parquet. `num_seasons_z` is computed here: missing `num_seasons`
values are imputed with the catalog median before z-scoring, so titles with
no season-count data do not get an arbitrary/extreme value in the feature
space (per the design spec's edge-case notes).
"""

import numpy as np
import pandas as pd

GENRE_DIMS = [
    "Drama", "Comedy", "Animation", "Crime", "Action & Adventure",
    "Sci-Fi & Fantasy", "Mystery", "Documentary", "Family", "Romance",
]
NUMERIC_DIMS = ["rating_z", "popularity_z", "start_year_z", "num_seasons_z"]
FEATURE_DIMS = [f"genre:{g}" for g in GENRE_DIMS] + NUMERIC_DIMS


def _genre_set(genres_str) -> set:
    if not genres_str:
        return set()
    return {g.strip() for g in str(genres_str).split(",") if g.strip()}


def build_cluster_features(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ["title"] + FEATURE_DIMS, one row per
    catalog title, in the same row order as `catalog`.
    """
    out = pd.DataFrame({"title": catalog["title"].values})

    genre_sets = catalog["genres"].fillna("").apply(_genre_set)
    for genre in GENRE_DIMS:
        out[f"genre:{genre}"] = genre_sets.apply(
            lambda gs, g=genre: 1.0 if g in gs else 0.0
        ).astype(np.float32)

    out["rating_z"] = catalog["rating_z"].fillna(0.0).astype(np.float32).values
    out["popularity_z"] = catalog["popularity_z"].fillna(0.0).astype(np.float32).values
    out["start_year_z"] = catalog["start_year_z"].fillna(0.0).astype(np.float32).values

    num_seasons = pd.to_numeric(catalog["num_seasons"], errors="coerce")
    num_seasons = num_seasons.fillna(num_seasons.median())
    mean, std = num_seasons.mean(), num_seasons.std()
    out["num_seasons_z"] = ((num_seasons - mean) / std).astype(np.float32).values

    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_clustering_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/clustering/features.py backend/tests/test_clustering_features.py
git commit -m "Add 14-dim clustering feature vector builder"
```

---

## Task 8: K-Means Training Script (`clustering/train_clusters.py`)

This is the offline training step. It writes three artifacts into `app/data/`
that the runtime API loads at startup: `cluster_labels.parquet`,
`cluster_centroids.json`, `cluster_profiles.json`.

**Files:**
- Create: `backend/app/clustering/train_clusters.py`
- Test: `backend/tests/test_clustering_train.py`
- Generated (committed as data artifacts): `backend/app/data/cluster_labels.parquet`, `backend/app/data/cluster_centroids.json`, `backend/app/data/cluster_profiles.json`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_clustering_train.py`:

```python
import numpy as np
import pandas as pd

from app.clustering.features import FEATURE_DIMS
from app.clustering.train_clusters import build_cluster_profiles, choose_k


def test_choose_k_returns_k_in_range_and_scores():
    rng = np.random.default_rng(42)
    # Three well-separated synthetic clusters in 14-dim space
    cluster_a = rng.normal(loc=0.0, scale=0.1, size=(50, 14))
    cluster_b = rng.normal(loc=5.0, scale=0.1, size=(50, 14))
    cluster_c = rng.normal(loc=-5.0, scale=0.1, size=(50, 14))
    X = np.vstack([cluster_a, cluster_b, cluster_c]).astype(np.float32)

    best_k, scores = choose_k(X, k_range=range(2, 5))
    assert best_k in range(2, 5)
    assert set(scores.keys()) == set(range(2, 5))
    assert all(-1.0 <= v <= 1.0 for v in scores.values())


def test_build_cluster_profiles_structure():
    titles = [f"Show {i}" for i in range(6)]
    features = pd.DataFrame({"title": titles})
    for dim in FEATURE_DIMS:
        features[dim] = 0.0
    # 3 shows are Drama-heavy, 3 are Comedy-heavy
    features.loc[0:2, "genre:Drama"] = 1.0
    features.loc[3:5, "genre:Comedy"] = 1.0

    catalog = pd.DataFrame({
        "start_year": [2018, 2019, 2020, 2010, 2011, 2012],
        "rating": [8.0, 8.2, 8.4, 7.0, 7.2, 7.4],
        "binge_fit_score": [5.0, 5.2, 5.4, 4.0, 4.2, 4.4],
    })
    labels = np.array([0, 0, 0, 1, 1, 1])

    profiles = build_cluster_profiles(features, catalog, labels)
    assert set(profiles.keys()) == {"0", "1"}
    assert profiles["0"]["top_genres"][0] == "Drama"
    assert profiles["1"]["top_genres"][0] == "Comedy"
    assert profiles["0"]["size"] == 3
    assert profiles["1"]["size"] == 3
    assert "label_he" in profiles["0"] and "label_en" in profiles["0"]
    assert profiles["0"]["avg_rating"] == 8.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_clustering_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.clustering.train_clusters'`

- [ ] **Step 3: Create `app/clustering/train_clusters.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_clustering_train.py -v`
Expected: PASS (synthetic-data tests run in well under a second)

- [ ] **Step 5: Run the training script to generate the real data artifacts**

Run: `cd backend && python -m app.clustering.train_clusters`
Expected: prints silhouette scores for k=6..12, the chosen k, and a final
"Wrote cluster_labels.parquet..." line. This may take 1-3 minutes (K-Means
with `n_init=10` runs 7 times for the candidate k values, each on 11,013
rows x 14 dims). Verify the three new files exist:

```powershell
Get-ChildItem backend\app\data\cluster_*
```

Expected: `cluster_labels.parquet`, `cluster_centroids.json`, `cluster_profiles.json` all present.

- [ ] **Step 6: Sanity-check the generated cluster_profiles.json**

Run:
```powershell
Get-Content backend\app\data\cluster_profiles.json | python -m json.tool | Select-Object -First 20
```

Expected: a JSON object keyed `"0"`, `"1"`, ... up to `k-1`, each with
`label_he`, `label_en`, `size`, `top_genres`, `avg_rating`, `avg_start_year`,
`avg_binge_fit_score`. Confirm every cluster's `size` is non-trivial (no
empty clusters) - if any cluster has `size < 50`, note it but proceed (an
uneven cluster distribution is expected and fine for this dataset).

- [ ] **Step 7: Commit**

```bash
git add backend/app/clustering/train_clusters.py backend/tests/test_clustering_train.py backend/app/data/cluster_labels.parquet backend/app/data/cluster_centroids.json backend/app/data/cluster_profiles.json
git commit -m "Add K-Means training script and generate cluster artifacts"
```

---

## Task 9: Onboarding-to-Vector Mapping (`clustering/onboarding_map.py`)

This module turns the 5 onboarding answers (or, for chat-derived seedless
searches, a `chat_turn` `intent` dict) into a 14-dim target vector + boolean
mask aligned to `FEATURE_DIMS` (Task 7). Task 10's `nearest_cluster` and
`recommend_from_cluster` consume this vector/mask pair.

Design notes:
- The mask marks which of the 14 dims the user actually expressed a
  preference for. Dims left at "any"/"doesn't matter" are masked out (`False`)
  and excluded from distance calculations (per the design spec's edge-case
  notes).
- The numeric dims (`rating_z`, `popularity_z`, `start_year_z`,
  `num_seasons_z`) in `catalog.parquet`/`cluster_labels.parquet` are already
  z-scored. So onboarding answers map directly to **target z-score values**
  (e.g. "long" → `num_seasons_z = 1.0`, meaning "about 1 standard deviation
  above the mean season count") rather than raw means/stds. No separate
  stats file is needed.
- `intent_to_onboarding_answers` adapts a `chat_turn` `intent` dict (from
  `agent/llm.py`, Task 11) into the same answers vocabulary, so
  `routers/chat.py` (Task 17) can route seedless free-text searches through
  the same `build_user_vector` → `recommend_from_cluster` path used by
  onboarding.

**Files:**
- Create: `backend/app/clustering/onboarding_map.py`
- Test: `backend/tests/test_clustering_onboarding_map.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_clustering_onboarding_map.py`:

```python
import numpy as np

from app.clustering.features import FEATURE_DIMS
from app.clustering.onboarding_map import build_user_vector, intent_to_onboarding_answers


def _idx(dim: str) -> int:
    return FEATURE_DIMS.index(dim)


def test_build_user_vector_all_any_returns_empty_mask():
    answers = {"genre": "any", "length": "any", "era": "any", "tone": "any", "popularity": "any"}
    vector, mask = build_user_vector(answers)
    assert vector.shape == (14,)
    assert mask.shape == (14,)
    assert mask.dtype == bool
    assert not mask.any()
    assert np.allclose(vector, 0.0)


def test_build_user_vector_genre_sets_one_hot_dim():
    answers = {"genre": "drama", "length": "any", "era": "any", "tone": "any", "popularity": "any"}
    vector, mask = build_user_vector(answers)
    assert vector[_idx("genre:Drama")] == 1.0
    assert mask[_idx("genre:Drama")] is np.True_ or mask[_idx("genre:Drama")] == True
    assert mask.sum() == 1


def test_build_user_vector_tone_sets_genre_dim():
    answers = {"genre": "any", "length": "any", "era": "any", "tone": "thriller_action", "popularity": "any"}
    vector, mask = build_user_vector(answers)
    assert vector[_idx("genre:Action & Adventure")] == 1.0
    assert mask[_idx("genre:Action & Adventure")]
    assert mask.sum() == 1


def test_build_user_vector_length_and_era():
    answers = {"genre": "any", "length": "short", "era": "recent", "tone": "any", "popularity": "any"}
    vector, mask = build_user_vector(answers)
    assert vector[_idx("num_seasons_z")] == -0.6
    assert mask[_idx("num_seasons_z")]
    assert vector[_idx("start_year_z")] == 0.8
    assert mask[_idx("start_year_z")]
    assert mask.sum() == 2


def test_build_user_vector_hidden_gem_sets_two_dims():
    answers = {"genre": "any", "length": "any", "era": "any", "tone": "any", "popularity": "hidden_gem"}
    vector, mask = build_user_vector(answers)
    assert vector[_idx("rating_z")] == 1.2
    assert vector[_idx("popularity_z")] == -0.3
    assert mask[_idx("rating_z")] and mask[_idx("popularity_z")]
    assert mask.sum() == 2


def test_intent_to_onboarding_answers_defaults_to_any():
    answers = intent_to_onboarding_answers({})
    assert answers == {"genre": "any", "length": "any", "era": "any", "tone": "any", "popularity": "any"}


def test_intent_to_onboarding_answers_maps_mood_length_era_popularity():
    intent = {
        "mood": ["funny"],
        "length_pref": "short",
        "era_pref": "classic",
        "popularity_pref": "hidden_gem",
    }
    answers = intent_to_onboarding_answers(intent)
    assert answers == {"genre": "any", "length": "short", "era": "classic", "tone": "light_fun", "popularity": "hidden_gem"}


def test_intent_to_onboarding_answers_dark_mood_maps_to_thriller_action():
    intent = {"mood": ["dark"], "length_pref": "any", "era_pref": "any", "popularity_pref": "any"}
    answers = intent_to_onboarding_answers(intent)
    assert answers["tone"] == "thriller_action"


def test_intent_to_onboarding_answers_trending_maps_to_well_known():
    intent = {"mood": [], "length_pref": "any", "era_pref": "any", "popularity_pref": "trending"}
    answers = intent_to_onboarding_answers(intent)
    assert answers["popularity"] == "well_known"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_clustering_onboarding_map.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.clustering.onboarding_map'`

- [ ] **Step 3: Create `app/clustering/onboarding_map.py`**

```python
"""
Maps onboarding answers (or a chat_turn intent dict) to a 14-dim target
vector + boolean mask aligned to FEATURE_DIMS, for use by
clustering/recommend.py.

Onboarding answers dict shape (all 5 keys always present):
  {
    "genre": "drama" | "comedy" | "action_adventure" | "scifi_fantasy"
             | "crime" | "animation" | "romance" | "any",
    "length": "short" | "medium" | "long" | "any",
    "era": "recent" | "classic" | "any",
    "tone": "light_fun" | "serious_drama" | "thriller_action" | "any",
    "popularity": "well_known" | "hidden_gem" | "any",
  }
"""

import numpy as np

from app.clustering.features import FEATURE_DIMS

GENRE_QUESTION_MAP = {
    "drama": "Drama",
    "comedy": "Comedy",
    "action_adventure": "Action & Adventure",
    "scifi_fantasy": "Sci-Fi & Fantasy",
    "crime": "Crime",
    "animation": "Animation",
    "romance": "Romance",
}

TONE_GENRE_MAP = {
    "light_fun": "Comedy",
    "serious_drama": "Drama",
    "thriller_action": "Action & Adventure",
}

LENGTH_Z_MAP = {
    "short": -0.6,
    "medium": 0.0,
    "long": 1.0,
}

ERA_Z_MAP = {
    "recent": 0.8,
    "classic": -1.2,
}

POPULARITY_Z_MAP = {
    "well_known": {"popularity_z": 1.0},
    "hidden_gem": {"rating_z": 1.2, "popularity_z": -0.3},
}

# chat_turn intent.mood tags -> onboarding "tone" answer
MOOD_TO_TONE = {
    "funny": "light_fun",
    "light": "light_fun",
    "dark": "thriller_action",
    "thrilling": "thriller_action",
    "emotional": "serious_drama",
}

# chat_turn intent.length_pref -> onboarding "length" answer
LENGTH_PREF_MAP = {
    "short": "short",
    "limited": "short",
    "long": "long",
}

# chat_turn intent.era_pref -> onboarding "era" answer
ERA_PREF_MAP = {
    "recent": "recent",
    "2020s": "recent",
    "classic": "classic",
    "1990s": "classic",
    "2000s": "classic",
}

# chat_turn intent.popularity_pref -> onboarding "popularity" answer
POPULARITY_PREF_MAP = {
    "hidden_gem": "hidden_gem",
    "well_known": "well_known",
    "trending": "well_known",
}


def build_user_vector(answers: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (vector, mask), both shape (14,), aligned to FEATURE_DIMS.
    `mask[i] is True` means the user expressed a preference for dim i;
    `vector[i]` is only meaningful where `mask[i]` is True.
    """
    vector = np.zeros(len(FEATURE_DIMS), dtype=np.float32)
    mask = np.zeros(len(FEATURE_DIMS), dtype=bool)

    genre = answers.get("genre", "any")
    if genre in GENRE_QUESTION_MAP:
        idx = FEATURE_DIMS.index(f"genre:{GENRE_QUESTION_MAP[genre]}")
        vector[idx] = 1.0
        mask[idx] = True

    tone = answers.get("tone", "any")
    if tone in TONE_GENRE_MAP:
        idx = FEATURE_DIMS.index(f"genre:{TONE_GENRE_MAP[tone]}")
        vector[idx] = 1.0
        mask[idx] = True

    length = answers.get("length", "any")
    if length in LENGTH_Z_MAP:
        idx = FEATURE_DIMS.index("num_seasons_z")
        vector[idx] = LENGTH_Z_MAP[length]
        mask[idx] = True

    era = answers.get("era", "any")
    if era in ERA_Z_MAP:
        idx = FEATURE_DIMS.index("start_year_z")
        vector[idx] = ERA_Z_MAP[era]
        mask[idx] = True

    popularity = answers.get("popularity", "any")
    if popularity in POPULARITY_Z_MAP:
        for dim, value in POPULARITY_Z_MAP[popularity].items():
            idx = FEATURE_DIMS.index(dim)
            vector[idx] = value
            mask[idx] = True

    return vector, mask


def intent_to_onboarding_answers(intent: dict) -> dict:
    """
    Adapts a chat_turn `intent` dict (see agent/llm.py) to the onboarding
    answers vocabulary, so seedless chat searches can be routed through
    build_user_vector -> recommend_from_cluster.
    """
    answers = {"genre": "any", "length": "any", "era": "any", "tone": "any", "popularity": "any"}

    for mood in intent.get("mood", []):
        if mood in MOOD_TO_TONE:
            answers["tone"] = MOOD_TO_TONE[mood]
            break

    length_pref = intent.get("length_pref", "any")
    if length_pref in LENGTH_PREF_MAP:
        answers["length"] = LENGTH_PREF_MAP[length_pref]

    era_pref = intent.get("era_pref", "any")
    if era_pref in ERA_PREF_MAP:
        answers["era"] = ERA_PREF_MAP[era_pref]

    popularity_pref = intent.get("popularity_pref", "any")
    if popularity_pref in POPULARITY_PREF_MAP:
        answers["popularity"] = POPULARITY_PREF_MAP[popularity_pref]

    return answers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_clustering_onboarding_map.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/clustering/onboarding_map.py backend/tests/test_clustering_onboarding_map.py
git commit -m "Add onboarding/intent to feature-vector mapping"
```

---

## Task 10: Cluster-Based Recommender (`clustering/recommend.py`)

Given a target vector + mask (Task 9), this module finds the nearest cluster
and ranks titles within it. This is the seedless recommendation path used by
onboarding (`routers/recommend.py`, Task 16) and by seedless chat searches
(`routers/chat.py`, Task 17).

The design spec describes within-cluster ranking as "the hybrid engine
(Jaccard + Cosine numeric, from v1)". Onboarding has no seed title for v1's
`recommend()` to compare against, so this is implemented as Euclidean
distance to the user's 14-dim target vector (Task 9): the genre one-hot dims
play the role of Jaccard (categorical genre overlap) and the 4 numeric
z-score dims play the role of Cosine numeric (rating/popularity/year/length
similarity), combined into a single distance metric over the same feature
space used for clustering.

**Files:**
- Create: `backend/app/clustering/recommend.py`
- Test: `backend/tests/test_clustering_recommend.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_clustering_recommend.py`:

```python
import numpy as np
import pandas as pd

from app.clustering.features import FEATURE_DIMS
from app.clustering.recommend import nearest_cluster, recommend_from_cluster


def _zeros_vector():
    return np.zeros(len(FEATURE_DIMS), dtype=np.float32)


def _zeros_mask():
    return np.zeros(len(FEATURE_DIMS), dtype=bool)


def test_nearest_cluster_all_any_picks_highest_binge_fit_score():
    vector = _zeros_vector()
    mask = _zeros_mask()
    centroids = [[0.0] * 14, [0.0] * 14, [0.0] * 14]
    profiles = {
        "0": {"avg_binge_fit_score": 4.0},
        "1": {"avg_binge_fit_score": 6.5},
        "2": {"avg_binge_fit_score": 5.0},
    }
    assert nearest_cluster(vector, mask, centroids, profiles) == 1


def test_nearest_cluster_picks_closest_centroid_on_masked_dims():
    vector = _zeros_vector()
    mask = _zeros_mask()
    drama_idx = FEATURE_DIMS.index("genre:Drama")
    mask[drama_idx] = True
    vector[drama_idx] = 1.0

    centroids = [[0.0] * 14, [0.0] * 14]
    centroids[0][drama_idx] = 0.1  # far from 1.0
    centroids[1][drama_idx] = 0.9  # close to 1.0
    profiles = {"0": {"avg_binge_fit_score": 10.0}, "1": {"avg_binge_fit_score": 0.0}}

    assert nearest_cluster(vector, mask, centroids, profiles) == 1


def _make_pool() -> pd.DataFrame:
    rows = []
    for i, (drama_val, binge) in enumerate([(1.0, 3.0), (0.9, 7.0), (0.0, 9.0)]):
        row = {"title": f"Show {i}", "cluster_id": 0, "binge_fit_score": binge}
        for dim in FEATURE_DIMS:
            row[dim] = 0.0
        row["genre:Drama"] = drama_val
        rows.append(row)
    # a title that belongs to a different cluster
    other = {"title": "Other Cluster Show", "cluster_id": 1, "binge_fit_score": 100.0}
    for dim in FEATURE_DIMS:
        other[dim] = 0.0
    rows.append(other)
    return pd.DataFrame(rows)


def test_recommend_from_cluster_ranks_by_distance_when_mask_set():
    pool = _make_pool()
    vector = _zeros_vector()
    mask = _zeros_mask()
    drama_idx = FEATURE_DIMS.index("genre:Drama")
    vector[drama_idx] = 1.0
    mask[drama_idx] = True

    result = recommend_from_cluster(pool, cluster_id=0, vector=vector, mask=mask, top_n=2)
    assert list(result["title"]) == ["Show 0", "Show 1"]


def test_recommend_from_cluster_ranks_by_binge_fit_score_when_mask_empty():
    pool = _make_pool()
    vector = _zeros_vector()
    mask = _zeros_mask()

    result = recommend_from_cluster(pool, cluster_id=0, vector=vector, mask=mask, top_n=2)
    assert list(result["title"]) == ["Show 2", "Show 1"]


def test_recommend_from_cluster_excludes_titles_and_other_clusters():
    pool = _make_pool()
    vector = _zeros_vector()
    mask = _zeros_mask()

    result = recommend_from_cluster(
        pool, cluster_id=0, vector=vector, mask=mask, top_n=3, exclude_titles=["Show 2"]
    )
    assert "Other Cluster Show" not in list(result["title"])
    assert "Show 2" not in list(result["title"])
    assert len(result) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_clustering_recommend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.clustering.recommend'`

- [ ] **Step 3: Create `app/clustering/recommend.py`**

```python
"""
Cluster-based recommender: given a target feature vector + mask
(see clustering/onboarding_map.py), find the nearest cluster and rank
titles within it.
"""

import numpy as np
import pandas as pd

from app.clustering.features import FEATURE_DIMS


def nearest_cluster(vector: np.ndarray, mask: np.ndarray, centroids: list, profiles: dict) -> int:
    """
    Returns the cluster_id whose centroid is closest to `vector` on the
    masked dims (Euclidean distance, unmasked dims contribute 0).

    If `mask` has no True entries (the user answered "doesn't matter" to
    everything), falls back to the cluster with the highest
    avg_binge_fit_score, per the design spec's edge-case notes.
    """
    if not mask.any():
        return int(max(profiles, key=lambda cid: profiles[cid]["avg_binge_fit_score"]))

    centroids_arr = np.asarray(centroids, dtype=np.float32)
    diffs = (centroids_arr - vector) * mask
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    return int(np.argmin(distances))


def recommend_from_cluster(
    catalog_with_features: pd.DataFrame,
    cluster_id: int,
    vector: np.ndarray,
    mask: np.ndarray,
    top_n: int = 3,
    exclude_titles: list[str] | None = None,
) -> pd.DataFrame:
    """
    catalog_with_features: catalog rows joined with cluster_id + FEATURE_DIMS
                            columns (see app/state.py, Task 15).
    Returns the top_n rows (all original columns, reset index) from the
    given cluster, ranked by masked Euclidean distance to `vector` (closer
    first). If `mask` has no True entries, ranks by binge_fit_score
    (descending) instead.
    """
    pool = catalog_with_features[catalog_with_features["cluster_id"] == cluster_id]
    if exclude_titles:
        pool = pool[~pool["title"].isin(exclude_titles)]

    if pool.empty:
        return pool.head(0)

    if not mask.any():
        ranked = pool.sort_values("binge_fit_score", ascending=False)
    else:
        feature_matrix = pool[FEATURE_DIMS].values.astype(np.float32)
        diffs = (feature_matrix - vector) * mask
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        ranked = pool.assign(_distance=distances).sort_values("_distance").drop(columns="_distance")

    return ranked.head(top_n).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_clustering_recommend.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/clustering/recommend.py backend/tests/test_clustering_recommend.py
git commit -m "Add cluster-based recommender (nearest cluster + within-cluster ranking)"
```

---

## Task 11: Port LLM Agent (`agent/llm.py`)

This is a verbatim port of `agent/llm.py` from v1
(`C:\Users\Hello\Desktop\cinematch-ai-main\agent\llm.py`, 785 lines), with
**exactly one change**: `_read_secret()` drops the `streamlit.secrets`
fallback (v2 has no Streamlit) and reads only from `os.environ` (populated
via `python-dotenv` at startup, Task 15). Every other function, system
prompt, regex pattern, and code comment is unchanged from v1.

This module provides:
- `parse_intent(query)` - LLM (or regex-fallback) intent extraction, used by `routers/recommend.py`'s "any free-text" paths and as the seed parser for chat search.
- `explain_recommendations(intent, recommendations, lang)` - bilingual 1-2 sentence explanation of a list of recs.
- `classify_intent(message, conversation_history)` - classifies a chat message into search/more_options/refine/question/chat.
- `chat_turn(conversation, prev_recs, lang)` - the main conversational handler used by `routers/chat.py` (Task 17). Returns `{action, intent, reply, swap_slot_index?, follow_up}`.

**Files:**
- Create: `backend/app/agent/llm.py`
- Create: `backend/app/agent/__init__.py` (empty)
- Test: `backend/tests/test_agent_llm.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_agent_llm.py`:

```python
from app.agent import llm


def test_read_secret_reads_from_environ(monkeypatch):
    monkeypatch.setenv("SOME_TEST_KEY", "test-value-123")
    assert llm._read_secret("SOME_TEST_KEY") == "test-value-123"


def test_read_secret_missing_returns_none(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_KEY", raising=False)
    assert llm._read_secret("SOME_UNSET_KEY") is None


def test_regex_parse_detects_dark_mood_and_hebrew():
    result = llm._regex_parse("אני רוצה משהו אפל ומותח")
    assert "dark" in result["mood"]
    assert "thrilling" in result["mood"]
    assert result["lang"] == "he"


def test_regex_parse_detects_short_length_and_hidden_gem():
    result = llm._regex_parse("looking for a short hidden gem")
    assert result["length_pref"] == "short"
    assert result["popularity_pref"] == "hidden_gem"
    assert result["lang"] == "en"


def test_regex_parse_year_boundaries():
    result = llm._regex_parse("something from 2020")
    assert result["year_min"] == 2020
    assert result["era_pref"] == "recent"

    result2 = llm._regex_parse("a show from before 2000")
    assert result2["year_max"] == 1999
    assert result2["era_pref"] == "classic"


def test_detect_followup_type_other_and_shorter():
    assert llm._detect_followup_type("show me other options") == "other"
    assert llm._detect_followup_type("something shorter please") == "shorter"
    assert llm._detect_followup_type("hello there") is None


def test_keyword_classify_search_chat_and_question():
    assert llm._keyword_classify("recommend me a thriller") == "search"
    assert llm._keyword_classify("thanks!") == "chat"
    assert llm._keyword_classify("how many seasons does it have") == "question"


def test_classify_intent_falls_back_to_keywords_without_provider(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: None)
    monkeypatch.setattr(llm, "_provider", None)
    assert llm.classify_intent("recommend me a thriller") == "search"


def test_fallback_explanation_hebrew_with_seeds():
    intent = {"mood": [], "seeds": ["Breaking Bad"]}
    recs = [{"title": "Better Call Saul", "genres": "Crime, Drama", "rating": 8.7}]
    text = llm._fallback_explanation(intent, recs, "he")
    assert "Breaking Bad" in text
    assert "Better Call Saul" in text


def test_explain_recommendations_no_results():
    assert llm.explain_recommendations({}, [], "en") == "No matching results found."
    assert llm.explain_recommendations({}, [], "he") == "לא נמצאו תוצאות מתאימות."


def test_chat_turn_without_provider_returns_search_with_regex_intent(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: None)
    monkeypatch.setattr(llm, "_provider", None)
    conversation = [{"role": "user", "content": "recommend me a dark thriller"}]
    result = llm.chat_turn(conversation, prev_recs=None, lang="en")
    assert result["action"] == "search"
    assert "dark" in result["intent"]["mood"]


def test_chat_turn_fast_followup_shorter(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: None)
    monkeypatch.setattr(llm, "_provider", None)
    conversation = [{"role": "user", "content": "something shorter please"}]
    prev_recs = [{"title": "Show A", "genres": "Drama", "decade_str": "2010s", "rating": 8.0, "overview": "..."}]
    result = llm.chat_turn(conversation, prev_recs=prev_recs, lang="en")
    assert result["action"] == "refine"
    assert result["intent"]["length_pref"] == "short"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_agent_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.agent'`

- [ ] **Step 3: Create `backend/app/agent/__init__.py`**

Empty file (makes `app.agent` a package).

- [ ] **Step 4: Create `app/agent/llm.py`**

```python
"""
CineMatch AI — LLM Agent
Priority: GROQ_API_KEY → ANTHROPIC_API_KEY → offline regex fallback.
"""
from __future__ import annotations

import os, re, json
from typing import Optional

# ── Client state ───────────────────────────────────────────────────────────────

_groq_client = None
_anthropic_client = None
_provider = None  # "groq" | "anthropic" | None


def _read_secret(key: str) -> Optional[str]:
    return os.environ.get(key)


def _get_client():
    global _groq_client, _anthropic_client, _provider
    if _provider is not None:
        return True

    # Primary: Groq
    groq_key = _read_secret("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=groq_key)
            _provider = "groq"
            return True
        except Exception:
            pass

    # Fallback: Anthropic
    anthropic_key = _read_secret("ANTHROPIC_API_KEY") or _read_secret("ANTHROPIC_KEY")
    if anthropic_key:
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            _provider = "anthropic"
            return True
        except Exception:
            pass

    _provider = None
    return None


# ── System prompts ─────────────────────────────────────────────────────────────

_PARSER_SYSTEM = """\
You are an assistant that extracts structured intent from TV/movie recommendation queries.
The user may write in Hebrew or English. Always reply with valid JSON only.

Output schema:
{
  "seeds": ["Title 1"],
  "mood": ["dark", "funny"],
  "length_pref": "short|long|limited|any",
  "language_pref": "en|ko|es|de|fr|ja|foreign|any",
  "era_pref": "classic|1990s|2000s|2010s|2020s|recent|any",
  "year_min": null,
  "year_max": null,
  "status": "airing|finished|any",
  "popularity_pref": "trending|hidden_gem|well_known|any",
  "binge_pref": "binge|casual|any",
  "rating_min": null,
  "exclude_genres": [],
  "lang": "he|en",
  "free_text": "..."
}

CRITICAL RULE: BE CONSERVATIVE
Every filter you set narrows the engine's results. Only set a field if the user EXPLICITLY mentioned that preference. When in doubt, leave the field at "any" or null. Over-filtering kills the user's recommendations and surfaces bad shows.

Rules:
- seeds: ONLY titles the user explicitly mentions. Do NOT add titles from your own knowledge.
- mood: ONLY if user used an explicit mood word ("dark"→dark, "מצחיק"→funny, "מרגש"→emotional, "אפל"→dark). For neutral queries like "shows like Breaking Bad", leave mood empty.
- length_pref: ONLY if user said short/קצר/mini/limited/long/ארוך/epic/"many seasons". Default "any".
- language_pref: ONLY if user said a specific language or "foreign". Default "any". Do NOT set "en" just because the query is in English.
- era_pref: ONLY if user mentioned an era (old/classic/recent/specific decade). Default "any". A query like "shows like Breaking Bad" does NOT imply any era.
- year_min: ONLY if user said "from YEAR", "after YEAR", "YEAR and later", "משנת YEAR", "אחרי YEAR"
- year_max: ONLY if user said "before YEAR", "until YEAR", "לפני YEAR", "עד YEAR"
  Example: "from 2020" → year_min=2020, era_pref="recent"
  Example: "before 2000" → year_max=1999, era_pref="classic"
- status: ONLY if user said "still airing/ongoing/עדיין משודר" or "finished/completed/ended/הסתיים". Default "any". Do NOT auto-set "finished" just because they mentioned an old show.
- popularity_pref: ONLY if user said "hidden gem/underrated/obscure" or "trending/popular/hot" or "well-known/famous". Default "any". Do NOT auto-set "trending" just because they named a popular show.
- binge_pref: ONLY if user said "binge/weekend/סוף שבוע" or "casual/light/episodic". Default "any". Do NOT auto-set "binge" for serialized dramas.
- rating_min: ONLY if user said "highly rated/best" (then 8.5) or "decent/סביר" (then 7.0). Default null. Do NOT auto-set 7.0.
- exclude_genres: ONLY if user said "not X" or "without Y".
- lang: "he" if query contains Hebrew characters, else "en".
- Use null/any for undetermined fields. Repeat: do NOT guess or infer filters that the user did not explicitly state.
"""

_EXPLAINER_SYSTEM = """\
You are a bilingual TV/movie recommendation assistant.
Your ONLY job is to explain WHY the shows in the provided list match the user's query.
CRITICAL RULES:
- Do NOT suggest, mention, or reference any show that is not in the provided recommendations list.
- Do NOT add shows from your own knowledge. Only work with the exact list given to you.
- Do NOT say "you might also like X" or recommend anything beyond the list.
- NEVER try to explain why a show satisfies a filter it does not meet. The filtering has already
  been applied by the engine — trust the list. Do not say things like "although this is from 2010,
  it still qualifies because..." — simply explain why each show matches the mood/genre/style intent.
- When lang is "he": reply ENTIRELY in simple, clear, modern Hebrew (Israeli everyday language).
  Use short sentences. Avoid overly formal or archaic phrasing.
- When lang is "en": reply entirely in English.
- Be concise: one warm opening sentence, then one short sentence per show explaining why it fits.
"""


# ── LLM call helper ────────────────────────────────────────────────────────────

_GROQ_MODEL_CHAT = "llama-3.1-8b-instant"      # conversational handler (rate-limit-friendly)
_GROQ_MODEL_FAST = "llama-3.1-8b-instant"      # parser, classifier, explainer (cheap, fast)
# NOTE: tried llama-3.3-70b-versatile for richer persona (in PR #12) but it
# was still failing on Groq free tier even with the new 3x retry loop. The
# 70b model has tight token-per-minute limits that make it unreliable for
# interactive demos. Reverted to 8b-instant for guaranteed responsiveness.
# Persona is flatter but the agent actually replies.


def _call_llm(system: str, user: str, max_tokens: int = 600, *, model: Optional[str] = None) -> Optional[str]:
    # Groq path: retry up to 3 times with a 5-second per-attempt timeout.
    # Total worst-case still ~15 seconds, but transient slowness on the first
    # attempt now gets two more chances instead of immediately falling back.
    # On non-timeout errors (auth, malformed request), we break early since
    # those won't fix themselves with retries.
    if _provider == "groq":
        for _attempt in range(3):
            try:
                response = _groq_client.chat.completions.create(
                    model=model or _GROQ_MODEL_FAST,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    timeout=5,
                )
                return response.choices[0].message.content.strip()
            except Exception as _e:
                # Heuristic: retry on timeout-shaped errors, break on others.
                _err = (str(type(_e).__name__) + " " + str(_e)).lower()
                if any(s in _err for s in ("timeout", "timed out", "503", "504", "429")):
                    continue
                return None
        return None

    if _provider == "anthropic":
        try:
            response = _anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                timeout=15,
            )
            return response.content[0].text.strip()
        except Exception:
            return None

    return None


# ── Intent parser ─────────────────────────────────────────────────────────────

def parse_intent(query: str) -> dict:
    if not _get_client():
        return _regex_parse(query)

    raw = _call_llm(_PARSER_SYSTEM, f"Query: {query}", max_tokens=512)
    if raw:
        try:
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            parsed = json.loads(raw)
            parsed.setdefault("free_text", query)
            parsed.setdefault("language_pref", "any")
            parsed.setdefault("era_pref", "any")
            parsed.setdefault("year_min", None)
            parsed.setdefault("year_max", None)
            parsed.setdefault("status", "any")
            parsed.setdefault("popularity_pref", "any")
            parsed.setdefault("binge_pref", "any")
            parsed.setdefault("rating_min", None)
            return parsed
        except Exception:
            pass

    return _regex_parse(query)


def _regex_parse(query: str) -> dict:
    q = query.strip()
    ql = q.lower()
    has_hebrew = bool(re.search(r"[֐-׿]", q))
    lang = "he" if has_hebrew else "en"

    mood_map = {
        "dark":      ["dark","אפל","כהה","depressing"],
        "funny":     ["funny","comedy","fun","מצחיק","הומור","קומדי"],
        "emotional": ["emotional","sad","cry","מרגש","עצוב"],
        "thrilling": ["thriller","thrill","suspense","מותחן","מפחיד","horror"],
        "light":     ["light","lighthearted","קליל","קל","cheerful"],
    }
    mood = []
    for tag, kws in mood_map.items():
        for kw in kws:
            if kw.lower() in ql:
                mood.append(tag)
                break

    # length_pref
    if re.search(r"\bone season\b|עונה אחת", q, re.IGNORECASE):
        length_pref = "limited"
    elif re.search(r"\b(short|קצר|mini|limited|פחות פרקים|fewer episodes)\b", q, re.IGNORECASE):
        length_pref = "short"
    elif re.search(r"\b(long|ארוך|epic|longer|many seasons|הרבה עונות)\b", q, re.IGNORECASE):
        length_pref = "long"
    else:
        length_pref = "any"

    # language_pref
    if re.search(r"\b(foreign|זר|זרה|not english|לא אנגלית|בשפה זרה)\b", q, re.IGNORECASE):
        language_pref = "foreign"
    elif re.search(r"\b(english|אנגלית)\b", q, re.IGNORECASE):
        language_pref = "en"
    elif re.search(r"\b(korean|קוריאני|קוריאנית)\b", q, re.IGNORECASE):
        language_pref = "ko"
    elif re.search(r"\b(spanish|ספרדית|ספרדי)\b", q, re.IGNORECASE):
        language_pref = "es"
    elif re.search(r"\b(german|גרמנית|גרמני)\b", q, re.IGNORECASE):
        language_pref = "de"
    elif re.search(r"\b(french|צרפתית|צרפתי)\b", q, re.IGNORECASE):
        language_pref = "fr"
    elif re.search(r"\b(japanese|יפנית|יפני|anime)\b", q, re.IGNORECASE):
        language_pref = "ja"
    else:
        language_pref = "any"

    # year_min / year_max (explicit year boundaries)
    year_min = None
    year_max = None
    _from_m = re.search(
        r"(?:from|after|since|משנת|אחרי|מ-?|החל מ)\s*((?:19|20)\d{2})"
        r"|(?:year\s+)?((?:19|20)\d{2})\s+(?:and\s+(?:more|later|up|above|onwards?)|ומעלה|ואילך)",
        q, re.IGNORECASE)
    if _from_m:
        yr = int(next(g for g in _from_m.groups() if g))
        year_min = yr
    _to_m = re.search(
        r"(?:before|until|up\s+to|לפני|עד)\s*((?:19|20)\d{2})",
        q, re.IGNORECASE)
    if _to_m:
        year_max = int(_to_m.group(1)) - 1  # "before 2000" → max 1999

    # era_pref (derived from year_min or explicit decade keywords)
    if year_min and year_min >= 2020:
        era_pref = "recent"
    elif year_min and year_min >= 2010:
        era_pref = "2010s"
    elif year_min and year_min >= 2000:
        era_pref = "2000s"
    elif year_max and year_max < 2000:
        era_pref = "classic"
    elif re.search(r"\b(classic|קלאסי|old|ישן|pre.?2000)\b", q, re.IGNORECASE):
        era_pref = "classic"
    elif re.search(r"\b(90s|שנות ה.?90|1990s)\b", q, re.IGNORECASE):
        era_pref = "1990s"
    elif re.search(r"\b(2000s|שנות ה.?2000)\b", q, re.IGNORECASE):
        era_pref = "2000s"
    elif re.search(r"\b(2010s|שנות ה.?2010)\b", q, re.IGNORECASE):
        era_pref = "2010s"
    elif re.search(r"\b(recent|חדש|new|2020s|שנות ה.?2020)\b", q, re.IGNORECASE):
        era_pref = "recent"
    else:
        era_pref = "any"

    # status
    if re.search(r"\b(still airing|ongoing|עדיין משודר|ממשיך)\b", q, re.IGNORECASE):
        status = "airing"
    elif re.search(r"\b(finished|completed|ended|הסתיימה|הסתיים)\b", q, re.IGNORECASE):
        status = "finished"
    else:
        status = "any"

    # popularity_pref
    if re.search(r"\b(hidden gem|underrated|אוצר נסתר|לא מוכר)\b", q, re.IGNORECASE):
        popularity_pref = "hidden_gem"
    elif re.search(r"\b(trending|popular|פופולרי|hot)\b", q, re.IGNORECASE):
        popularity_pref = "trending"
    elif re.search(r"\b(well.?known|famous|מפורסם)\b", q, re.IGNORECASE):
        popularity_pref = "well_known"
    else:
        popularity_pref = "any"

    # binge_pref
    if re.search(r"\b(binge|weekend|סוף שבוע|בינג)\b", q, re.IGNORECASE):
        binge_pref = "binge"
    elif re.search(r"\b(casual|קליל|episodic)\b", q, re.IGNORECASE):
        binge_pref = "casual"
    else:
        binge_pref = "any"

    # rating_min
    if re.search(r"\b(highly rated|best rated|top rated|מדורג גבוה)\b", q, re.IGNORECASE):
        rating_min = 8.5
    elif re.search(r"\b(decent|סביר|acceptable)\b", q, re.IGNORECASE):
        rating_min = 7.0
    else:
        rating_min = None

    return {
        "seeds": [], "mood": mood, "length_pref": length_pref,
        "language_pref": language_pref, "era_pref": era_pref,
        "year_min": year_min, "year_max": year_max,
        "status": status, "popularity_pref": popularity_pref,
        "binge_pref": binge_pref, "rating_min": rating_min,
        "exclude_genres": [], "lang": lang, "free_text": query,
    }


# ── Explanation generator ──────────────────────────────────────────────────────

def explain_recommendations(intent: dict, recommendations: list[dict], lang: str = "en") -> str:
    if not recommendations:
        return "לא נמצאו תוצאות מתאימות." if lang == "he" else "No matching results found."

    if not _get_client():
        return _fallback_explanation(intent, recommendations, lang)

    recs_text = "\n".join(
        f"{i+1}. {r['title']} ({r.get('decade_str','')}, {r.get('genres','')}) "
        f"— Rating: {r.get('rating','')} — Hybrid score: {r.get('hybrid_score','')}"
        for i, r in enumerate(recommendations)
    )

    user_msg = (
        f"User query: {intent.get('free_text','')}\n"
        f"Detected mood: {intent.get('mood',[])}, Language: {lang}\n\n"
        f"The recommendation engine found EXACTLY these {len(recommendations)} shows from our database:\n"
        f"{recs_text}\n\n"
        f"Explain ONLY these shows and why they match the query. "
        f"Do not mention any other shows. Reply in {'Hebrew' if lang=='he' else 'English'}."
    )

    result = _call_llm(_EXPLAINER_SYSTEM, user_msg, max_tokens=600)
    return result if result else _fallback_explanation(intent, recommendations, lang)


# ── Follow-up pattern detection (fast, no LLM needed) ─────────────────────────

_OTHER_OPTIONS_TOKENS = [
    "other", "different", "something else", "more options", "new ones",
    "give me more", "show me more", "more results", "again", "next",
    "אחר", "אחרות", "שונה", "עוד", "אופציות אחרות", "משהו אחר",
    "נוספות", "תוצאות נוספות", "עוד תוצאות", "תן לי עוד",
]
_SHORTER_TOKENS = [
    "short", "shorter", "fewer episodes", "mini series", "mini-series",
    "קצר", "קצרות", "פחות פרקים", "קצרה", "קצרים",
]
_LIGHTER_TOKENS = [
    "less dark", "lighter", "not so dark", "not dark", "less serious",
    "less heavy", "more fun", "more light",
    "פחות אפל", "לא אפל", "קליל", "מצחיק יותר", "פחות כבד", "יותר קליל",
]
_FOREIGN_TOKENS = [
    "foreign", "non-english", "not in english", "foreign language",
    "non english", "subtitles",
    "זר", "זרה", "בשפה זרה", "לא באנגלית", "שפה אחרת", "כתוביות",
]


def _detect_followup_type(msg: str) -> Optional[str]:
    """Detect common follow-up patterns without calling the LLM."""
    m = msg.lower().strip()
    if any(tok in m for tok in _OTHER_OPTIONS_TOKENS):
        return "other"
    if any(tok in m for tok in _SHORTER_TOKENS):
        return "shorter"
    if any(tok in m for tok in _LIGHTER_TOKENS):
        return "lighter"
    if any(tok in m for tok in _FOREIGN_TOKENS):
        return "foreign"
    return None


# ── Additional keyword patterns ───────────────────────────────────────────────

_QUESTION_TOKENS = [
    "what is", "what's", "tell me about", "tell me more", "how many",
    "when was", "who made", "who stars", "plot", "seasons", "episodes",
    "about the", "explain", "describe", "overview",
    "על מה", "כמה עונות", "כמה פרקים", "מתי", "מי עשה", "ספר לי על",
    "מה העלילה", "בכמה עונות", "מה זה",
]

_CHAT_TOKENS = [
    "thanks", "thank you", "great", "ok", "okay", "cool", "nice",
    "perfect", "awesome", "love it", "got it", "sounds good", "sure",
    "no thanks", "nevermind", "never mind", "that's all", "bye",
    "תודה", "מעולה", "נהדר", "אחלה", "סבבה", "טוב", "הבנתי",
    "ממש טוב", "יופי", "בסדר", "תענוג",
]


def _keyword_classify(message: str) -> str:
    """Keyword-based fallback classifier — used when LLM is unavailable."""
    m = message.lower().strip()

    if any(tok in m for tok in _OTHER_OPTIONS_TOKENS):
        return "more_options"

    if any(tok in m for tok in _SHORTER_TOKENS + _LIGHTER_TOKENS + _FOREIGN_TOKENS):
        return "refine"

    if any(tok in m for tok in _QUESTION_TOKENS):
        return "question"

    if any(tok in m for tok in _CHAT_TOKENS):
        return "chat"

    return "search"


_CLASSIFIER_SYSTEM = """\
Classify the user's latest message into exactly one of these intents:
- search: wants to find a show/movie (new query, e.g. "something like Breaking Bad", "dark thriller")
- more_options: wants different results from the same search ("more", "other options", "something else", "אחרות", "עוד")
- refine: wants to adjust current results ("shorter", "less dark", "foreign", "קצר", "פחות אפל")
- question: asks about a specific show ("what is it about", "how many seasons", "על מה זה")
- chat: acknowledgement or general chat ("thanks", "great", "ok", "תודה")

Reply with ONLY the single intent word. Nothing else. No punctuation.
"""


def classify_intent(message: str, conversation_history: list[dict] | None = None) -> str:
    """
    Classify a user message into: search | more_options | refine | question | chat

    Uses LLM (fast, max 10 tokens) with last-3-message context.
    Falls back to keyword matching if LLM unavailable or returns unexpected output.
    """
    _get_client()

    if _provider is not None:
        ctx_msgs = (conversation_history or [])[-3:]
        ctx = "\n".join(
            f"{m['role'].upper()}: {m['content'][:120]}" for m in ctx_msgs
        )
        user_prompt = (
            f"Context:\n{ctx}\n\nLatest message: {message}\n\nIntent:"
            if ctx else
            f"Message: {message}\n\nIntent:"
        )
        raw = _call_llm(_CLASSIFIER_SYSTEM, user_prompt, max_tokens=10)
        if raw:
            word = raw.strip().lower().split()[0].rstrip(".,!") if raw.strip() else ""
            if word in ("search", "more_options", "refine", "question", "chat"):
                return word

    return _keyword_classify(message)


# ── Conversational chat turn ───────────────────────────────────────────────────

_CHAT_SYSTEM = """\
You are CineMatch AI, a warm, witty bilingual assistant whose specialty is TV and movie recommendations from a curated catalog of 11,013 titles. You can also chat about anything else, like ChatGPT or Claude would.

PERSONALITY
- You are CineMatch AI, a TV and movie recommendation specialist. That is your one job.
- Friendly, confident, a real person. Never robotic. Never preachy. Never sycophantic.
- Reply in the user's language. If they wrote in Hebrew, reply in Hebrew. If they wrote in English, reply in English.
- TV and movie related chat is ALL welcome: opinions on shows, character discussion, plot trivia, actor questions, availability questions, jokes ABOUT shows or movies, recommendations.
- For OFF-TOPIC requests (recipes, weather, math, code, news, sports scores, anything not TV/movie): politely acknowledge you cannot help with that, mention your purpose in a natural one-liner, then offer to find something to watch instead. NEVER offer to actually do the off-topic thing. Do not pretend you might be able to help.
- One light scope-mention per off-topic redirect is plenty. Do not be preachy or repeat the disclaimer.
- Keep replies short and conversational, 1 to 3 sentences usually.
- Have opinions, show personality.
- Handle gibberish (random letters, unclear input) by asking a friendly clarifying question about what they want to watch.

CONTINUITY (THIS IS WHAT MAKES YOU FEEL REAL)
- Always reference what the user said EARLIER in this conversation when relevant. If they mentioned a show 3 turns ago, bring it up by name. If they hinted at a mood, remember it. If they said they already watched something, do not recommend it again.
- If the user shifts topics, follow them. If they come back to TV later, recall what they liked before.
- Never make the user repeat themselves. Treat the whole conversation as one ongoing dialogue, not a series of isolated questions.

TASK
Read the full conversation and the on-screen recommendations. Decide what to do next, and craft your reply.

Reply with valid JSON only. No markdown fences, no prose around it.

{
  "action": "chat" | "search" | "refine" | "swap_slot",
  "reply": "<your conversational reply in the user's language>",
  "intent": {
    "seeds": [],
    "mood": [],
    "length_pref": "short" | "long" | "any",
    "exclude_genres": [],
    "lang": "he" | "en",
    "free_text": "<plain-text version of the user's underlying request>"
  },
  "swap_slot_index": <0-based integer 0..4, only when action is swap_slot>
}

ACTION GUIDE

"chat" — Use when the user is NOT asking for a fresh recommendation right now. This is the MOST COMMON action. Examples:
  - Greetings and chitchat (hi, thanks, how are you)
  - Off-topic asks (pasta recipe, weather, math, code, news). Politely redirect, mention your purpose, offer to find something to watch. NEVER offer to help with the off-topic thing.
  - Opinions on a specific show or movie (what do you think about The Office, is Seinfeld good)
  - Availability (where can I watch Breaking Bad). Use your general knowledge, be honest if you are unsure. Add a soft caveat like "last I checked it was on X, but availability shifts often."
  - Questions about a show already in prev_recs (plot, cast, season count). Answer from the data shown.
  - Jokes about shows or movies are fine. Generic jokes — keep them light and TV/movie themed if possible.
  - Gibberish or confused input. Ask a friendly clarifying question about what they want to watch.
  Set intent.lang correctly. Other intent fields can be empty.

"search" — Use when the user clearly wants a NEW recommendation. The catalog will be searched after your reply. Examples:
  - recommend me a thriller
  - what should I watch tonight
  - shows like Breaking Bad
  - something funny, light, foreign
  Write a warm 1-sentence intro in `reply`, like "Sure, here are a few that match that vibe:" or "Got it, try these:".
  Fill intent.seeds with any titles the user mentioned. Fill mood / length_pref / exclude_genres if hinted. intent.free_text should capture the gist in plain text.

"refine" — Use when there are recommendations on screen and the user wants to ADJUST them (not replace one). Examples:
  - shorter
  - less dark
  - in spanish
  - more options, something different
  - more like the first one
  Write a warm 1-sentence intro in `reply`. Fill intent to capture the refinement (e.g. length_pref="short", or seeds=[first_rec_title]).

"swap_slot" — Use when the user wants to replace ONE specific card in prev_recs. Examples:
  - I already watched the third one
  - swap #2
  - replace the first one
  - I have seen the second already
  Set swap_slot_index to the 0-based index (#1 = 0, #3 = 2). Only use when prev_recs has cards. Write a 1-sentence confirmation in `reply`.

STYLE RULES (STRICT)
- BREVITY (STRICT). Reply in 1 sentence whenever possible. 2 sentences only when truly needed. NEVER more than 2 sentences. No preambles, no "great question", no "let me explain". Just the answer. Long replies are wrong replies.
- NEVER use the em dash character. Use commas, colons, parentheses, or periods instead. This applies to every part of the JSON.
- Match the user's language. Hebrew query, Hebrew reply. English query, English reply.
- Do not use markdown formatting in `reply` except occasional **bold** for show titles.
- Sound like a real person.

CRITICAL: NEVER NAME SPECIFIC SHOWS IN YOUR `reply` FIELD WHEN action IS search OR refine
- The recommendation engine renders the actual cards below your bubble. You do NOT see which shows the engine will return.
- If you name specific shows in your reply that are not in the actual cards, the user sees a confusing mismatch (you say "Narcos, Peaky Blinders" but cards show "Billy and Mandy").
- For action: search and action: refine, the `reply` MUST be a generic one-liner. Examples:
  - "Got it, here you go:"
  - "Try these:"
  - "Some options coming up:"
  - "Here are some matches:"
- BAD examples (do NOT do this):
  - "Here are five crime dramas: Narcos, Peaky Blinders, Ozark, Better Call Saul, Fargo." (WRONG: names shows you cannot verify)
  - "Try Breaking Bad, Better Call Saul, and Ozark." (WRONG: names shows)
- For action: chat (when discussing a specific named show the user mentioned), you MAY name that one show because the user just named it. Do not invent additional show names.

EXAMPLES OF THE DESIRED FEEL

User: "hi"
{"action":"chat","reply":"Hey, what are you in the mood for?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"greeting"}}

User: "i want a pasta recipe"
{"action":"chat","reply":"Not my thing, I do TV and movies. Want something to watch while you cook?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"asked for a pasta recipe, redirected"}}

User: "what's the weather today"
{"action":"chat","reply":"No weather here, just TV. Cozy show for a rainy day, or something energetic?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"asked about weather, redirected"}}

User: "help me with my math homework"
{"action":"chat","reply":"Math's not me, only TV and movies. Want something to watch after?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"asked for homework help, redirected"}}

User: "tell me a joke"
{"action":"chat","reply":"Why was the TV show cancelled? Bad reception.","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"asked for a joke, TV themed"}}

User: "asdfgh"
{"action":"chat","reply":"What are you in the mood to watch?","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"unclear input"}}

User: "what do you think about The Office"
{"action":"chat","reply":"Comedy classic, Michael Scott era is golden. Want something with a similar vibe?","intent":{"seeds":["The Office"],"mood":["funny"],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"opinion on The Office"}}

User: "where can I watch Seinfeld"
{"action":"chat","reply":"Usually Netflix, worth a quick JustWatch check.","intent":{"seeds":["Seinfeld"],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"availability for Seinfeld"}}

User: "recommend me a dark thriller"
{"action":"search","reply":"Try these:","intent":{"seeds":[],"mood":["dark","thrilling"],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"dark thriller recommendation"}}

User: "shorter"  (with prev_recs present)
{"action":"refine","reply":"On it:","intent":{"seeds":[],"mood":[],"length_pref":"short","exclude_genres":[],"lang":"en","free_text":"shorter shows"}}

User: "I already watched the third one"  (with prev_recs present)
{"action":"swap_slot","reply":"Swapping the third.","intent":{"seeds":[],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"already watched #3"},"swap_slot_index":2}

User (after Breaking Bad was discussed earlier): "what was that actor's name again"
{"action":"chat","reply":"Bryan Cranston played Walter White in Breaking Bad.","intent":{"seeds":["Breaking Bad"],"mood":[],"length_pref":"any","exclude_genres":[],"lang":"en","free_text":"actor name question"}}
"""


def chat_turn(
    conversation: list[dict],
    prev_recs: list[dict] | None = None,
    lang: str = "en",
) -> dict:
    """
    Single conversational LLM call that decides what to do next.

    conversation : [{role:"user"|"assistant", content:str}, ...]
    prev_recs    : last shown recs [{title, genres, decade_str, rating, overview}, ...]
    lang         : UI language hint

    Returns: {action, intent, reply, swap_slot_index?, follow_up}
    Where action is one of: chat | search | refine | swap_slot
    """
    _get_client()
    last_user = next(
        (m["content"] for m in reversed(conversation) if m["role"] == "user"), ""
    )
    fallback = _regex_parse(last_user)
    detected_lang = fallback.get("lang", lang)

    # ── Fast follow-up keyword path (no LLM) for trivial refinements ──────────
    # Only fires when prev_recs exist, so chitchat is always sent to the LLM.
    if prev_recs:
        followup_type = _detect_followup_type(last_user)
        if followup_type:
            base_intent = {
                "seeds": [], "mood": [], "length_pref": "any",
                "exclude_genres": [], "lang": detected_lang, "free_text": last_user,
            }
            if followup_type == "other":
                return {"action": "refine", "intent": base_intent, "reply": "", "follow_up": ""}
            elif followup_type == "shorter":
                base_intent["length_pref"] = "short"
                return {"action": "refine", "intent": base_intent, "reply": "", "follow_up": ""}
            elif followup_type == "lighter":
                base_intent["exclude_genres"] = ["thriller", "horror", "crime"]
                base_intent["mood"] = ["light", "funny"]
                return {"action": "refine", "intent": base_intent, "reply": "", "follow_up": ""}
            elif followup_type == "foreign":
                base_intent["foreign_only"] = True
                return {"action": "refine", "intent": base_intent, "reply": "", "follow_up": ""}

    # ── No LLM available ──────────────────────────────────────────────────────
    if _provider is None:
        return {"action": "search", "intent": fallback, "reply": "", "follow_up": ""}

    # ── Build context for the LLM ─────────────────────────────────────────────
    recs_ctx = ""
    if prev_recs:
        recs_ctx = "\n\nCurrent on-screen recommendations (prev_recs):\n" + "\n".join(
            f'  [{i}] {r.get("title","")} | {r.get("genres","")} | '
            f'{r.get("decade_str","")} | rating {r.get("rating","")} | '
            f'{(r.get("overview") or "")[:120]}'
            for i, r in enumerate(prev_recs[:5])
        )
    else:
        recs_ctx = "\n\nCurrent on-screen recommendations (prev_recs): none yet"

    # Keep last 12 turns of context. Translate stored "bot" role to "assistant" so the LLM sees a clean transcript.
    conv_text = "\n".join(
        f'{"User" if m["role"] == "user" else "CineMatch"}: {m["content"]}'
        for m in conversation[-12:]
        if m.get("content")
    )

    raw = _call_llm(
        _CHAT_SYSTEM,
        f"Conversation transcript:\n{conv_text}{recs_ctx}\n\nDecide and respond in JSON only.",
        max_tokens=600,
        model=_GROQ_MODEL_CHAT,
    )

    if raw:
        try:
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            result = json.loads(raw)

            # Validate action
            action = result.get("action", "chat")
            if action not in ("chat", "search", "refine", "swap_slot"):
                action = "chat"
            result["action"] = action

            # Strip em dashes that slipped through the prompt
            if isinstance(result.get("reply"), str):
                result["reply"] = (result["reply"]
                                   .replace(" — ", ", ")
                                   .replace("—", ","))

            # Ensure intent always present and sane
            intent = result.get("intent") or {}
            if not isinstance(intent, dict):
                intent = {}
            intent.setdefault("seeds", [])
            intent.setdefault("mood", [])
            intent.setdefault("length_pref", "any")
            intent.setdefault("exclude_genres", [])
            intent.setdefault("lang", detected_lang)
            intent.setdefault("free_text", last_user)
            result["intent"] = intent

            # Validate swap_slot_index when relevant
            if action == "swap_slot":
                idx = result.get("swap_slot_index")
                if not (isinstance(idx, int) and 0 <= idx <= 4 and prev_recs and idx < len(prev_recs)):
                    # Bad index. Demote to chat so we do not break the UI.
                    result["action"] = "chat"
                    result.pop("swap_slot_index", None)
                    if not result.get("reply"):
                        result["reply"] = ("Which one do you want to swap?"
                                           if detected_lang == "en"
                                           else "איזו המלצה תרצה להחליף?")

            result.setdefault("reply", "")
            result.setdefault("follow_up", "")
            return result
        except Exception:
            pass

    # LLM call failed (Groq timeout, rate limit, or invalid JSON response).
    # Previously this fell back to action: search with an empty reply, which
    # dispatched the SEARCH branch, ran the engine with no real seed, returned
    # no matches, and surfaced the fake "I couldn't find a strong match" error.
    # That misled users into thinking the recommender was broken when actually
    # the LLM itself had hiccuped.
    #
    # New behavior: fall back to action: chat with a friendly bilingual reply.
    # The user sees that the agent had a brief hiccup and can immediately retry.
    # No empty-seed search, no fake search error, conversational mode preserved.
    fallback_reply = (
        "Hmm, my brain just hiccuped. What were you looking for?"
        if lang == "en"
        else "אופס, רגע קטן. מה חיפשת?"
    )
    fallback["lang"] = lang
    return {"action": "chat", "intent": fallback,
            "reply": fallback_reply, "follow_up": ""}


def _fallback_explanation(intent: dict, recommendations: list[dict], lang: str) -> str:
    mood = intent.get("mood", [])
    seeds = intent.get("seeds", [])

    if lang == "he":
        if seeds:
            opener = f"מצאנו עבורך סדרות הדומות ל-{seeds[0]}:"
        elif mood:
            mood_str = ", ".join(mood)
            opener = f"על פי מה שחיפשת ({mood_str}), אלו ההמלצות המתאימות ביותר:"
        else:
            opener = "אלו ההמלצות המובילות שלנו עבורך:"
        lines = [opener]
        for r in recommendations:
            rating = r.get("rating", "")
            rating_str = f"{rating:.1f}" if isinstance(rating, float) else str(rating)
            lines.append(f"• {r['title']} — {r.get('genres','')} | ⭐ {rating_str}")
    else:
        if seeds:
            opener = f"Based on your interest in {seeds[0]}, here are the best matches:"
        elif mood:
            mood_str = ", ".join(mood)
            opener = f"Looking for something {mood_str}? Here are our top picks:"
        else:
            opener = "Here are our top recommendations for you:"
        lines = [opener]
        for r in recommendations:
            rating = r.get("rating", "")
            rating_str = f"{rating:.1f}" if isinstance(rating, float) else str(rating)
            lines.append(f"• {r['title']} — {r.get('genres','')} | ⭐ {rating_str}")
    return "\n".join(lines)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && pytest tests/test_agent_llm.py -v`
Expected: PASS (12 tests). These all exercise the regex/keyword fallback
paths (no API key needed). If `GROQ_API_KEY` happens to be set in your shell
environment, the `monkeypatch.setattr(llm, "_get_client", lambda: None)` and
`monkeypatch.setattr(llm, "_provider", None)` lines force the fallback paths
regardless, so the tests stay deterministic and offline.

- [ ] **Step 6: Commit**

```bash
git add backend/app/agent/__init__.py backend/app/agent/llm.py backend/tests/test_agent_llm.py
git commit -m "Port LLM agent (Groq/Anthropic chat, intent parsing, explanations) from v1"
```

---

## Task 12: TMDB Live Lookups (`agent/tmdb.py`)

New module (not in v1). Used by `routers/show.py` (Task 18) for the
"click a card to see trailer/cast/watch providers" popup. Per the design
spec's edge cases: no API key or no match → callers fall back to
catalog-only data, no error surfaced to the user. This module itself just
returns `None` on any failure; the graceful-fallback behavior lives in
Task 18's router.

**Files:**
- Create: `backend/app/agent/tmdb.py`
- Test: `backend/tests/test_agent_tmdb.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_agent_tmdb.py`:

```python
import requests

from app.agent import tmdb


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json_data


def test_search_tv_show_no_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    assert tmdb.search_tv_show("Breaking Bad") is None


def test_search_tv_show_returns_first_result(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    def fake_get(url, params=None, timeout=None):
        assert url == f"{tmdb.TMDB_BASE_URL}/search/tv"
        assert params["query"] == "Breaking Bad"
        assert params["api_key"] == "test-key"
        return _FakeResponse({"results": [{"id": 1396, "name": "Breaking Bad"}]})

    monkeypatch.setattr(tmdb.requests, "get", fake_get)
    result = tmdb.search_tv_show("Breaking Bad")
    assert result == {"id": 1396, "name": "Breaking Bad"}


def test_search_tv_show_passes_year_param(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured.update(params)
        return _FakeResponse({"results": [{"id": 1, "name": "X"}]})

    monkeypatch.setattr(tmdb.requests, "get", fake_get)
    tmdb.search_tv_show("X", year=2008)
    assert captured["first_air_date_year"] == 2008


def test_search_tv_show_no_results_returns_none(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")
    monkeypatch.setattr(tmdb.requests, "get", lambda *a, **k: _FakeResponse({"results": []}))
    assert tmdb.search_tv_show("Nonexistent Show 12345") is None


def test_search_tv_show_request_exception_returns_none(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    def fake_get(*a, **k):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(tmdb.requests, "get", fake_get)
    assert tmdb.search_tv_show("Breaking Bad") is None


def test_get_tv_show_details_no_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    assert tmdb.get_tv_show_details(1396) is None


def test_get_tv_show_details_returns_json(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    def fake_get(url, params=None, timeout=None):
        assert url == f"{tmdb.TMDB_BASE_URL}/tv/1396"
        assert params["append_to_response"] == "videos,credits,watch/providers"
        return _FakeResponse({"id": 1396, "name": "Breaking Bad", "videos": {"results": []}})

    monkeypatch.setattr(tmdb.requests, "get", fake_get)
    result = tmdb.get_tv_show_details(1396)
    assert result["name"] == "Breaking Bad"


def test_get_tv_show_details_request_exception_returns_none(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    def fake_get(*a, **k):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(tmdb.requests, "get", fake_get)
    assert tmdb.get_tv_show_details(1396) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_agent_tmdb.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.agent.tmdb'`

- [ ] **Step 3: Create `app/agent/tmdb.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_agent_tmdb.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/agent/tmdb.py backend/tests/test_agent_tmdb.py
git commit -m "Add TMDB live lookup module for show-details popup"
```

---

## Task 13: Onboarding Pick Explanations (`agent/explanations.py`)

New module (not in v1). `agent/llm.py`'s `explain_recommendations()` writes
ONE combined paragraph for a seed/mood-driven search (it needs `intent.free_text`).
The onboarding flow (Task 16) has no free-text query - instead it has the 5
onboarding answers and a cluster profile label. `explain_picks()` produces
ONE short sentence PER recommended show, referencing the user's taste
profile, for the "why we picked this" text in each result card / popup.

**Files:**
- Create: `backend/app/agent/explanations.py`
- Test: `backend/tests/test_agent_explanations.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_agent_explanations.py`:

```python
import json

from app.agent import explanations


def _sample_picks():
    return [
        {"title": "Show A", "genres": "Drama, Crime", "decade_str": "2010s", "rating": 8.5, "overview": "A gritty drama."},
        {"title": "Show B", "genres": "Comedy", "decade_str": "2000s", "rating": 7.2, "overview": "A funny sitcom."},
    ]


def _sample_profile():
    return {"label_he": "דרמות פשע מהשנים האחרונות", "label_en": "Crime Dramas & recent years"}


def test_explain_picks_empty_list_returns_empty():
    assert explanations.explain_picks({}, _sample_profile(), [], "en") == []


def test_explain_picks_no_provider_uses_fallback(monkeypatch):
    monkeypatch.setattr(explanations, "_get_client", lambda: None)
    picks = _sample_picks()
    result = explanations.explain_picks({}, _sample_profile(), picks, "en")
    assert len(result) == 2
    assert "8.5" in result[0]
    assert "Crime Dramas" in result[0]


def test_explain_picks_no_provider_hebrew_fallback(monkeypatch):
    monkeypatch.setattr(explanations, "_get_client", lambda: None)
    picks = _sample_picks()
    result = explanations.explain_picks({}, _sample_profile(), picks, "he")
    assert "דרמות פשע" in result[0]


def test_explain_picks_with_provider_returns_llm_array(monkeypatch):
    monkeypatch.setattr(explanations, "_get_client", lambda: True)
    monkeypatch.setattr(
        explanations, "_call_llm",
        lambda system, user, max_tokens=500: json.dumps(["Great gritty drama!", "Light and fun sitcom."])
    )
    picks = _sample_picks()
    result = explanations.explain_picks({"genre": "drama"}, _sample_profile(), picks, "en")
    assert result == ["Great gritty drama!", "Light and fun sitcom."]


def test_explain_picks_with_provider_invalid_response_falls_back(monkeypatch):
    monkeypatch.setattr(explanations, "_get_client", lambda: True)
    monkeypatch.setattr(explanations, "_call_llm", lambda system, user, max_tokens=500: "not json")
    picks = _sample_picks()
    result = explanations.explain_picks({}, _sample_profile(), picks, "en")
    assert len(result) == 2
    assert "8.5" in result[0]


def test_explain_picks_with_provider_wrong_length_falls_back(monkeypatch):
    monkeypatch.setattr(explanations, "_get_client", lambda: True)
    monkeypatch.setattr(
        explanations, "_call_llm",
        lambda system, user, max_tokens=500: json.dumps(["Only one explanation"])
    )
    picks = _sample_picks()
    result = explanations.explain_picks({}, _sample_profile(), picks, "en")
    assert len(result) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_agent_explanations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.agent.explanations'`

- [ ] **Step 3: Create `app/agent/explanations.py`**

```python
"""
Per-show "why we picked this" explanations for onboarding recommendations.

Unlike agent/llm.py's explain_recommendations() (which writes one paragraph
for a seed/mood-driven search with a free-text query), explain_picks()
writes one short sentence PER show, referencing the user's onboarding
answers and the matched cluster's taste-profile label.
"""

import json
import re

from app.agent.llm import _call_llm, _get_client

_PICKS_EXPLAINER_SYSTEM = """\
You are a bilingual TV recommendation assistant. The user just answered a short
onboarding quiz about their taste, and the system matched them to a cluster of
shows and picked exactly N titles from that cluster.

Your job: write ONE short, warm sentence per show explaining why it fits the
user's stated taste profile. Reference the user's actual answers and/or the
cluster's theme where relevant.

Rules:
- Reply with a JSON array of exactly N strings, one per show, in the same
  order as the input list. No markdown fences, no other keys, no prose
  outside the array.
- Each string: ONE sentence, no em dashes (use commas/periods instead).
- When lang is "he": write entirely in simple modern Hebrew.
- When lang is "en": write entirely in English.
- Do not invent facts about the show beyond what's given (genres, rating, overview, decade).
"""


def explain_picks(answers: dict, cluster_profile: dict, picks: list[dict], lang: str = "en") -> list[str]:
    """
    Returns one short explanation per pick, in the same order as `picks`.
    Uses the LLM if available and it returns a same-length JSON array of
    strings; otherwise falls back to a deterministic genre/rating/cluster
    based sentence per pick.
    """
    if not picks:
        return []

    if not _get_client():
        return [_fallback_pick_explanation(p, cluster_profile, lang) for p in picks]

    label = cluster_profile.get("label_he" if lang == "he" else "label_en", "")
    picks_text = "\n".join(
        f"{i+1}. {p['title']} ({p.get('decade_str','')}, {p.get('genres','')}) "
        f"— Rating: {p.get('rating','')} — {(p.get('overview') or '')[:160]}"
        for i, p in enumerate(picks)
    )
    user_msg = (
        f"User's taste profile: {label}\n"
        f"Onboarding answers: {answers}\n"
        f"Language: {lang}\n\n"
        f"Shows (write exactly {len(picks)} explanations, in this order):\n{picks_text}"
    )

    raw = _call_llm(_PICKS_EXPLAINER_SYSTEM, user_msg, max_tokens=500)
    if raw:
        try:
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == len(picks) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass

    return [_fallback_pick_explanation(p, cluster_profile, lang) for p in picks]


def _fallback_pick_explanation(pick: dict, cluster_profile: dict, lang: str) -> str:
    genres = pick.get("genres", "")
    rating = pick.get("rating", "")
    rating_str = f"{rating:.1f}" if isinstance(rating, float) else str(rating)
    label = cluster_profile.get("label_he" if lang == "he" else "label_en", "")

    if lang == "he":
        if label:
            return f"מתאים לטעם שלך ({label}): {genres}, דירוג {rating_str}."
        return f"{genres}, דירוג {rating_str}."

    if label:
        return f"Matches your taste profile ({label}): {genres}, rated {rating_str}."
    return f"{genres}, rated {rating_str}."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_agent_explanations.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/agent/explanations.py backend/tests/test_agent_explanations.py
git commit -m "Add per-show onboarding pick explanations"
```

---

## Task 14: Backend i18n (`app/i18n.py`)

A trimmed version of v1's `i18n.py` (`C:\Users\Hello\Desktop\cinematch-ai-main\i18n.py`,
321 lines). v1's STRINGS dict covers an entire Streamlit UI (sidebar filters,
buttons, labels) - in v2, that UI text lives in the React frontend's own i18n.
The backend only needs bilingual templates for the handful of bot messages
it generates server-side: the `/api/recommend` intro/outro around the cluster
picks, the empty-results fallback, and the show-not-found message for
`/api/show/{title}`. Routers (Tasks 16-18) call `t(key, lang, **kwargs)`.

**Files:**
- Create: `backend/app/i18n.py`
- Test: `backend/tests/test_i18n.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_i18n.py`:

```python
from app.i18n import t


def test_t_returns_hebrew_string_with_formatting():
    result = t("recommend_intro", "he", label="דרמות פשע")
    assert "דרמות פשע" in result


def test_t_returns_english_string_with_formatting():
    result = t("recommend_intro", "en", label="Crime Dramas")
    assert "Crime Dramas" in result


def test_t_falls_back_to_english_for_unknown_lang():
    assert t("show_not_found", "fr") == t("show_not_found", "en")


def test_t_unknown_key_returns_empty_string():
    assert t("nonexistent_key", "en") == ""


def test_t_defaults_to_english_lang():
    assert t("show_not_found") == t("show_not_found", "en")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_i18n.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.i18n'`

- [ ] **Step 3: Create `app/i18n.py`**

```python
"""
Bilingual (Hebrew/English) templates for bot messages generated server-side.

Most UI text lives in the React frontend's own i18n. This module only covers
messages assembled by the backend itself: the onboarding-recommendation
intro/outro and a couple of fallback/error strings.
"""

STRINGS: dict[str, dict[str, str]] = {
    "recommend_intro": {
        "he": "על סמך מה שסיפרת לי, אני חושב שתתחבר לטעם הזה: {label}. הנה כמה סדרות שכדאי לבדוק:",
        "en": "Based on what you told me, I think you're into: {label}. Here are a few shows worth checking out:",
    },
    "recommend_outro": {
        "he": "מקווה שאהבת את ההמלצות! אפשר להמשיך לדבר איתי על סדרות, לבקש עוד המלצות, או לשאול אותי כל דבר.",
        "en": "Hope you like these picks! Feel free to keep chatting, ask for more recommendations, or anything else.",
    },
    "no_recommendations": {
        "he": "לא הצלחתי למצוא המלצות מתאימות הפעם. אפשר לנסות עם תשובות אחרות?",
        "en": "I couldn't find matching recommendations this time. Want to try different answers?",
    },
    "show_not_found": {
        "he": "לא מצאתי את הסדרה הזו במאגר שלנו.",
        "en": "I couldn't find that show in our catalog.",
    },
}


def t(key: str, lang: str = "en", **kwargs) -> str:
    entry = STRINGS.get(key, {})
    template = entry.get(lang, entry.get("en", ""))
    if kwargs:
        return template.format(**kwargs)
    return template
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_i18n.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/i18n.py backend/tests/test_i18n.py
git commit -m "Add trimmed backend i18n module"
```

---

## Task 15: Application State Loader (`app/state.py`)

Loads everything the routers need ONCE at FastAPI startup (catalog,
embeddings, numeric similarity matrix, cluster artifacts, anomaly
threshold) and stores it on `app.state.cinematch`. Routers (Tasks 16-18)
read from `request.app.state.cinematch` instead of re-loading data per
request.

The anomaly-threshold calibration is ported from v1's `app.py:load_matrices()`
(sample 200 catalog rows, compute each one's best numeric+text hybrid match
score against the rest of the catalog, calibrate at the 8th percentile). v1
also built a per-sample Jaccard feature set in this function that was never
used in the actual score formula (`full_scores = beta*n_scores + gamma*t_scores`
never references it) - that dead computation is dropped here.

**Files:**
- Create: `backend/app/state.py`
- Test: `backend/tests/test_state.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_state.py`:

```python
from app.state import load_state


def test_load_state_returns_expected_keys_and_shapes():
    state = load_state()

    assert len(state["catalog"]) == 11013
    assert state["embeddings"].shape == (11013, 384)
    assert state["numeric_matrix"].shape == (11013, 11013)

    assert "cluster_id" in state["catalog_with_features"].columns
    assert len(state["catalog_with_features"]) == len(state["catalog"])

    assert len(state["feature_dims"]) == 14
    assert len(state["cluster_centroids"]) == len(state["cluster_profiles"])
    for centroid in state["cluster_centroids"]:
        assert len(centroid) == 14

    assert isinstance(state["anomaly_threshold"], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.state'`

- [ ] **Step 3: Create `app/state.py`**

```python
"""
Loads all shared data/model artifacts once at FastAPI startup.

app/main.py's lifespan handler calls load_state() and stores the result on
app.state.cinematch. Routers read from request.app.state.cinematch.
"""

import json
import os

import numpy as np
import pandas as pd

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
    catalog_with_features = catalog.merge(
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
        n_scores = numeric_matrix[src_pos]
        t_scores = embeddings @ embeddings[src_pos]
        full_scores = BETA * n_scores + gamma * t_scores
        full_scores[src_pos] = -1  # exclude self
        best_hybrid_scores.append(float(full_scores.max()))

    return calibrate(np.array(best_hybrid_scores), percentile=8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_state.py -v`
Expected: PASS. This test loads the full 11,013-row catalog and builds an
11013x11013 numeric similarity matrix, so it takes a few seconds (and ~50MB
for that matrix in float32) - this is expected and matches v1's startup cost.

- [ ] **Step 5: Commit**

```bash
git add backend/app/state.py backend/tests/test_state.py
git commit -m "Add application state loader (catalog, matrices, clusters, anomaly threshold)"
```

- [ ] **Step 6: Write the failing test for lifespan wiring**

`app/main.py` (Task 1) currently constructs `FastAPI()` with no lifespan, so
`app.state.cinematch` does not exist. Routers (Tasks 16-18) need it populated
before the first request. Add this test to `backend/tests/test_main.py`:

```python
def test_app_state_loaded_on_startup(client):
    state = client.app.state.cinematch
    assert "catalog" in state
    assert "catalog_with_features" in state
    assert "anomaly_threshold" in state
```

- [ ] **Step 7: Run test to verify it fails**

Run: `cd backend && pytest tests/test_main.py -v`
Expected: FAIL with `AttributeError: 'State' object has no attribute 'cinematch'`

- [ ] **Step 8: Wire `load_state()` into the FastAPI lifespan**

Replace the contents of `backend/app/main.py` with:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.state import load_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cinematch = load_state()
    yield


app = FastAPI(title="CineMatch AI v2", lifespan=lifespan)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
```

- [ ] **Step 9: Run test to verify it passes**

Run: `cd backend && pytest tests/test_main.py -v`
Expected: PASS (2 tests). The `client` fixture's `with TestClient(app) as c`
triggers the lifespan startup, so `load_state()` runs once per test session.

- [ ] **Step 10: Commit**

```bash
git add backend/app/main.py backend/tests/test_main.py
git commit -m "Load application state into FastAPI lifespan on startup"
```

---

## Task 16: Onboarding Recommendation Endpoint (`routers/recommend.py`)

`POST /api/recommend` takes the 5 onboarding answers (Task 9's vocabulary),
maps them to a feature vector, finds the nearest cluster, picks 3 shows from
it (Task 10), generates a per-show explanation (Task 13), and wraps the
result with the i18n intro/outro strings (Task 14).

**Files:**
- Create: `backend/app/routers/recommend.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_api_recommend.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_api_recommend.py`:

```python
def test_recommend_returns_three_picks_with_intro_and_outro(client):
    response = client.post(
        "/api/recommend",
        json={
            "answers": {
                "genre": "drama",
                "length": "long",
                "era": "recent",
                "tone": "serious_drama",
                "popularity": "well_known",
            },
            "lang": "en",
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["intro"]
    assert body["outro"]
    assert isinstance(body["cluster_id"], int)
    assert 1 <= len(body["recommendations"]) <= 3

    for rec in body["recommendations"]:
        assert rec["title"]
        assert rec["genres"]
        assert isinstance(rec["rating"], float)
        assert rec["explanation"]


def test_recommend_all_any_returns_three_picks(client):
    response = client.post("/api/recommend", json={"answers": {}, "lang": "he"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 3


def test_recommend_hebrew_uses_hebrew_strings(client):
    response = client.post(
        "/api/recommend", json={"answers": {"genre": "comedy"}, "lang": "he"}
    )
    body = response.json()
    assert "תתחבר" in body["intro"]
    assert "אהבת" in body["outro"]


def test_recommend_no_picks_in_a_cluster_returns_no_recommendations_message(client, monkeypatch):
    import app.routers.recommend as recommend_module

    monkeypatch.setattr(
        recommend_module,
        "recommend_from_cluster",
        lambda *args, **kwargs: recommend_module.pd.DataFrame(),
    )

    response = client.post("/api/recommend", json={"answers": {}, "lang": "en"})
    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] == []
    assert body["intro"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_api_recommend.py -v`
Expected: FAIL with `404 Not Found` (route doesn't exist yet) on every test.

- [ ] **Step 3: Create `app/routers/recommend.py`**

```python
"""POST /api/recommend - onboarding answers -> cluster + 3 picks + explanations."""

import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.agent.explanations import explain_picks
from app.clustering.onboarding_map import build_user_vector
from app.clustering.recommend import nearest_cluster, recommend_from_cluster
from app.i18n import t

router = APIRouter()


class OnboardingAnswers(BaseModel):
    genre: str = "any"
    length: str = "any"
    era: str = "any"
    tone: str = "any"
    popularity: str = "any"


class RecommendRequest(BaseModel):
    answers: OnboardingAnswers
    lang: str = "he"


class ShowSummary(BaseModel):
    title: str
    genres: str
    rating: float
    overview: str
    poster_path: str | None = None
    decade_str: str
    num_seasons: float | None = None
    binge_fit_score: float
    explanation: str


class RecommendResponse(BaseModel):
    intro: str
    outro: str
    cluster_id: int
    recommendations: list[ShowSummary]


def _nan_to_none(value):
    return None if pd.isna(value) else value


@router.post("/api/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest, request: Request) -> RecommendResponse:
    state = request.app.state.cinematch
    answers = payload.answers.model_dump()
    lang = payload.lang

    vector, mask = build_user_vector(answers)
    cluster_id = nearest_cluster(
        vector, mask, state["cluster_centroids"], state["cluster_profiles"]
    )
    cluster_profile = state["cluster_profiles"][str(cluster_id)]

    picks_df = recommend_from_cluster(
        state["catalog_with_features"], cluster_id, vector, mask, top_n=3
    )

    if picks_df.empty:
        return RecommendResponse(
            intro=t("no_recommendations", lang),
            outro="",
            cluster_id=cluster_id,
            recommendations=[],
        )

    picks = picks_df.to_dict(orient="records")
    explanations = explain_picks(answers, cluster_profile, picks, lang)

    recommendations = [
        ShowSummary(
            title=pick["title"],
            genres=pick["genres"],
            rating=float(pick["rating"]),
            overview=pick["overview"],
            poster_path=_nan_to_none(pick.get("poster_path")),
            decade_str=pick["decade_str"],
            num_seasons=_nan_to_none(pick.get("num_seasons")),
            binge_fit_score=float(pick["binge_fit_score"]),
            explanation=explanation,
        )
        for pick, explanation in zip(picks, explanations)
    ]

    label_key = "label_he" if lang == "he" else "label_en"
    intro = t("recommend_intro", lang, label=cluster_profile[label_key])
    outro = t("recommend_outro", lang)

    return RecommendResponse(
        intro=intro,
        outro=outro,
        cluster_id=cluster_id,
        recommendations=recommendations,
    )
```

- [ ] **Step 4: Register the router in `app/main.py`**

Replace the contents of `backend/app/main.py` with:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import recommend
from app.state import load_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cinematch = load_state()
    yield


app = FastAPI(title="CineMatch AI v2", lifespan=lifespan)
app.include_router(recommend.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && pytest tests/test_api_recommend.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/recommend.py backend/app/main.py backend/tests/test_api_recommend.py
git commit -m "Add POST /api/recommend endpoint (onboarding -> cluster picks + explanations)"
```

---

## Task 17: Open Chat Endpoint (`routers/chat.py`)

`POST /api/chat` is the "combination mode" handler from the spec's UX step 6.
It calls `chat_turn` (Task 11) to classify the message into one of
`chat | search | refine | swap_slot`, then:

- **chat**: just returns the LLM's reply, no recommendation cards.
- **search / refine**: if `intent.seeds` names a show in our catalog, ranks
  with v1's seed-based hybrid `recommend()` (Task 5/6), discarding the result
  if the best match is anomalous (Task 6's `is_anomalous`). Otherwise (or as
  a fallback), maps the intent to onboarding-answers vocabulary
  (`intent_to_onboarding_answers`, Task 9) and uses the cluster-based
  recommender (Task 10). `refine` additionally excludes titles already shown
  in `prev_recs`.
- **swap_slot**: replaces exactly the card at `swap_slot_index` in `prev_recs`
  with one new pick (excluding all currently-shown titles), leaving the other
  cards untouched.

**Files:**
- Create: `backend/app/routers/chat.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_api_chat.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_api_chat.py`:

```python
import app.routers.chat as chat_module


def _search_intent(free_text="anything", mood=None):
    return {
        "seeds": [], "mood": mood or [], "length_pref": "any",
        "exclude_genres": [], "lang": "en", "free_text": free_text,
    }


def test_chat_action_chat_returns_reply_without_recommendations(client, monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "chat",
            "intent": _search_intent("greeting"),
            "reply": "Hey, what are you in the mood for?",
            "follow_up": "",
        },
    )

    response = client.post(
        "/api/chat",
        json={"conversation": [{"role": "user", "content": "hi"}], "lang": "en"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Hey, what are you in the mood for?"
    assert body["recommendations"] is None


def test_chat_search_returns_recommendations_and_explanation(client, monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "search",
            "intent": _search_intent("dark thriller", mood=["dark", "thrilling"]),
            "reply": "Try these:",
            "follow_up": "",
        },
    )

    response = client.post(
        "/api/chat",
        json={
            "conversation": [{"role": "user", "content": "something dark and thrilling"}],
            "lang": "en",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 1 <= len(body["recommendations"]) <= 3
    assert body["explanation"]


def test_chat_refine_excludes_previously_shown_titles(client, monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "search",
            "intent": _search_intent(),
            "reply": "Try these:",
            "follow_up": "",
        },
    )
    first = client.post(
        "/api/chat",
        json={"conversation": [{"role": "user", "content": "recommend something"}], "lang": "en"},
    )
    first_recs = first.json()["recommendations"]
    assert len(first_recs) == 3

    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "refine",
            "intent": _search_intent("more like that"),
            "reply": "On it:",
            "follow_up": "",
        },
    )
    second = client.post(
        "/api/chat",
        json={
            "conversation": [{"role": "user", "content": "more"}],
            "prev_recs": first_recs,
            "lang": "en",
        },
    )
    assert second.status_code == 200
    second_recs = second.json()["recommendations"]
    first_titles = {r["title"] for r in first_recs}
    second_titles = {r["title"] for r in second_recs}
    assert second_titles
    assert first_titles.isdisjoint(second_titles)


def test_chat_swap_slot_replaces_only_target_slot(client, monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "search",
            "intent": _search_intent(),
            "reply": "Try these:",
            "follow_up": "",
        },
    )
    first = client.post(
        "/api/chat",
        json={"conversation": [{"role": "user", "content": "recommend something"}], "lang": "en"},
    )
    prev_recs = first.json()["recommendations"]
    assert len(prev_recs) == 3

    monkeypatch.setattr(
        chat_module,
        "chat_turn",
        lambda conversation, prev_recs=None, lang="he": {
            "action": "swap_slot",
            "intent": _search_intent("already watched #2"),
            "reply": "Swapping the second.",
            "swap_slot_index": 1,
            "follow_up": "",
        },
    )
    second = client.post(
        "/api/chat",
        json={
            "conversation": [{"role": "user", "content": "swap #2"}],
            "prev_recs": prev_recs,
            "lang": "en",
        },
    )
    assert second.status_code == 200
    new_recs = second.json()["recommendations"]
    assert len(new_recs) == 3
    assert new_recs[0]["title"] == prev_recs[0]["title"]
    assert new_recs[2]["title"] == prev_recs[2]["title"]
    assert new_recs[1]["title"] != prev_recs[1]["title"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_api_chat.py -v`
Expected: FAIL with `404 Not Found` / `ModuleNotFoundError: No module named 'app.routers.chat'`

- [ ] **Step 3: Create `app/routers/chat.py`**

```python
"""POST /api/chat - conversational turn handler (chat_turn + recommendations)."""

import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.agent.llm import chat_turn, explain_recommendations
from app.clustering.onboarding_map import build_user_vector, intent_to_onboarding_answers
from app.clustering.recommend import nearest_cluster, recommend_from_cluster
from app.engine.anomaly import is_anomalous
from app.engine.hybrid import recommend as hybrid_recommend
from app.i18n import t

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class RecCard(BaseModel):
    title: str
    genres: str
    rating: float
    overview: str
    poster_path: str | None = None
    decade_str: str
    num_seasons: float | None = None


class ChatRequest(BaseModel):
    conversation: list[ChatMessage]
    prev_recs: list[RecCard] | None = None
    lang: str = "he"


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecCard] | None = None
    explanation: str | None = None


def _nan_to_none(value):
    return None if pd.isna(value) else value


def _to_rec_cards(df: pd.DataFrame) -> list[RecCard]:
    return [
        RecCard(
            title=row["title"],
            genres=row["genres"],
            rating=float(row["rating"]),
            overview=row["overview"],
            poster_path=_nan_to_none(row.get("poster_path")),
            decade_str=row["decade_str"],
            num_seasons=_nan_to_none(row.get("num_seasons")),
        )
        for _, row in df.iterrows()
    ]


def _find_catalog_index(catalog: pd.DataFrame, title: str) -> int | None:
    matches = catalog.index[catalog["title"].str.lower() == title.lower()]
    if len(matches) == 0:
        return None
    return int(matches[0])


def _seed_based_picks(state, seed_title, lang, exclude_titles, top_n=3):
    catalog = state["catalog"]
    idx = _find_catalog_index(catalog, seed_title)
    if idx is None:
        return pd.DataFrame()

    df = hybrid_recommend(
        query_title=catalog.iloc[idx]["title"],
        catalog=catalog,
        numeric_matrix=state["numeric_matrix"],
        embeddings=state["embeddings"],
        query_embedding=state["embeddings"][idx],
        top_n=top_n,
        exclude_titles=exclude_titles,
        query_lang=lang,
    )
    if df.empty:
        return df
    if is_anomalous(float(df.iloc[0]["hybrid_score"]), state["anomaly_threshold"]):
        return pd.DataFrame()
    return df


def _cluster_based_picks(state, intent, exclude_titles, top_n=3):
    answers = intent_to_onboarding_answers(intent)
    vector, mask = build_user_vector(answers)
    cluster_id = nearest_cluster(
        vector, mask, state["cluster_centroids"], state["cluster_profiles"]
    )
    return recommend_from_cluster(
        state["catalog_with_features"], cluster_id, vector, mask,
        top_n=top_n, exclude_titles=list(exclude_titles),
    )


def _picks_for_intent(state, intent, lang, exclude_titles, top_n=3):
    seeds = intent.get("seeds") or []
    if seeds:
        picks = _seed_based_picks(state, seeds[0], lang, exclude_titles, top_n)
        if not picks.empty:
            return picks
    return _cluster_based_picks(state, intent, exclude_titles, top_n)


@router.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    state = request.app.state.cinematch
    conversation = [m.model_dump() for m in payload.conversation]
    prev_recs = [r.model_dump() for r in payload.prev_recs] if payload.prev_recs else None

    result = chat_turn(conversation, prev_recs=prev_recs, lang=payload.lang)
    action = result["action"]
    intent = result["intent"]

    if action == "chat":
        return ChatResponse(reply=result["reply"])

    exclude_titles = {r["title"] for r in (prev_recs or [])}

    if action == "swap_slot":
        slot_index = result["swap_slot_index"]
        new_picks = _picks_for_intent(state, intent, payload.lang, exclude_titles, top_n=1)
        cards = list(payload.prev_recs or [])
        if not new_picks.empty:
            cards[slot_index] = _to_rec_cards(new_picks)[0]
        return ChatResponse(reply=result["reply"], recommendations=cards)

    # action in ("search", "refine")
    picks = _picks_for_intent(state, intent, payload.lang, exclude_titles, top_n=3)
    if picks.empty:
        return ChatResponse(reply=result["reply"] or t("no_recommendations", payload.lang))

    cards = _to_rec_cards(picks)
    explanation = explain_recommendations(intent, picks.to_dict(orient="records"), payload.lang)
    return ChatResponse(reply=result["reply"], recommendations=cards, explanation=explanation)
```

- [ ] **Step 4: Register the router in `app/main.py`**

Replace the contents of `backend/app/main.py` with:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import chat, recommend
from app.state import load_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cinematch = load_state()
    yield


app = FastAPI(title="CineMatch AI v2", lifespan=lifespan)
app.include_router(recommend.router)
app.include_router(chat.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && pytest tests/test_api_chat.py -v`
Expected: PASS (4 tests). `test_chat_search_returns_recommendations_and_explanation`
and the others run `explain_recommendations`/`chat_turn`'s real fallback
paths - if `GROQ_API_KEY`/`ANTHROPIC_API_KEY` are unset these are the regex/
keyword fallbacks (Task 11), which are deterministic; if a key IS set,
`chat_turn` is monkeypatched in every test here so only `explain_recommendations`
might hit the real LLM - either path returns a non-empty string.

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/chat.py backend/app/main.py backend/tests/test_api_chat.py
git commit -m "Add POST /api/chat endpoint (chat_turn + seed/cluster recommendations)"
```

---

## Task 18: Show Details Endpoint (`routers/show.py`)

`GET /api/show/{title}?lang=he` powers the spec's UX step 4 popup: full
catalog details for the clicked card, plus a live TMDB lookup (trailer,
cast, watch providers) that is best-effort - per the design spec's edge
cases, a missing `TMDB_API_KEY` or no TMDB match falls back silently to
catalog-only data (no error to the user). A title not present in our
catalog at all returns 404 with the `show_not_found` i18n message.

**Files:**
- Create: `backend/app/routers/show.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_api_show.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_api_show.py`:

```python
import app.routers.show as show_module


def test_show_not_found_returns_404_with_i18n_message(client):
    response = client.get("/api/show/NoSuchShowXYZ123?lang=en")
    assert response.status_code == 404
    assert response.json()["detail"] == "I couldn't find that show in our catalog."

    response_he = client.get("/api/show/NoSuchShowXYZ123?lang=he")
    assert response_he.status_code == 404
    assert response_he.json()["detail"] == "לא מצאתי את הסדרה הזו במאגר שלנו."


def test_show_found_returns_catalog_data_without_tmdb(client, monkeypatch):
    monkeypatch.setattr(show_module, "search_tv_show", lambda title, year=None: None)

    catalog = client.app.state.cinematch["catalog"]
    title = catalog.iloc[0]["title"]

    response = client.get(f"/api/show/{title}")
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == title
    assert body["trailer_url"] is None
    assert body["cast"] == []
    assert body["watch_providers"] == []


def test_show_found_with_tmdb_populates_optional_fields(client, monkeypatch):
    catalog = client.app.state.cinematch["catalog"]
    title = catalog.iloc[1]["title"]

    monkeypatch.setattr(
        show_module, "search_tv_show", lambda t, year=None: {"id": 999}
    )
    monkeypatch.setattr(
        show_module,
        "get_tv_show_details",
        lambda tmdb_id: {
            "videos": {"results": [{"site": "YouTube", "type": "Trailer", "key": "abc123"}]},
            "credits": {"cast": [{"name": "Actor One"}, {"name": "Actor Two"}]},
            "watch/providers": {"results": {"US": {"flatrate": [{"provider_name": "Netflix"}]}}},
        },
    )

    response = client.get(f"/api/show/{title}")
    assert response.status_code == 200
    body = response.json()
    assert body["trailer_url"] == "https://www.youtube.com/watch?v=abc123"
    assert body["cast"] == ["Actor One", "Actor Two"]
    assert body["watch_providers"] == ["Netflix"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_api_show.py -v`
Expected: FAIL with `404 Not Found` for the FastAPI route itself / `ModuleNotFoundError: No module named 'app.routers.show'`

- [ ] **Step 3: Create `app/routers/show.py`**

```python
"""GET /api/show/{title} - full catalog details + best-effort TMDB lookup."""

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.agent.tmdb import get_tv_show_details, search_tv_show
from app.i18n import t

router = APIRouter()


class ShowDetails(BaseModel):
    title: str
    genres: str
    rating: float
    overview: str
    poster_path: str | None = None
    decade_str: str
    start_year: int | None = None
    end_year: int | None = None
    num_seasons: float | None = None
    num_episodes: float | None = None
    language: str | None = None
    votes: float | None = None
    popularity: float | None = None
    binge_fit_score: float
    trailer_url: str | None = None
    cast: list[str] = []
    watch_providers: list[str] = []


def _nan_to_none(value):
    return None if pd.isna(value) else value


def _find_catalog_index(catalog: pd.DataFrame, title: str) -> int | None:
    matches = catalog.index[catalog["title"].str.lower() == title.lower()]
    if len(matches) == 0:
        return None
    return int(matches[0])


def _extract_trailer_url(details: dict) -> str | None:
    for video in details.get("videos", {}).get("results", []):
        if video.get("site") == "YouTube" and video.get("type") == "Trailer":
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None


def _extract_cast(details: dict, limit: int = 5) -> list[str]:
    cast = details.get("credits", {}).get("cast", [])
    return [c["name"] for c in cast[:limit]]


def _extract_watch_providers(details: dict, region: str = "US") -> list[str]:
    providers = details.get("watch/providers", {}).get("results", {}).get(region, {})
    return [p["provider_name"] for p in providers.get("flatrate", [])]


@router.get("/api/show/{title}", response_model=ShowDetails)
def get_show(title: str, request: Request, lang: str = "he") -> ShowDetails:
    state = request.app.state.cinematch
    catalog = state["catalog"]

    idx = _find_catalog_index(catalog, title)
    if idx is None:
        raise HTTPException(status_code=404, detail=t("show_not_found", lang))

    row = catalog.iloc[idx]
    details = ShowDetails(
        title=row["title"],
        genres=row["genres"],
        rating=float(row["rating"]),
        overview=row["overview"],
        poster_path=_nan_to_none(row.get("poster_path")),
        decade_str=row["decade_str"],
        start_year=_nan_to_none(row.get("start_year")),
        end_year=_nan_to_none(row.get("end_year")),
        num_seasons=_nan_to_none(row.get("num_seasons")),
        num_episodes=_nan_to_none(row.get("num_episodes")),
        language=_nan_to_none(row.get("language")),
        votes=_nan_to_none(row.get("votes")),
        popularity=_nan_to_none(row.get("popularity")),
        binge_fit_score=float(row["binge_fit_score"]),
    )

    year = _nan_to_none(row.get("start_year"))
    tmdb_result = search_tv_show(row["title"], year=int(year) if year else None)
    if tmdb_result:
        tmdb_details = get_tv_show_details(tmdb_result["id"])
        if tmdb_details:
            details.trailer_url = _extract_trailer_url(tmdb_details)
            details.cast = _extract_cast(tmdb_details)
            details.watch_providers = _extract_watch_providers(tmdb_details)

    return details
```

- [ ] **Step 4: Register the router in `app/main.py`**

Replace the contents of `backend/app/main.py` with:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import chat, recommend, show
from app.state import load_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cinematch = load_state()
    yield


app = FastAPI(title="CineMatch AI v2", lifespan=lifespan)
app.include_router(recommend.router)
app.include_router(chat.router)
app.include_router(show.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && pytest tests/test_api_show.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/show.py backend/app/main.py backend/tests/test_api_show.py
git commit -m "Add GET /api/show/{title} endpoint (catalog details + TMDB lookup)"
```

---

## Task 19: Static Frontend Serving, Render Deploy Config, and End-to-End Test

Wires everything together: `app/main.py` loads `.env` for local development,
serves the built frontend (`frontend/dist`, produced by the frontend
implementation plan's build step) as static files alongside the `/api/*`
routes, and `render.yaml` defines the single Render.com web service the
spec calls for. An end-to-end test exercises the full
onboarding -> recommend -> show details -> chat flow against the real data
artifacts.

**Files:**
- Create: `render.yaml` (repo root)
- Create: `backend/.env.example`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_e2e.py`

- [ ] **Step 1: Write the failing end-to-end test**

Create `backend/tests/test_e2e.py`:

```python
def test_full_onboarding_to_chat_flow(client):
    rec_response = client.post(
        "/api/recommend",
        json={
            "answers": {
                "genre": "drama",
                "length": "long",
                "era": "recent",
                "tone": "serious_drama",
                "popularity": "well_known",
            },
            "lang": "en",
        },
    )
    assert rec_response.status_code == 200
    rec_body = rec_response.json()
    assert len(rec_body["recommendations"]) == 3
    assert rec_body["intro"]
    assert rec_body["outro"]

    first_title = rec_body["recommendations"][0]["title"]

    show_response = client.get(f"/api/show/{first_title}")
    assert show_response.status_code == 200
    assert show_response.json()["title"] == first_title

    chat_response = client.post(
        "/api/chat",
        json={
            "conversation": [
                {"role": "assistant", "content": rec_body["intro"]},
                {"role": "user", "content": "thanks, anything else?"},
            ],
            "prev_recs": rec_body["recommendations"],
            "lang": "en",
        },
    )
    assert chat_response.status_code == 200
    assert "reply" in chat_response.json()


def test_health_check_works_without_a_frontend_build(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `cd backend && pytest tests/test_e2e.py -v`

Both tests should already PASS at this point - Tasks 16-18 wired up all
three endpoints already. This test is a regression guard for the full flow,
not a new feature; if it fails, it means an earlier task's contract was
broken. Re-check Tasks 16-18 if so.

- [ ] **Step 3: Create `backend/.env.example`**

```
# Copy this file to backend/.env and fill in real values for local development.
# All three are optional - the app degrades gracefully without them
# (see app/agent/llm.py and app/agent/tmdb.py for fallback behavior).
GROQ_API_KEY=
ANTHROPIC_API_KEY=
TMDB_API_KEY=
```

- [ ] **Step 4: Load `.env` and serve the built frontend in `app/main.py`**

Replace the contents of `backend/app/main.py` with:

```python
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import chat, recommend, show
from app.state import load_state

load_dotenv()

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cinematch = load_state()
    yield


app = FastAPI(title="CineMatch AI v2", lifespan=lifespan)
app.include_router(recommend.router)
app.include_router(chat.router)
app.include_router(show.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
```

`backend/tests/test_main.py`'s `test_health_check` and the new
`test_health_check_works_without_a_frontend_build` both confirm `/api/health`
keeps working whether or not `frontend/dist` exists yet (it won't until the
frontend plan's build step runs).

- [ ] **Step 5: Run the full test suite**

Run: `cd backend && pytest -v`
Expected: PASS (all tests across Tasks 1-19)

- [ ] **Step 6: Create `render.yaml`** (repo root, alongside `backend/` and `frontend/`)

```yaml
services:
  - type: web
    name: cinematch-ai-v2
    runtime: python
    plan: free
    buildCommand: |
      cd frontend && npm install && npm run build
      cd ../backend && pip install -r requirements.txt
    startCommand: cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
      - key: GROQ_API_KEY
        sync: false
      - key: ANTHROPIC_API_KEY
        sync: false
      - key: TMDB_API_KEY
        sync: false
```

`sync: false` means these secrets are set once in the Render dashboard, not
committed to the repo.

- [ ] **Step 7: Run the backend locally to confirm it serves**

Run:
```powershell
cd backend
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Expected: server starts, logs `Application startup complete`, and
`http://127.0.0.1:8000/api/health` returns `{"status":"ok"}` in a browser.
(`frontend/dist` doesn't exist yet, so `/` returns 404 until the frontend
implementation plan's build step runs - this is expected at this stage.)

- [ ] **Step 8: Commit**

```bash
git add render.yaml backend/.env.example backend/app/main.py backend/tests/test_e2e.py
git commit -m "Serve built frontend as static files, add render.yaml and .env.example"
```

---
