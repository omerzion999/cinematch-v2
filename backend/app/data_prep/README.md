# data_prep (DEV-ONLY)

One-off scripts that rebuild the committed runtime artifacts in `../data/`. They
are **not** imported by the running app, and their heavy dependency
(`sentence-transformers`, which pulls `torch` + `transformers`) is **DEV-ONLY**:
it is installed into the local venv but is deliberately **NOT** in
`backend/requirements.txt`, so it never ships to Render. The app only reads the
committed `catalog.parquet` + `embeddings.npy` at runtime.

## Pipeline (run in order, from the repo root)

1. **`enrich_catalog.py`** — backfills EMPTY `overview` fields in
   `catalog.parquet` from our Assignment 1 source `תרגיל 1/tvs.csv` (matched on
   `name`/`original_name`, highest `vote_count` wins). Fills ~2,300 rows incl.
   every famous show; overview coverage goes 74% -> 95%.
   ```
   python backend/app/data_prep/enrich_catalog.py [path/to/tvs.csv]
   ```

2. **`build_embeddings.py`** — regenerates `embeddings.npy` from the enriched
   overviews with `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, L2-normalized,
   float32). Dev-only install first:
   ```
   backend/.venv/bin/pip install sentence-transformers   # DEV ONLY, not in requirements.txt
   python backend/app/data_prep/build_embeddings.py
   ```

## Invariant (do not break)

`embeddings.npy` is **positional** (catalog row `i` <-> `embeddings[i]`) and
`cluster_labels.parquet` joins on `title`. So the catalog row **count**, row
**order**, and the `title` column must stay byte-identical across an enrichment.
`enrich_catalog.py` asserts this before writing; `build_embeddings.py` keeps the
catalog row order. If you ever reorder/rename catalog rows, regenerate
`cluster_labels.parquet` too (`app/clustering/train_clusters.py`).
