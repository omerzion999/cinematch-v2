"""Dev-only data-preparation scripts.

These modules are run ONCE at prep time to (re)build the committed runtime
artifacts in app/data/ (catalog.parquet, embeddings.npy). They are NOT imported
by the running app and their heavy deps (e.g. sentence-transformers) are NOT in
backend/requirements.txt and must never reach Render. See README.md.
"""
