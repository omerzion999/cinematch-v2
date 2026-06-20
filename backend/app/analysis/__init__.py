"""
Offline data-analysis scripts for the project report (not imported at runtime).

Each module reads app/data/catalog.parquet and writes CSV/JSON artifacts into
docs/analysis/. They use only pandas / numpy / scikit-learn / scipy (already
backend deps), so nothing is added to the runtime requirements.

Run from backend/:
    python -m app.analysis.trends
    python -m app.analysis.model_eval
    python -m app.analysis.export_data
"""

import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "analysis")


def out_path(name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)
