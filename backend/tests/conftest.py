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
