"""
Fuzzy catalog title lookup.

"similar to X" and show-detail requests must survive typos and partial titles
("breakin bad", "the offce", "office"). This resolves a free-text title to a
catalog row index: exact match first, then substring containment, then a
difflib close match. Returns None when nothing is close enough.
"""

import difflib
import re

import pandas as pd

_FUZZY_CUTOFF = 0.8
_MIN_SUBSTRING_LEN = 4


def find_catalog_index(catalog: pd.DataFrame, title: str) -> int | None:
    if not title or not title.strip():
        return None
    lower = catalog["title"].str.lower()
    q = title.lower().strip()

    exact = catalog.index[lower == q]
    if len(exact):
        return int(exact[0])

    if len(q) >= _MIN_SUBSTRING_LEN:
        contains = catalog.index[lower.str.contains(re.escape(q), na=False)]
        if len(contains):
            # Prefer the shortest containing title (usually the canonical one).
            return int(catalog["title"].loc[contains].str.len().idxmin())

    close = difflib.get_close_matches(q, lower.tolist(), n=1, cutoff=_FUZZY_CUTOFF)
    if close:
        idx = catalog.index[lower == close[0]]
        if len(idx):
            return int(idx[0])

    return None
