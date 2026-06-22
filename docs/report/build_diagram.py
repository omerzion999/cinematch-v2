"""
Generate the data-pipeline flow chart embedded in section 4 of the report.

Dev-only tool (needs matplotlib, not a runtime dependency). Run from repo root:
    python docs/report/build_diagram.py
Output: docs/report/data_flow.png

Step labels are kept in English (technical pipeline terms) so the figure renders
cleanly; the surrounding Hebrew text in the report explains each stage.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "data_flow.png")

DARK = "#1F3864"
ACCENT = "#2E6CB5"
LIGHT = "#EAF1FA"

STEPS = [
    "4 Kaggle sources\n(TMDb, IMDb x2, Disney+)",
    "Merge to\nunified schema",
    "Dedupe\nby title",
    "Normalize genres\n+ handle NaN",
    "Feature engineering\n(z-scores, decade,\nbinge_fit_score)",
    "Overview backfill\nfrom tvs.csv\n(74% -> 95%)",
    "Embeddings\n(multilingual\nMiniLM, 384-d)",
    "catalog.parquet\n+ embeddings.npy\n(11,013 rows)",
]


def build() -> None:
    n = len(STEPS)
    cols = 4
    rows = 2
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")

    bw, bh = 0.86, 0.62
    centers = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Serpentine layout: top row left->right, bottom row right->left.
        if row == 1:
            col = cols - 1 - col
        y = (rows - 1 - row) + 0.5
        x = col + 0.5
        centers.append((x, y, row, col))
        last = i == n - 1
        box = FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.5, edgecolor=DARK,
            facecolor=DARK if last else LIGHT,
        )
        ax.add_patch(box)
        ax.text(
            x, y, STEPS[i], ha="center", va="center", fontsize=8.5,
            color="white" if last else DARK,
            fontweight="bold" if last else "normal",
        )

    # Arrows between consecutive steps (follow the serpentine path).
    for i in range(n - 1):
        x0, y0, r0, _ = centers[i]
        x1, y1, r1, _ = centers[i + 1]
        if r0 == r1:
            start = (x0 + (bw / 2 if x1 > x0 else -bw / 2), y0)
            end = (x1 + (-bw / 2 if x1 > x0 else bw / 2), y1)
        else:
            start = (x0, y0 - bh / 2)
            end = (x1, y1 + bh / 2)
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14,
            linewidth=1.6, color=ACCENT, shrinkA=0, shrinkB=0,
        ))

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
