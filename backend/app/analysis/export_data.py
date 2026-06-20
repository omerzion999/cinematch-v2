"""
Export the post-manipulation catalog (the data deliverable) as CSV and zip it.
CSV is used instead of XLSX to avoid adding openpyxl; the rubric allows any
format (zip is fine).

Run: python -m app.analysis.export_data
"""

import os
import zipfile

import pandas as pd

from app.analysis import DATA_DIR, OUT_DIR, out_path


def main():
    catalog = pd.read_parquet(os.path.join(DATA_DIR, "catalog.parquet"))
    csv_path = out_path("catalog_post_manipulation.csv")
    catalog.to_csv(csv_path, index=False)

    zip_path = os.path.join(OUT_DIR, "catalog_post_manipulation.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="catalog_post_manipulation.csv")

    print(f"Exported {len(catalog)} rows, {len(catalog.columns)} columns")
    print(f"  CSV: {csv_path}")
    print(f"  ZIP: {zip_path}")
    print(f"columns: {list(catalog.columns)}")


if __name__ == "__main__":
    main()
