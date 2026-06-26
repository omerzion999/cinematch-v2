"""
Export the post-manipulation data deliverable (the rubric's "data after your
manipulations") as a ZIP. The ZIP bundles the final catalog CSV, a Hebrew
README / data dictionary, and the analysis result CSVs, so the deliverable shows
both the cleaned data and the insights derived from it. CSV avoids adding openpyxl;
the rubric allows any format.

Run: python -m app.analysis.export_data
"""

import os
import zipfile

import pandas as pd

from app.analysis import DATA_DIR, OUT_DIR, out_path

# (column, Hebrew description) for the data dictionary.
_COLUMNS = [
    ("title", "שם הסדרה (מפתח ייחודי)"),
    ("language", "שפת המקור"),
    ("start_year", "שנת תחילת השידור"),
    ("end_year", "שנת סיום השידור (ריק אם עדיין משודרת)"),
    ("genres", "ז'אנרים (מופרדים בפסיק)"),
    ("rating", "דירוג IMDb (0-10)"),
    ("votes", "מספר הצבעות"),
    ("popularity", "מדד פופולריות"),
    ("overview", "תקציר העלילה (הושלם ממקור tvs לסדרות שחסר בהן)"),
    ("poster_path", "נתיב פוסטר ב-TMDB"),
    ("num_episodes", "מספר פרקים"),
    ("num_seasons", "מספר עונות"),
    ("source_dataset", "מאגר המקור שממנו נלקחה הרשומה"),
    ("decade", "עשור (מספרי)"),
    ("decade_str", "עשור (מחרוזת, למשל 2010s)"),
    ("genre_set_str", "קבוצת הז'אנרים (מופרדת ב-|) למדד Jaccard"),
    ("rating_z", "ציון תקן (z-score) של הדירוג"),
    ("votes_z", "ציון תקן של מספר ההצבעות"),
    ("start_year_z", "ציון תקן של שנת ההתחלה"),
    ("popularity_z", "ציון תקן של הפופולריות"),
    ("rating_bucket", "חלוקת הדירוג לקטגוריות"),
    ("binge_fit_score", "ניקוד מהונדס: (דירוג/10)*4 + הצבעות + עדכניות + שידור פעיל"),
]

_ANALYSIS_CSVS = [
    "genre_share_by_decade.csv",
    "rating_by_decade.csv",
    "genre_trend_2000s_to_2020s.csv",
    "genre_entropy.csv",
    "clustering_silhouette.csv",
    "clustering_profiles.csv",
    "similarity_correlations.csv",
]


def _readme(n_rows: int, n_cols: int) -> str:
    lines = [
        "CineMatch v2 - הנתונים לאחר המניפולציות",
        "מגישים: תומר בלונד (322211103), עומר ציון (322757469)",
        "",
        f"catalog_post_manipulation.csv - הקטלוג הסופי: {n_rows:,} סדרות, {n_cols} עמודות.",
        "",
        "מקורות (Kaggle): TMDb, IMDb top-5000, Disney+, ו-IMDb נוסף.",
        "מניפולציות שבוצעו:",
        "  1. מיזוג ארבעת המקורות לסכמה אחידה.",
        "  2. הסרת כפילויות לפי שם הסדרה.",
        "  3. נרמול מחרוזות הז'אנר וטיפול בערכים חסרים.",
        "  4. הנדסת מאפיינים: z-scores, decade, rating_bucket, binge_fit_score.",
        "  5. השלמת תקצירים חסרים ממקור tvs (כיסוי עלה מ-74% ל-95%).",
        "  6. חישוב שיבוצים סמנטיים (384 ממדים) על התקצירים (קובץ נפרד embeddings.npy).",
        "",
        "מילון נתונים (data dictionary):",
    ]
    for col, desc in _COLUMNS:
        lines.append(f"  {col}: {desc}")
    lines += [
        "",
        "קבצי ניתוח נלווים (תוצרים):",
        "  genre_share_by_decade.csv - התפלגות ז'אנרים לפי עשור.",
        "  rating_by_decade.csv - דירוג ו-binge_fit לפי עשור.",
        "  genre_trend_2000s_to_2020s.csv - מגמות ז'אנר.",
        "  genre_entropy.csv - אנטרופיית שאנון של תמהיל הז'אנרים.",
        "  clustering_silhouette.csv - ציוני silhouette של K-Means לפי k.",
        "  clustering_profiles.csv - פרופילי האשכולות.",
        "  similarity_correlations.csv - מתאמי Spearman בין מדדי הדמיון.",
    ]
    return "\n".join(lines)


def main():
    catalog = pd.read_parquet(os.path.join(DATA_DIR, "catalog.parquet"))
    csv_path = out_path("catalog_post_manipulation.csv")
    catalog.to_csv(csv_path, index=False)

    readme_path = out_path("README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(_readme(len(catalog), len(catalog.columns)))

    zip_path = os.path.join(OUT_DIR, "catalog_post_manipulation.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(readme_path, arcname="README.txt")
        zf.write(csv_path, arcname="catalog_post_manipulation.csv")
        for name in _ANALYSIS_CSVS:
            path = os.path.join(OUT_DIR, name)
            if os.path.exists(path):
                zf.write(path, arcname=f"analysis/{name}")

    print(f"Exported {len(catalog)} rows, {len(catalog.columns)} columns")
    print(f"  ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        print("  contents:", zf.namelist())


if __name__ == "__main__":
    main()
