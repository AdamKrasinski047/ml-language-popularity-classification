from pathlib import Path

# Public URL to the dataset stored in this GitHub repo (raw file)
DATA_URL = (
    "https://raw.githubusercontent.com/AdamKrasinski047/"
    "ml-language-popularity-classification/main/"
    "data_source/popularity_languages_2004_2024.csv"
)

# Local cache paths (ignored by git)
RAW_DIR = Path("data/raw")
RAW_FILE = RAW_DIR / "popularity_languages_2004_2024.csv"
PROCESSED_DIR = Path("data/processed")
PROCESSED_FILE = PROCESSED_DIR / "dataset.csv"
