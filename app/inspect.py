import pandas as pd


def inspect_dataset(processed_csv_path: str) -> None:
    df = pd.read_csv(processed_csv_path)

    print("== BASIC ==")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("Columns:", list(df.columns))

    print("\n== MISSING VALUES ==")
    na = df.isna().sum().sort_values(ascending=False)
    print(na[na > 0] if (na > 0).any() else "No missing values")

    print("\n== TARGET DISTRIBUTION (top5) ==")
    if "top5" in df.columns:
        print(df["top5"].value_counts(normalize=False))
        print(df["top5"].value_counts(normalize=True))

    print("\n== DESCRIBE (NUMERIC) ==")
    num_cols = [c for c in df.columns if c not in ("Language", "Date")]
    print(df[num_cols].describe())

    print("\n== CORRELATION MATRIX (NUMERIC) ==")
    corr = df[num_cols].corr(numeric_only=True)
    print(corr)

    print("\n== TOP LANGUAGES BY MEAN POPULARITY ==")
    if "Language" in df.columns and "Popularity" in df.columns:
        top = df.groupby("Language")["Popularity"].mean().sort_values(ascending=False).head(10)
        print(top)
