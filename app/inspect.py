import pandas as pd


def _print_target_distribution(df: pd.DataFrame, col: str) -> None:
    print(f"\n== TARGET DISTRIBUTION ({col}) ==")
    vc = df[col].value_counts(dropna=False)
    vcp = df[col].value_counts(normalize=True, dropna=False)
    print(vc)
    print(vcp)

    # baseline = accuracy, gdy zawsze przewidujesz najczęstszą klasę
    majority = int(vc.idxmax())
    baseline = float(vcp.max())
    print(f"Baseline (always predict {majority}): {baseline:.4f}")


def inspect_dataset(processed_csv_path: str) -> None:
    df = pd.read_csv(processed_csv_path)

    print("== BASIC ==")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("Columns:", list(df.columns))

    print("\n== MISSING VALUES ==")
    na = df.isna().sum().sort_values(ascending=False)
    print(na[na > 0] if (na > 0).any() else "No missing values")

    # --- quick dataset sanity stats (useful in report) ---
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        except Exception:
            pass

    if "Date" in df.columns and "Language" in df.columns:
        n_months = df["Date"].nunique()
        n_langs = df["Language"].nunique()
        avg_langs_per_month = df.groupby("Date")["Language"].nunique().mean()
        print("\n== DATASET SANITY ==")
        print(f"Unique months: {n_months}")
        print(f"Unique languages: {n_langs}")
        print(f"Avg languages per month: {avg_langs_per_month:.2f}")

    # --- target distributions ---
    # Prefer the target used in model: top10_h12 (then top10_h6, then top10/top5 if present)
    preferred_order = ["top10_h12", "top10_h6", "top10", "top5"]
    available_targets = [c for c in preferred_order if c in df.columns]

    # Also include any other columns that look like targets (top{n} or top{n}_h{k})
    other_targets = [c for c in df.columns if c.startswith("top") and c not in available_targets]
    # Sort others for stable output
    other_targets = sorted(other_targets)

    if available_targets or other_targets:
        for col in available_targets + other_targets:
            # only binary-looking targets: 0/1
            if set(df[col].dropna().unique()).issubset({0, 1}):
                _print_target_distribution(df, col)
    else:
        print("\n== TARGET DISTRIBUTION ==")
        print("No target-like columns found (expected columns like top10_h12).")

    print("\n== DESCRIBE (NUMERIC) ==")
    # numeric candidates: exclude obvious non-numeric
    num_cols = [c for c in df.columns if c not in ("Language", "Date")]
    # keep only numeric
    num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]
    print(df[num_cols].describe())

    print("\n== CORRELATION MATRIX (NUMERIC) ==")
    corr = df[num_cols].corr(numeric_only=True)
    print(corr)

    print("\n== TOP LANGUAGES BY MEAN POPULARITY ==")
    if "Language" in df.columns and "Popularity" in df.columns:
        top = df.groupby("Language")["Popularity"].mean().sort_values(ascending=False).head(10)
        print(top)
