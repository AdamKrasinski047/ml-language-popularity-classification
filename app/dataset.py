import pandas as pd


class DatasetBuilder:
    def load_raw(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "Date" not in df.columns:
            raise ValueError("Missing required column: Date")
        return df

    def to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Surowe dane mają zwykle format "January 2004" itd.
        out["Date"] = pd.to_datetime(out["Date"], format="%B %Y", errors="coerce")
        out = out.dropna(subset=["Date"])

        long_df = out.melt(id_vars=["Date"], var_name="Language", value_name="Popularity")
        long_df["Popularity"] = pd.to_numeric(long_df["Popularity"], errors="coerce").fillna(0.0)

        return long_df.sort_values(["Language", "Date"]).reset_index(drop=True)

    def add_label_topN(self, df_long: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Create label for being in TOP-N in a given month (Date).
        Adds column: top{N}, e.g. top10.
        """
        if "Date" not in df_long.columns or "Popularity" not in df_long.columns or "Language" not in df_long.columns:
            raise ValueError("df_long must contain columns: Date, Language, Popularity")

        out = df_long.copy()
        # ranking w obrębie miesiąca (Date): im większa Popularity, tym lepsza pozycja
        out["rank_in_month"] = out.groupby("Date")["Popularity"].rank(method="dense", ascending=False)
        out[f"top{top_n}"] = (out["rank_in_month"] <= top_n).astype(int)

        return out.drop(columns=["rank_in_month"])

    def add_future_labels_topN(
        self,
        df_labeled: pd.DataFrame,
        top_n: int = 10,
        horizons: tuple[int, ...] = (6, 12),
    ) -> pd.DataFrame:
        """
        Create future labels: whether a language will be in TOP-N after H months.
        For each horizon H creates: top{N}_h{H} (e.g. top10_h6, top10_h12)

        Uses shift(-H) within each Language time series (sorted by Date).
        """
        label_col = f"top{top_n}"
        if label_col not in df_labeled.columns:
            raise ValueError(f"Missing required label column: {label_col}. Call add_label_topN first.")

        out = df_labeled.copy()
        out = out.sort_values(["Language", "Date"]).reset_index(drop=True)

        for h in horizons:
            future_col = f"{label_col}_h{h}"
            out[future_col] = out.groupby("Language")[label_col].shift(-h)

        # Usuń wiersze, dla których nie ma już przyszłych etykiet (końcówki szeregu)
        drop_cols = [f"{label_col}_h{h}" for h in horizons]
        out = out.dropna(subset=drop_cols).copy()

        # Rzutowanie na int (po dropna mamy floaty, bo NaN)
        for col in drop_cols:
            out[col] = out[col].astype(int)

        return out
