import pandas as pd


class DatasetBuilder:
    def load_raw(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "Date" not in df.columns:
            raise ValueError("Missing required column: Date")
        return df

    def to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"], format="%B %Y", errors="coerce")
        out = out.dropna(subset=["Date"])

        long_df = out.melt(id_vars=["Date"], var_name="Language", value_name="Popularity")
        long_df["Popularity"] = pd.to_numeric(long_df["Popularity"], errors="coerce").fillna(0.0)

        return long_df.sort_values(["Language", "Date"]).reset_index(drop=True)

    def add_label_top5(self, df_long: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        out = df_long.copy()
        out["rank_in_month"] = out.groupby("Date")["Popularity"].rank(method="dense", ascending=False)
        out["top5"] = (out["rank_in_month"] <= top_n).astype(int)
        return out.drop(columns=["rank_in_month"])
