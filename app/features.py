import pandas as pd


class FeatureEngineer:
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["year"] = out["Date"].dt.year
        out["month"] = out["Date"].dt.month
        return out

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        g = out.groupby("Language", group_keys=False)

        out["lag_1"] = g["Popularity"].shift(1)
        out["ma_3"] = g["Popularity"].shift(1).rolling(3).mean()
        out["delta_1"] = out["Popularity"] - out["lag_1"]

        # po lagach pojawią się NaN na początku serii – usuwamy
        out = out.dropna(subset=["lag_1", "ma_3", "delta_1"]).reset_index(drop=True)
        return out
