import pandas as pd

from app.dataset import DatasetBuilder
from app.features import FeatureEngineer


def test_to_long_has_expected_columns():
    df = pd.DataFrame({
        "Date": ["January 2020", "February 2020"],
        "Python": [10, 12],
        "Java": [9, 8],
    })

    builder = DatasetBuilder()
    long_df = builder.to_long(df)

    assert set(["Date", "Language", "Popularity"]).issubset(long_df.columns)
    assert len(long_df) == 4  # 2 months * 2 languages


def test_add_label_top5_is_binary():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01"] * 6),
        "Language": ["A", "B", "C", "D", "E", "F"],
        "Popularity": [60, 50, 40, 30, 20, 10],
    })

    builder = DatasetBuilder()
    labeled = builder.add_label_top5(df, top_n=5)

    assert set(labeled["top5"].unique()).issubset({0, 1})
    assert labeled["top5"].sum() == 5  # top 5 should be 1s


def test_feature_engineering_creates_lags_and_drops_nans():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "Language": ["Python"] * 4,
        "Popularity": [10, 12, 11, 13],
        "top5": [1, 1, 1, 1],
    })

    fe = FeatureEngineer()
    df = fe.add_time_features(df)
    featured = fe.add_lag_features(df)

    assert set(["lag_1", "ma_3", "delta_1"]).issubset(featured.columns)
    # after lag/rolling and dropna, should have fewer rows than input
    assert len(featured) < 4
