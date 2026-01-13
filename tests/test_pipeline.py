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


def test_add_label_topN_is_binary_and_counts_top5_correctly():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01"] * 6),
        "Language": ["A", "B", "C", "D", "E", "F"],
        "Popularity": [60, 50, 40, 30, 20, 10],
    })

    builder = DatasetBuilder()
    labeled = builder.add_label_topN(df, top_n=5)

    # kolumna nazywa się top{N}
    assert "top5" in labeled.columns
    assert set(labeled["top5"].unique()).issubset({0, 1})

    # przy 6 językach i top_n=5 dokładnie 5 powinno mieć 1
    assert int(labeled["top5"].sum()) == 5


def test_add_future_labels_topN_creates_horizons_and_drops_tail_rows():
    # 13 miesięcy dla jednego języka -> dla h=12 po dropna zostanie 1 wiersz
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=13, freq="MS"),
        "Language": ["Python"] * 13,
        "Popularity": list(range(13)),
    })

    builder = DatasetBuilder()
    labeled = builder.add_label_topN(df, top_n=1)  # top1 zawsze 1, bo tylko 1 język w danym miesiącu
    future = builder.add_future_labels_topN(labeled, top_n=1, horizons=(12,))

    assert "top1_h12" in future.columns
    assert len(future) == 1  # 13 - 12 = 1


def test_feature_engineering_creates_lags_and_drops_nans():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "Language": ["Python"] * 4,
        "Popularity": [10, 12, 11, 13],
        "top10_h12": [0, 0, 0, 0],  # dowolna kolumna targetu, fe jej nie używa
    })

    fe = FeatureEngineer()
    df = fe.add_time_features(df)
    featured = fe.add_lag_features(df)

    assert set(["lag_1", "ma_3", "delta_1"]).issubset(featured.columns)

    # dla 4 miesięcy:
    # lag_1 jest od 2. miesiąca
    # ma_3 = rolling(3) na (shifted) -> dopiero w 4. miesiącu mamy 3 poprzednie wartości
    # po dropna zostanie dokładnie 1 wiersz
    assert len(featured) == 1
