import argparse

from app.config import DATA_URL, RAW_FILE, PROCESSED_DIR, PROCESSED_FILE
from app.fetcher import DataFetcher
from app.dataset import DatasetBuilder
from app.features import FeatureEngineer


def cmd_fetch() -> None:
    path = DataFetcher(DATA_URL, RAW_FILE).fetch()
    print(f"OK: downloaded dataset to: {path}")


def cmd_prepare() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    builder = DatasetBuilder()
    fe = FeatureEngineer()

    raw = builder.load_raw(str(RAW_FILE))
    long_df = builder.to_long(raw)
    labeled = builder.add_label_top5(long_df, top_n=5)

    featured = fe.add_time_features(labeled)
    featured = fe.add_lag_features(featured)

    featured.to_csv(PROCESSED_FILE, index=False)
    print(f"OK: prepared dataset saved to: {PROCESSED_FILE}")
    print(f"Rows: {len(featured)} | Columns: {len(featured.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML Language Popularity Classification (project CLI)"
    )
    parser.add_argument(
        "command",
        choices=["fetch", "prepare"],
        help="Action to run",
    )
    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch()
    elif args.command == "prepare":
        cmd_prepare()


if __name__ == "__main__":
    main()
