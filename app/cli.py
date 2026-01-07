import argparse

from app.config import DATA_URL, RAW_FILE
from app.fetcher import DataFetcher


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML Language Popularity Classification (project CLI)"
    )
    parser.add_argument(
        "command",
        choices=["fetch"],
        help="Action to run",
    )
    args = parser.parse_args()

    if args.command == "fetch":
        path = DataFetcher(DATA_URL, RAW_FILE).fetch()
        print(f"OK: downloaded dataset to: {path}")


if __name__ == "__main__":
    main()
