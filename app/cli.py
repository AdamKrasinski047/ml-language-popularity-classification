import argparse

from app.inspect import inspect_dataset
from app.config import DATA_URL, RAW_FILE, PROCESSED_DIR, PROCESSED_FILE
from app.fetcher import DataFetcher
from app.dataset import DatasetBuilder
from app.features import FeatureEngineer
from app.evaluate import train_and_evaluate
from app.plots import make_plots


TOP_N_DEFAULT = 10
HORIZON_CHOICES = (6, 12)


def cmd_fetch() -> None:
    path = DataFetcher(DATA_URL, RAW_FILE).fetch()
    print(f"OK: downloaded dataset to: {path}")


def cmd_prepare(top_n: int = TOP_N_DEFAULT) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    builder = DatasetBuilder()
    fe = FeatureEngineer()

    raw = builder.load_raw(str(RAW_FILE))
    long_df = builder.to_long(raw)

    labeled = builder.add_label_topN(long_df, top_n=top_n)
    labeled = builder.add_future_labels_topN(
        labeled,
        top_n=top_n,
        horizons=HORIZON_CHOICES,
    )

    featured = fe.add_time_features(labeled)
    featured = fe.add_lag_features(featured)

    featured.to_csv(PROCESSED_FILE, index=False)
    print(f"OK: prepared dataset saved to: {PROCESSED_FILE}")
    print(f"Rows: {len(featured)} | Columns: {len(featured.columns)}")


def _target_from_horizon(top_n: int, horizon: int) -> str:
    return f"top{top_n}_h{horizon}"


def cmd_train(split_year: int, top_n: int, horizon: int) -> None:
    target_col = _target_from_horizon(top_n, horizon)

    path = train_and_evaluate(
        str(PROCESSED_FILE),
        split_year=split_year,
        target_col=target_col,
    )
    print(f"OK: trained model and saved metrics to: {path}")


def cmd_evaluate(split_year: int, top_n: int, horizon: int) -> None:
    # For rubric clarity; current implementation evaluates during training
    target_col = _target_from_horizon(top_n, horizon)

    path = train_and_evaluate(
        str(PROCESSED_FILE),
        split_year=split_year,
        target_col=target_col,
    )
    print(f"OK: evaluation metrics saved to: {path}")


def cmd_inspect() -> None:
    inspect_dataset(str(PROCESSED_FILE))


def cmd_plot() -> None:
    out_dir = make_plots(str(PROCESSED_FILE))
    print(f"OK: plots saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML Language Popularity Classification (project CLI)"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # fetch
    sub.add_parser("fetch", help="Download raw dataset")

    # prepare
    p_prepare = sub.add_parser("prepare", help="Prepare processed dataset")
    p_prepare.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)

    # inspect
    sub.add_parser("inspect", help="Inspect processed dataset stats")

    # train
    p_train = sub.add_parser("train", help="Train and evaluate model")
    p_train.add_argument("--split-year", type=int, default=2020)
    p_train.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    p_train.add_argument("--horizon", type=int, choices=HORIZON_CHOICES, default=12)

    # evaluate (alias)
    p_eval = sub.add_parser("evaluate", help="Evaluate model (runs training+evaluation)")
    p_eval.add_argument("--split-year", type=int, default=2020)
    p_eval.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    p_eval.add_argument("--horizon", type=int, choices=HORIZON_CHOICES, default=12)

    # plot
    sub.add_parser("plot", help="Generate plots into artifacts/plots")

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch()
    elif args.command == "prepare":
        cmd_prepare(top_n=args.top_n)
    elif args.command == "inspect":
        cmd_inspect()
    elif args.command == "train":
        cmd_train(split_year=args.split_year, top_n=args.top_n, horizon=args.horizon)
    elif args.command == "evaluate":
        cmd_evaluate(split_year=args.split_year, top_n=args.top_n, horizon=args.horizon)
    elif args.command == "plot":
        cmd_plot()


if __name__ == "__main__":
    main()
