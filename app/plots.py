from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def make_plots(processed_csv_path: str, model_path: str = "artifacts/model.joblib") -> Path:
    out_dir = Path("artifacts") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # 1) Trend popularności TOP10 języków wg średniej popularności
    top_langs = (
        df.groupby("Language")["Popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .index
        .tolist()
    )

    pivot = df[df["Language"].isin(top_langs)].pivot_table(
        index="Date", columns="Language", values="Popularity", aggfunc="mean"
    ).sort_index()

    plt.figure()
    pivot.plot()
    plt.title("Popularity over time (Top 10 languages by mean popularity)")
    plt.xlabel("Date")
    plt.ylabel("Popularity")
    plt.tight_layout()
    plt.savefig(out_dir / "popularity_top10_timeseries.png", dpi=150)
    plt.close()

    # 2) Heatmapa: TOP10 membership (miesiąc x język) dla top_langs
    membership = df[df["Language"].isin(top_langs)].pivot_table(
        index="Date", columns="Language", values="top10", aggfunc="max"
    ).sort_index()

    plt.figure()
    plt.imshow(membership.T, aspect="auto", interpolation="nearest")
    plt.title("TOP10 membership heatmap (Date x Language)")
    plt.xlabel("Date index (monthly)")
    plt.ylabel("Language")
    plt.yticks(range(len(membership.columns)), membership.columns)
    plt.tight_layout()
    plt.savefig(out_dir / "top10_membership_heatmap.png", dpi=150)
    plt.close()

    # 3) Confusion matrix (jeśli model istnieje)
    model_file = Path(model_path)
    if model_file.exists():
        model = joblib.load(model_file)

        # cechy takie jak w modelu
        feature_cols = ["Language", "Popularity", "lag_1", "ma_3", "delta_1", "year", "month"]
        target_col = "top10_h12" if "top10_h12" in df.columns else None

        if target_col:
            X = df[feature_cols].copy()
            y = df[target_col].astype(int).copy()
            y_pred = model.predict(X)

            plt.figure()
            ConfusionMatrixDisplay.from_predictions(y, y_pred)
            plt.title("Confusion matrix (full dataset, for visualization)")
            plt.tight_layout()
            plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
            plt.close()

    return out_dir
