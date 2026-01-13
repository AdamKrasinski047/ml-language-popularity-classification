from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainConfig:
    """
    Training configuration.

    IMPORTANT:
    - model_path() and metrics_path() are METHODS (computed paths).
    - Do NOT pass model_path=... or metrics_path=... into TrainConfig(...).
      If your evaluate.py does that, it's wrong and will crash.
    """

    # output
    artifacts_dir: Path = Path("artifacts")

    # time split boundary (train <= split_year, test > split_year)
    split_year: int = 2020

    # forecasting setup (used only for metadata + naming)
    top_n: int = 10
    horizon_months: int = 12  # e.g. 6 or 12

    # target column (must exist in processed dataset)
    target_col: str = "top10_h12"

    # features
    categorical_cols: Tuple[str, ...] = ("Language",)
    numeric_cols: Tuple[str, ...] = ("Popularity", "lag_1", "ma_3", "delta_1", "year", "month")

    # model selection
    random_state: int = 42
    n_splits_cv: int = 5  # used in TimeSeriesSplit on TRAIN ONLY

    def model_path(self) -> Path:
        # avoid overwriting when training different horizons/targets
        return self.artifacts_dir / f"model_{self.target_col}.joblib"

    def metrics_path(self) -> Path:
        return self.artifacts_dir / f"metrics_{self.target_col}.json"


class ModelTrainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

    def _build_preprocessor(self) -> ColumnTransformer:
        cat = OneHotEncoder(handle_unknown="ignore")
        num = Pipeline(steps=[("scaler", StandardScaler())])

        return ColumnTransformer(
            transformers=[
                ("cat", cat, list(self.cfg.categorical_cols)),
                ("num", num, list(self.cfg.numeric_cols)),
            ],
            remainder="drop",
        )

    def _time_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if "year" not in df.columns:
            raise ValueError("Missing required column: year (run FeatureEngineer.add_time_features first)")

        train_df = df[df["year"] <= self.cfg.split_year].copy()
        test_df = df[df["year"] > self.cfg.split_year].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(
                "Time split produced empty train or test set. "
                "Check split_year and data range."
            )

        # IMPORTANT:
        # Your dataset is a panel: (Date x Language). TimeSeriesSplit expects a single ordered index.
        # We enforce a stable global ordering by Date then Language.
        if "Date" in train_df.columns:
            train_df = train_df.sort_values(["Date", "Language"]).reset_index(drop=True)
        if "Date" in test_df.columns:
            test_df = test_df.sort_values(["Date", "Language"]).reset_index(drop=True)

        return train_df, test_df

    def _make_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if self.cfg.target_col not in df.columns:
            raise ValueError(f"Missing target column: {self.cfg.target_col}")

        feature_cols = list(self.cfg.categorical_cols) + list(self.cfg.numeric_cols)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[feature_cols].copy()
        y = df[self.cfg.target_col].astype(int).copy()
        return X, y

    @staticmethod
    def _json_safe_best_params(best_params: Dict[str, Any]) -> Dict[str, Any]:
        safe = dict(best_params)

        # GridSearchCV stores estimator objects in params (e.g. "clf": LogisticRegression(...))
        if "clf" in safe and hasattr(safe["clf"], "__class__"):
            safe["clf"] = safe["clf"].__class__.__name__

        # Convert numpy scalars to Python scalars for JSON
        for k, v in list(safe.items()):
            try:
                if hasattr(v, "item"):
                    safe[k] = v.item()
            except Exception:
                pass

        return safe

    @staticmethod
    def _baseline_accuracy(y_true: pd.Series) -> float:
        # "always predict majority class"
        majority = int(y_true.value_counts().idxmax())
        y_pred = [majority] * len(y_true)
        return float(accuracy_score(y_true, y_pred))

    @staticmethod
    def _safe_roc_auc(y_true: pd.Series, y_score: Optional[pd.Series]) -> Optional[float]:
        # ROC-AUC requires both classes present
        if y_score is None:
            return None
        if y_true.nunique() < 2:
            return None
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return None

    def train_and_select(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

        train_df, test_df = self._time_split(df)
        X_train, y_train = self._make_xy(train_df)
        X_test, y_test = self._make_xy(test_df)

        pre = self._build_preprocessor()

        # Candidate models
        lr = LogisticRegression(max_iter=2000, random_state=self.cfg.random_state)
        rf = RandomForestClassifier(random_state=self.cfg.random_state)

        pipe = Pipeline(steps=[("pre", pre), ("clf", lr)])

        param_grid = [
            {
                "clf": [lr],
                "clf__C": [0.1, 1.0, 5.0],
                "clf__class_weight": [None, "balanced"],
            },
            {
                "clf": [rf],
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5],
                "clf__class_weight": [None, "balanced"],
            },
        ]

        # Proper CV for time-ordered data: train-only TimeSeriesSplit
        # Note: With panel data, this is an approximation (global time ordering).
        tscv = TimeSeriesSplit(n_splits=self.cfg.n_splits_cv)

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=tscv,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        # probability for class "1" if supported
        y_proba_1: Optional[pd.Series] = None
        try:
            proba = best_model.predict_proba(X_test)
            y_proba_1 = pd.Series(proba[:, 1], index=y_test.index)
        except Exception:
            y_proba_1 = None

        best_params_safe = self._json_safe_best_params(search.best_params_)
        best_estimator_name = best_model.named_steps["clf"].__class__.__name__

        # extra sanity: class balance on test
        test_class_counts = y_test.value_counts().to_dict()
        test_class_ratio = (y_test.value_counts(normalize=True)).to_dict()

        metrics: Dict[str, Any] = {
            # experiment metadata
            "task": "classification",
            "target": self.cfg.target_col,
            "top_n": int(self.cfg.top_n),
            "horizon_months": int(self.cfg.horizon_months),
            "split_year": int(self.cfg.split_year),
            "categorical_cols": list(self.cfg.categorical_cols),
            "numeric_cols": list(self.cfg.numeric_cols),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_years": [int(train_df["year"].min()), int(train_df["year"].max())],
            "test_years": [int(test_df["year"].min()), int(test_df["year"].max())],
            "test_class_counts": {str(k): int(v) for k, v in test_class_counts.items()},
            "test_class_ratio": {str(k): float(v) for k, v in test_class_ratio.items()},

            # model selection info
            "best_estimator": best_estimator_name,
            "best_params": best_params_safe,

            # baselines + metrics
            "baseline_accuracy_majority": self._baseline_accuracy(y_test),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "roc_auc": self._safe_roc_auc(y_test, y_proba_1),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
        }

        joblib.dump(best_model, self.cfg.model_path())
        return metrics


class ModelEvaluator:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

    def load_model(self):
        return joblib.load(self.cfg.model_path())
