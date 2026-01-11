from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

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
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainConfig:
    artifacts_dir: Path = Path("artifacts")
    model_path: Path = Path("artifacts/model.joblib")
    metrics_path: Path = Path("artifacts/metrics.json")

    # time split boundary (train <= split_year, test > split_year)
    split_year: int = 2020

    # target
    target_col: str = "top5"

    # features
    categorical_cols: Tuple[str, ...] = ("Language",)
    numeric_cols: Tuple[str, ...] = ("Popularity", "lag_1", "ma_3", "delta_1", "year", "month")


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
        train_df = df[df["year"] <= self.cfg.split_year].copy()
        test_df = df[df["year"] > self.cfg.split_year].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(
                "Time split produced empty train or test set. "
                "Check split_year and data range."
            )

        return train_df, test_df

    def _make_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        feature_cols = list(self.cfg.categorical_cols) + list(self.cfg.numeric_cols)
        X = df[feature_cols].copy()
        y = df[self.cfg.target_col].astype(int).copy()
        return X, y

    @staticmethod
    def _json_safe_best_params(best_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert GridSearchCV best_params_ to JSON-serializable dictionary.
        Especially important for entries like {"clf": LogisticRegression(...)}.
        """
        safe = dict(best_params)

        if "clf" in safe and hasattr(safe["clf"], "__class__"):
            safe["clf"] = safe["clf"].__class__.__name__

        # Convert numpy scalars (if any) to Python types
        for k, v in list(safe.items()):
            try:
                # numpy scalar has .item()
                if hasattr(v, "item"):
                    safe[k] = v.item()
            except Exception:
                pass

        return safe

    def train_and_select(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

        train_df, test_df = self._time_split(df)
        X_train, y_train = self._make_xy(train_df)
        X_test, y_test = self._make_xy(test_df)

        pre = self._build_preprocessor()

        # Candidate models
        lr = LogisticRegression(max_iter=2000, random_state=42)
        rf = RandomForestClassifier(random_state=42)

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

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        best_params_safe = self._json_safe_best_params(search.best_params_)
        best_estimator_name = best_model.named_steps["clf"].__class__.__name__

        metrics: Dict[str, Any] = {
            "split_year": int(self.cfg.split_year),
            "best_estimator": best_estimator_name,
            "best_params": best_params_safe,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        }

        joblib.dump(best_model, self.cfg.model_path)
        return metrics


class ModelEvaluator:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

    def load_model(self):
        return joblib.load(self.cfg.model_path)
