import json
from pathlib import Path
import pandas as pd

from app.model import TrainConfig, ModelTrainer


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def train_and_evaluate(processed_csv_path: str, split_year: int = 2020) -> Path:
    df = pd.read_csv(processed_csv_path)

    cfg = TrainConfig(split_year=split_year)
    trainer = ModelTrainer(cfg)
    metrics = trainer.train_and_select(df)

    save_metrics(metrics, cfg.metrics_path)
    return cfg.metrics_path
