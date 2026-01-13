from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from app.model import TrainConfig, ModelTrainer


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def _infer_topn_horizon(target_col: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Infer (top_n, horizon_months) from target column like: top10_h6, top5_h12, etc.
    Returns (None, None) if it can't infer.
    """
    m = re.fullmatch(r"top(\d+)_h(\d+)", target_col.strip())
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def train_and_evaluate(
    processed_csv_path: str,
    *,
    split_year: int = 2020,
    target_col: str = "top10_h12",
    top_n: Optional[int] = None,
    horizon_months: Optional[int] = None,
    artifacts_dir: str = "artifacts",
) -> Path:
    df = pd.read_csv(processed_csv_path)

    inferred_top_n, inferred_h = _infer_topn_horizon(target_col)

    if top_n is None:
        top_n = inferred_top_n if inferred_top_n is not None else 10
    if horizon_months is None:
        horizon_months = inferred_h if inferred_h is not None else 12

    cfg = TrainConfig(
        artifacts_dir=Path(artifacts_dir),
        split_year=split_year,
        top_n=int(top_n),
        horizon_months=int(horizon_months),
        target_col=target_col,
    )

    trainer = ModelTrainer(cfg)
    metrics = trainer.train_and_select(df)

    save_metrics(metrics, cfg.metrics_path())
    return cfg.metrics_path()
