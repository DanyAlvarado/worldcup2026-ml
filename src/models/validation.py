"""Validación temporal y métricas de calibración.

Usa ``TimeSeriesSplit`` sobre el dataset *ya ordenado por fecha*, de modo que
cada fold entrena con el pasado y valida con el futuro inmediato. Nunca se usan
datos futuros para predecir el pasado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

from src.models.base import MatchPredictor
from src.utils.logging import get_logger

logger = get_logger(__name__)


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Brier score multiclase (media de Brier one-vs-rest sobre las 3 clases)."""
    scores = []
    for cls in range(3):
        scores.append(brier_score_loss((y_true == cls).astype(int), proba[:, cls]))
    return float(np.mean(scores))


def time_series_cross_validate(
    model_factory,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "result",
    n_splits: int = 5,
) -> dict[str, float]:
    """Valida un modelo con TimeSeriesSplit y reporta métricas calibradas.

    Args:
        model_factory: Callable sin argumentos que devuelve un ``MatchPredictor``
            nuevo (se reentrena en cada fold).
        df: Dataset ordenado cronológicamente.
        feature_cols: Columnas de entrada.
        target_col: Columna objetivo 1X2.
        n_splits: Número de folds temporales.

    Returns:
        Métricas promedio: ``log_loss``, ``brier``, ``accuracy``.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics: dict[str, list[float]] = {"log_loss": [], "brier": [], "accuracy": []}

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(df), start=1):
        train, val = df.iloc[tr_idx], df.iloc[val_idx]
        model: MatchPredictor = model_factory()

        # El Poisson necesita los marcadores como target; el clasificador, la clase.
        if hasattr(model, "score_matrix"):
            model.fit(train, train[["home_score", "away_score"]])
        else:
            model.fit(train, train[target_col])

        proba = np.clip(model.predict_proba(val), 1e-9, 1.0)
        proba /= proba.sum(axis=1, keepdims=True)
        y_true = val[target_col].to_numpy()

        ll = log_loss(y_true, proba, labels=[0, 1, 2])
        br = multiclass_brier(y_true, proba)
        acc = accuracy_score(y_true, proba.argmax(axis=1))

        metrics["log_loss"].append(ll)
        metrics["brier"].append(br)
        metrics["accuracy"].append(acc)
        logger.info("Fold %d | logloss=%.4f brier=%.4f acc=%.3f",
                    fold, ll, br, acc)

    return {k: float(np.mean(v)) for k, v in metrics.items()}
