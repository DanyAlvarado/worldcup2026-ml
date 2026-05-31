"""Pipeline de entrenamiento end-to-end.

Orquesta: carga -> features (causales) -> validación temporal -> entrenamiento
final sobre todo el histórico -> persistencia de artefactos. Ejecutable como
script: ``python -m src.models.train``.
"""
from __future__ import annotations

import joblib

from src.config import (
    ENSEMBLE_MODEL_PATH,
    POISSON_MODEL_PATH,
    PROCESSED_FEATURES,
    TEAM_STATE_PATH,
)
from src.data.loader import load_results
from src.features.builder import FEATURE_COLUMNS, FeatureBuilder
from src.models.ensemble_model import EnsembleClassifier
from src.models.poisson_model import PoissonDixonColes
from src.models.validation import time_series_cross_validate
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Restringir a fútbol "moderno" mejora la relevancia (menos ruido del s. XIX/XX).
MODERN_CUTOFF = "1990-01-01"


def main() -> None:
    """Entrena y persiste ambos modelos + el estado de equipos."""
    logger.info("=== 1. Carga de datos ===")
    matches = load_results()
    matches = matches[matches["date"] >= MODERN_CUTOFF].reset_index(drop=True)

    logger.info("=== 2. Feature engineering (causal) ===")
    builder = FeatureBuilder()
    df = builder.build(matches)
    df = df.dropna().reset_index(drop=True)
    df.to_parquet(PROCESSED_FEATURES, index=False)

    logger.info("=== 3. Validación temporal: LightGBM 1X2 ===")
    lgbm_metrics = time_series_cross_validate(
        EnsembleClassifier, df, FEATURE_COLUMNS
    )
    logger.info("LightGBM CV: %s", lgbm_metrics)

    logger.info("=== 4. Validación temporal: Poisson Dixon-Coles ===")
    poisson_metrics = time_series_cross_validate(
        PoissonDixonColes, df, ["elo_diff", "neutral"]
    )
    logger.info("Poisson CV: %s", poisson_metrics)

    logger.info("=== 5. Entrenamiento final sobre todo el histórico ===")
    ensemble = EnsembleClassifier().fit(df, df["result"])
    poisson = PoissonDixonColes().fit(df, df[["home_score", "away_score"]])

    joblib.dump(ensemble, ENSEMBLE_MODEL_PATH)
    joblib.dump(poisson, POISSON_MODEL_PATH)
    joblib.dump(builder.team_state, TEAM_STATE_PATH)
    logger.info("Artefactos guardados en %s", ENSEMBLE_MODEL_PATH.parent)

    logger.info(
        "RESUMEN | LightGBM logloss=%.4f acc=%.3f | Poisson logloss=%.4f",
        lgbm_metrics["log_loss"], lgbm_metrics["accuracy"],
        poisson_metrics["log_loss"],
    )


if __name__ == "__main__":
    main()
