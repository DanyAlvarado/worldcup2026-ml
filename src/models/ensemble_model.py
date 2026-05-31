"""Clasificador 1X2 de ensamble (gradient boosting), libre de fuga de datos.

Usa un ``Pipeline`` de scikit-learn (imputación -> escalado -> boosting) para que
toda transformación se ajuste solo con el fold de entrenamiento, evitando data
leakage durante la validación temporal.

Por portabilidad se usa ``HistGradientBoostingClassifier`` de scikit-learn como
implementación por defecto (mismo enfoque de *histogram-based gradient boosting*
que LightGBM, pero sin dependencias nativas como ``libomp``). Si LightGBM está
disponible en el entorno, se utiliza automáticamente como backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED
from src.features.builder import FEATURE_COLUMNS
from src.models.base import MatchPredictor


def _build_classifier():
    """Devuelve el clasificador de boosting (LightGBM si existe, si no sklearn)."""
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:  # noqa: BLE001 - fallback portátil sin libomp
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_depth=6,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=SEED,
        )


def _default_pipeline() -> Pipeline:
    """Pipeline imputación + escalado + boosting multiclase."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", _build_classifier()),
        ]
    )


@dataclass
class EnsembleClassifier(MatchPredictor):
    """Wrapper LightGBM 1X2 conforme a la interfaz ``MatchPredictor``."""

    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    pipeline: Pipeline = field(default_factory=_default_pipeline)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleClassifier":
        """Entrena el pipeline con las columnas de features definidas."""
        self.pipeline.fit(X[self.features], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidades 1X2. Garantiza orden de clases [0, 1, 2]."""
        proba = self.pipeline.predict_proba(X[self.features])
        classes = list(self.pipeline.named_steps["clf"].classes_)
        ordered = np.zeros((len(X), 3))
        for col, cls in enumerate(classes):
            ordered[:, int(cls)] = proba[:, col]
        return ordered
