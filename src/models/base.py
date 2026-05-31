"""Interfaz común para los modelos predictivos (principio DIP)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class MatchPredictor(ABC):
    """Contrato que todo modelo de predicción de partidos debe cumplir."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y) -> "MatchPredictor":
        """Entrena el modelo. Devuelve self para encadenar."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidades 1X2. Shape (n, 3): [P(local), P(empate), P(visita)]."""
