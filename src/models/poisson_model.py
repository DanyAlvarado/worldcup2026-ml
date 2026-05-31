"""Modelo de goles Poisson bivariada con corrección Dixon-Coles.

Modela los goles de cada equipo como variables Poisson cuyas medias (lambdas)
dependen de la diferencia de Elo y la ventaja de localía. La corrección
Dixon-Coles ajusta la probabilidad de marcadores bajos (0-0, 1-0, 0-1, 1-1),
donde la Poisson independiente subestima la dependencia.

Este modelo es clave para la simulación Monte Carlo porque produce *marcadores*,
necesarios para los desempates por diferencia de goles en fase de grupos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.models.base import MatchPredictor

_MAX_GOALS = 10


@dataclass
class PoissonDixonColes(MatchPredictor):
    """Poisson bivariada parametrizada por Elo + localía.

    El log de la media de goles se modela como:
        log(lambda_home) = b0 + b1 * elo_diff_norm + home_adv
        log(lambda_away) = b0 - b1 * elo_diff_norm
    Los parámetros (b0, b1, home_adv, rho) se ajustan por máxima verosimilitud.

    Attributes:
        b0: Intercepto (nivel base de goles).
        b1: Sensibilidad a la diferencia de Elo.
        home_adv: Ventaja de localía en escala log-goles.
        rho: Parámetro de dependencia Dixon-Coles.
    """

    b0: float = 0.1
    b1: float = 0.9
    home_adv: float = 0.25
    rho: float = -0.1
    _elo_scale: float = 400.0

    @staticmethod
    def _dc_adjustment(h: int, a: int, lh: float, la: float, rho: float) -> float:
        """Factor de corrección tau de Dixon-Coles para marcadores bajos."""
        if h == 0 and a == 0:
            return 1.0 - lh * la * rho
        if h == 0 and a == 1:
            return 1.0 + lh * rho
        if h == 1 and a == 0:
            return 1.0 + la * rho
        if h == 1 and a == 1:
            return 1.0 - rho
        return 1.0

    def _lambdas(self, elo_diff: np.ndarray, is_home_real: np.ndarray) -> tuple:
        """Calcula (lambda_home, lambda_away) para cada partido."""
        x = elo_diff / self._elo_scale
        adv = self.home_adv * is_home_real
        lam_home = np.exp(self.b0 + self.b1 * x + adv)
        lam_away = np.exp(self.b0 - self.b1 * x)
        return lam_home, lam_away

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "PoissonDixonColes":
        """Ajusta parámetros por máxima verosimilitud (Nelder-Mead).

        Args:
            X: Debe contener ``elo_diff`` y ``neutral``.
            y: DataFrame con columnas ``home_score`` y ``away_score``.
        """
        elo_diff = X["elo_diff"].to_numpy(dtype=float)
        is_home_real = 1.0 - X["neutral"].to_numpy(dtype=float)
        hs = y["home_score"].to_numpy(dtype=int)
        as_ = y["away_score"].to_numpy(dtype=int)

        def neg_log_lik(params: np.ndarray) -> float:
            b0, b1, home_adv, rho = params
            rho = float(np.clip(rho, -0.2, 0.2))
            x = elo_diff / self._elo_scale
            lam_h = np.exp(b0 + b1 * x + home_adv * is_home_real)
            lam_a = np.exp(b0 - b1 * x)
            ph = poisson.pmf(hs, lam_h)
            pa = poisson.pmf(as_, lam_a)
            tau = np.ones_like(ph)
            low = (hs <= 1) & (as_ <= 1)
            for i in np.where(low)[0]:
                tau[i] = self._dc_adjustment(
                    int(hs[i]), int(as_[i]), lam_h[i], lam_a[i], rho
                )
            likelihood = np.clip(ph * pa * tau, 1e-12, None)
            return -np.sum(np.log(likelihood))

        res = minimize(
            neg_log_lik,
            x0=np.array([self.b0, self.b1, self.home_adv, self.rho]),
            method="Nelder-Mead",
            options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-3},
        )
        self.b0, self.b1, self.home_adv, self.rho = res.x
        self.rho = float(np.clip(self.rho, -0.2, 0.2))
        return self

    def score_matrix(self, elo_diff: float, is_home_real: float) -> np.ndarray:
        """Matriz de probabilidad conjunta de marcadores (MAX_GOALS+1)^2.

        Returns:
            Matriz ``M`` donde ``M[i, j] = P(local marca i, visita marca j)``.
        """
        lam_h, lam_a = self._lambdas(
            np.array([elo_diff]), np.array([is_home_real])
        )
        lh, la = float(lam_h[0]), float(lam_a[0])
        goals = np.arange(_MAX_GOALS + 1)
        ph = poisson.pmf(goals, lh)
        pa = poisson.pmf(goals, la)
        matrix = np.outer(ph, pa)
        for i in (0, 1):
            for j in (0, 1):
                matrix[i, j] *= self._dc_adjustment(i, j, lh, la, self.rho)
        return matrix / matrix.sum()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidades 1X2 derivadas de la matriz de marcadores."""
        out = np.zeros((len(X), 3))
        elo_diff = X["elo_diff"].to_numpy(dtype=float)
        is_home_real = 1.0 - X["neutral"].to_numpy(dtype=float)
        for idx, (ed, ih) in enumerate(zip(elo_diff, is_home_real)):
            m = self.score_matrix(ed, ih)
            home_win = np.tril(m, -1).sum()
            draw = np.trace(m)
            away_win = np.triu(m, 1).sum()
            out[idx] = [home_win, draw, away_win]
        return out
