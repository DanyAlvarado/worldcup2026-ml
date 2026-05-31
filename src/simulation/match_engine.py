"""Motor de partido para la simulación: convierte estado de equipos en marcadores.

Construye el vector de features de un enfrentamiento 2026 a partir del
``TeamState`` entrenado y usa el modelo Poisson para muestrear marcadores
concretos (necesarios para desempates) y el LightGBM para probabilidades 1X2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.builder import FEATURE_COLUMNS, TeamState
from src.features.environment import is_host
from src.models.ensemble_model import EnsembleClassifier
from src.models.poisson_model import PoissonDixonColes


@dataclass
class MatchEngine:
    """Genera marcadores y probabilidades para enfrentamientos del Mundial 2026."""

    team_state: TeamState
    poisson: PoissonDixonColes
    ensemble: EnsembleClassifier
    rng: np.random.Generator

    def _feature_row(self, home: str, away: str, neutral: bool) -> pd.DataFrame:
        """Construye la fila de features para un enfrentamiento."""
        elo_h, elo_a = self.team_state.elo.pre_match_ratings(home, away, neutral)
        form_h = self.team_state.form.features(home)
        form_a = self.team_state.form.features(away)
        pm_h, pw_h = self.team_state.pedigree.get_pedigree(home)
        pm_a, pw_a = self.team_state.pedigree.get_pedigree(away)

        data = {
            "elo_home": elo_h, "elo_away": elo_a, "elo_diff": elo_h - elo_a,
            "form_ppg_home": form_h["form_ppg"], "form_ppg_away": form_a["form_ppg"],
            "avg_gf_home": form_h["avg_gf"], "avg_ga_home": form_h["avg_ga"],
            "avg_gf_away": form_a["avg_gf"], "avg_ga_away": form_a["avg_ga"],
            "win_streak_home": form_h["win_streak"],
            "win_streak_away": form_a["win_streak"],
            "wc_matches_home": float(pm_h), "wc_wins_home": float(pw_h),
            "wc_matches_away": float(pm_a), "wc_wins_away": float(pw_a),
            "is_home_host": float(is_host(home) and not neutral),
            "neutral": float(neutral),
        }
        return pd.DataFrame([data], columns=FEATURE_COLUMNS)

    def sample_score(
        self, home: str, away: str, neutral: bool = True
    ) -> tuple[int, int]:
        """Muestrea un marcador concreto desde la matriz Poisson-Dixon-Coles."""
        elo_h, elo_a = self.team_state.elo.pre_match_ratings(home, away, neutral)
        matrix = self.poisson.score_matrix(elo_h - elo_a, 0.0 if neutral else 1.0)
        flat = matrix.ravel()
        idx = self.rng.choice(len(flat), p=flat)
        n_cols = matrix.shape[1]
        return int(idx // n_cols), int(idx % n_cols)

    def win_probabilities(
        self, home: str, away: str, neutral: bool = True
    ) -> dict[str, float]:
        """Probabilidades 1X2 (blend Poisson + LightGBM) para la UI."""
        row = self._feature_row(home, away, neutral)
        p_lgbm = self.ensemble.predict_proba(row)[0]
        p_pois = self.poisson.predict_proba(row)[0]
        blend = 0.5 * p_lgbm + 0.5 * p_pois
        blend = blend / blend.sum()
        return {"home_win": float(blend[0]), "draw": float(blend[1]),
                "away_win": float(blend[2])}
