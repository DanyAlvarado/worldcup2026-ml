"""Sistema de puntuación Elo para selecciones nacionales.

Implementa el método World Football Elo Ratings: K ajustado por importancia del
torneo y escalado por diferencia de goles, con ventaja de localía. El estado
(rating por equipo) se actualiza partido a partido de forma estrictamente
causal, por lo que es seguro para evitar data leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import (
    DEFAULT_TOURNAMENT_WEIGHT,
    EloConfig,
    TOURNAMENT_WEIGHT,
)


@dataclass
class EloRatingSystem:
    """Mantiene y actualiza ratings Elo de selecciones.

    El sistema es *online*: se recorre el histórico en orden cronológico y, para
    cada partido, primero se leen los ratings vigentes (features pre-partido) y
    luego se actualizan con el resultado. Esto garantiza que las features de un
    partido nunca contengan información de su propio resultado ni de partidos
    futuros.

    Attributes:
        config: Hiperparámetros del sistema Elo.
        ratings: Mapa equipo -> rating actual.
    """

    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        """Rating actual del equipo (base si nunca ha jugado)."""
        return self.ratings.get(team, self.config.base_rating)

    @staticmethod
    def _expected_score(rating_a: float, rating_b: float) -> float:
        """Probabilidad esperada de que A gane (escala logística de 400)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _k_factor(self, tournament: str, goal_diff: int) -> float:
        """K efectivo: peso del torneo escalado por margen de goles."""
        base_k = TOURNAMENT_WEIGHT.get(tournament, DEFAULT_TOURNAMENT_WEIGHT)
        if not self.config.goal_diff_scaling:
            return base_k
        margin = abs(goal_diff)
        if margin <= 1:
            multiplier = 1.0
        elif margin == 2:
            multiplier = 1.5
        else:  # margen de 3+ goles
            multiplier = (11 + margin) / 8.0
        return base_k * multiplier

    def pre_match_ratings(
        self, home: str, away: str, neutral: bool
    ) -> tuple[float, float]:
        """Ratings efectivos antes del partido, incluyendo ventaja de localía.

        Args:
            home: Equipo local.
            away: Equipo visitante.
            neutral: Si el partido es en cancha neutral (sin ventaja de localía).

        Returns:
            Tupla (rating_local_efectivo, rating_visitante).
        """
        home_r = self.get(home)
        away_r = self.get(away)
        if not neutral:
            home_r += self.config.home_advantage
        return home_r, away_r

    def update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        tournament: str,
        neutral: bool,
    ) -> tuple[float, float]:
        """Actualiza los ratings tras un partido y devuelve los pre-partido.

        Returns:
            Los ratings (local, visitante) *antes* de la actualización, que son
            las features causales del partido.
        """
        home_eff, away_eff = self.pre_match_ratings(home, away, neutral)
        pre_home, pre_away = self.get(home), self.get(away)

        expected_home = self._expected_score(home_eff, away_eff)
        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        k = self._k_factor(tournament, home_score - away_score)
        delta = k * (actual_home - expected_home)

        self.ratings[home] = pre_home + delta
        self.ratings[away] = pre_away - delta
        return pre_home, pre_away
