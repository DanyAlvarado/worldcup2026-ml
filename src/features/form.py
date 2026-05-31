"""Forma reciente: rachas, lags y promedios móviles por equipo.

Mantiene un historial acotado (deque) por equipo y deriva métricas causales:
puntos por partido recientes, goles a favor/contra promedio, racha de
victorias/derrotas. Todo se lee *antes* de incorporar el partido actual.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from src.config import ROLLING_WINDOW


@dataclass
class _TeamForm:
    """Resultados recientes de un equipo (más reciente al final)."""

    results: deque = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    goals_for: deque = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    goals_against: deque = field(
        default_factory=lambda: deque(maxlen=ROLLING_WINDOW)
    )


@dataclass
class FormTracker:
    """Calcula features de forma reciente de manera causal."""

    window: int = ROLLING_WINDOW
    _state: dict[str, _TeamForm] = field(
        default_factory=lambda: defaultdict(_TeamForm)
    )

    def features(self, team: str) -> dict[str, float]:
        """Snapshot de forma del equipo antes del próximo partido.

        Returns:
            Diccionario con ``form_ppg`` (puntos por partido), ``avg_gf``,
            ``avg_ga`` y ``win_streak`` (rachas; negativo = derrotas).
        """
        f = self._state[team]
        n = len(f.results)
        if n == 0:
            return {"form_ppg": 1.0, "avg_gf": 1.0, "avg_ga": 1.0, "win_streak": 0.0}

        points = sum(3 if r == 1 else 1 if r == 0 else 0 for r in f.results)
        streak = 0
        for r in reversed(f.results):
            if r == 1 and streak >= 0:
                streak += 1
            elif r == -1 and streak <= 0:
                streak -= 1
            else:
                break
        return {
            "form_ppg": points / n,
            "avg_gf": sum(f.goals_for) / n,
            "avg_ga": sum(f.goals_against) / n,
            "win_streak": float(streak),
        }

    def update(
        self, home: str, away: str, home_score: int, away_score: int
    ) -> None:
        """Incorpora el resultado al historial de ambos equipos.

        Codificación de ``results``: 1 victoria, 0 empate, -1 derrota.
        """
        if home_score > away_score:
            home_res, away_res = 1, -1
        elif home_score < away_score:
            home_res, away_res = -1, 1
        else:
            home_res = away_res = 0

        h, a = self._state[home], self._state[away]
        h.results.append(home_res)
        h.goals_for.append(home_score)
        h.goals_against.append(away_score)
        a.results.append(away_res)
        a.goals_for.append(away_score)
        a.goals_against.append(home_score)
