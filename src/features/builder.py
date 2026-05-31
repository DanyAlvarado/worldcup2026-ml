"""Orquestador de feature engineering, libre de data leakage.

Recorre el histórico de partidos en orden cronológico estricto. Para cada
partido:
    1. Lee las features pre-partido (Elo, forma, pedigrí) -> snapshot causal.
    2. Registra esas features junto con el target (resultado real).
    3. Actualiza el estado de todos los trackers con el resultado.

Como ningún tracker mira hacia el futuro, el dataset resultante es seguro para
``TimeSeriesSplit``. El estado final de los trackers se expone vía
``team_state`` para usarlo en la predicción del Mundial 2026.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import HOST_NATIONS
from src.features.elo import EloRatingSystem
from src.features.form import FormTracker
from src.features.pedigree import PedigreeTracker
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TeamState:
    """Estado final de todos los trackers tras procesar el histórico.

    Se persiste para alimentar la simulación del Mundial 2026 con los ratings y
    formas más recientes de cada selección.
    """

    elo: EloRatingSystem
    form: FormTracker
    pedigree: PedigreeTracker


@dataclass
class FeatureBuilder:
    """Construye el dataset de entrenamiento de forma causal."""

    elo: EloRatingSystem = field(default_factory=EloRatingSystem)
    form: FormTracker = field(default_factory=FormTracker)
    pedigree: PedigreeTracker = field(default_factory=PedigreeTracker)

    def _result_label(self, home_score: int, away_score: int) -> int:
        """Target 1X2 desde la óptica del local: 0=local gana, 1=empate, 2=visita."""
        if home_score > away_score:
            return 0
        if home_score == away_score:
            return 1
        return 2

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Genera el DataFrame de features+target a partir del histórico.

        Args:
            matches: Resultados validados y ordenados por fecha (ver loader).

        Returns:
            DataFrame con una fila por partido y todas las columnas de features,
            más ``home_score``, ``away_score`` y ``result`` (target 1X2).
        """
        rows: list[dict[str, float]] = []

        for row in matches.itertuples(index=False):
            home, away = row.home_team, row.away_team
            neutral = bool(row.neutral)

            # --- 1. Snapshot causal (features PRE-partido) -------------------
            elo_h, elo_a = self.elo.pre_match_ratings(home, away, neutral)
            form_h = self.form.features(home)
            form_a = self.form.features(away)
            ped_h_m, ped_h_w = self.pedigree.get_pedigree(home)
            ped_a_m, ped_a_w = self.pedigree.get_pedigree(away)

            feat: dict[str, float] = {
                "date": row.date,
                "home_team": home,
                "away_team": away,
                "tournament": row.tournament,
                # Elo
                "elo_home": elo_h,
                "elo_away": elo_a,
                "elo_diff": elo_h - elo_a,
                # Forma
                "form_ppg_home": form_h["form_ppg"],
                "form_ppg_away": form_a["form_ppg"],
                "avg_gf_home": form_h["avg_gf"],
                "avg_ga_home": form_h["avg_ga"],
                "avg_gf_away": form_a["avg_gf"],
                "avg_ga_away": form_a["avg_ga"],
                "win_streak_home": form_h["win_streak"],
                "win_streak_away": form_a["win_streak"],
                # Pedigrí
                "wc_matches_home": float(ped_h_m),
                "wc_wins_home": float(ped_h_w),
                "wc_matches_away": float(ped_a_m),
                "wc_wins_away": float(ped_a_w),
                # Entorno
                "is_home_host": float(home in HOST_NATIONS and not neutral),
                "neutral": float(neutral),
                # Targets
                "home_score": int(row.home_score),
                "away_score": int(row.away_score),
                "result": self._result_label(row.home_score, row.away_score),
            }
            rows.append(feat)

            # --- 2. Actualizar estado con el resultado real -----------------
            self.elo.update(
                home, away, row.home_score, row.away_score,
                row.tournament, neutral,
            )
            self.form.update(home, away, row.home_score, row.away_score)
            self.pedigree.update(
                home, away, row.home_score, row.away_score, row.tournament
            )

        df = pd.DataFrame(rows)
        logger.info("Dataset de features construido: %d filas, %d cols",
                    len(df), df.shape[1])
        return df

    @property
    def team_state(self) -> TeamState:
        """Estado final de los trackers (para la simulación 2026)."""
        return TeamState(elo=self.elo, form=self.form, pedigree=self.pedigree)


FEATURE_COLUMNS: list[str] = [
    "elo_home", "elo_away", "elo_diff",
    "form_ppg_home", "form_ppg_away",
    "avg_gf_home", "avg_ga_home", "avg_gf_away", "avg_ga_away",
    "win_streak_home", "win_streak_away",
    "wc_matches_home", "wc_wins_home", "wc_matches_away", "wc_wins_away",
    "is_home_host", "neutral",
]
