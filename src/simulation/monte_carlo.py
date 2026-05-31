"""Simulación Monte Carlo del Mundial 2026 completo.

Ejecuta el torneo N veces (por defecto 10,000), respetando la estructura oficial:
12 grupos de 4 -> clasifican 1ros, 2dos y los 8 mejores terceros (32) -> bracket
de eliminación directa. Agrega frecuencias de avance por ronda y probabilidad de
título por selección.

Etapas rastreadas:
    round32  -> alcanzó dieciseisavos (los 32 clasificados)
    round16  -> alcanzó octavos (16)
    quarter  -> alcanzó cuartos (8)
    semi     -> alcanzó semifinales (4)
    final    -> alcanzó la final (2)
    champion -> campeón (1)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import GROUPS_2026, N_SIMULATIONS, SEED
from src.simulation.group_stage import TeamRecord, simulate_group
from src.simulation.knockout import run_bracket
from src.simulation.match_engine import MatchEngine
from src.utils.logging import get_logger

logger = get_logger(__name__)

STAGES = ["round32", "round16", "quarter", "semi", "final", "champion"]


def _best_thirds(
    thirds: list[TeamRecord], n: int, rng: np.random.Generator
) -> list[str]:
    """Selecciona los ``n`` mejores terceros por criterio FIFA."""
    jitter = {id(t): rng.random() for t in thirds}
    ordered = sorted(
        thirds,
        key=lambda r: (r.points, r.gd, r.gf, jitter[id(r)]),
        reverse=True,
    )
    return [t.team for t in ordered[:n]]


@dataclass
class SimulationResults:
    """Resultados agregados de la simulación Monte Carlo."""

    n_sims: int
    stage_counts: dict[str, Counter] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Tabla equipo x probabilidad de alcanzar cada etapa (ordenada por título)."""
        teams: set[str] = set()
        for counter in self.stage_counts.values():
            teams.update(counter.keys())

        rows = []
        for team in teams:
            row: dict[str, object] = {"team": team}
            for stage in STAGES:
                row[stage] = self.stage_counts[stage][team] / self.n_sims
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("champion", ascending=False)
        df = df.rename(columns={
            "round32": "P(Dieciseisavos)", "round16": "P(Octavos)",
            "quarter": "P(Cuartos)", "semi": "P(Semis)",
            "final": "P(Final)", "champion": "P(Campeón)",
        })
        return df.reset_index(drop=True)


@dataclass
class MonteCarloSimulator:
    """Ejecuta y agrega múltiples simulaciones completas del torneo."""

    engine: MatchEngine
    groups: dict[str, list[str]] = field(default_factory=lambda: dict(GROUPS_2026))

    def _simulate_once(self) -> dict[str, list[str]]:
        """Una corrida completa: grupos + bracket. Devuelve avances por ronda."""
        firsts, seconds, thirds = [], [], []
        for name, teams in self.groups.items():
            res = simulate_group(name, teams, self.engine)
            firsts.append(res.first)
            seconds.append(res.second)
            thirds.append(res.third)

        best_thirds = _best_thirds(thirds, 8, self.engine.rng)
        qualified = (
            [r.team for r in firsts]
            + [r.team for r in seconds]
            + best_thirds
        )
        # Mezcla aleatoria del emparejamiento (proxy del cuadro real).
        self.engine.rng.shuffle(qualified)
        bracket = run_bracket(qualified, self.engine)
        bracket["round32"] = qualified
        return bracket

    def run(self, n_sims: int = N_SIMULATIONS) -> SimulationResults:
        """Ejecuta ``n_sims`` torneos y agrega los resultados.

        Args:
            n_sims: Número de simulaciones Monte Carlo.

        Returns:
            ``SimulationResults`` con conteos por etapa.
        """
        stage_counts: dict[str, Counter] = {s: Counter() for s in STAGES}

        for i in range(n_sims):
            bracket = self._simulate_once()
            for stage in STAGES:
                for team in bracket[stage]:
                    stage_counts[stage][team] += 1
            if (i + 1) % max(1, n_sims // 10) == 0:
                logger.info("Simulación %d/%d", i + 1, n_sims)

        return SimulationResults(n_sims=n_sims, stage_counts=stage_counts)


def build_simulator(team_state, poisson, ensemble, seed: int = SEED):
    """Factory: crea un ``MonteCarloSimulator`` con RNG sembrado."""
    rng = np.random.default_rng(seed)
    engine = MatchEngine(
        team_state=team_state, poisson=poisson, ensemble=ensemble, rng=rng
    )
    return MonteCarloSimulator(engine=engine)
