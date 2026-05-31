"""Fase de grupos con reglas de desempate FIFA.

Cada grupo juega un round-robin (6 partidos). El ranking respeta el orden FIFA:
puntos -> diferencia de goles -> goles a favor -> (desempate aleatorio como
proxy de fair play / sorteo). Devuelve 1ros, 2dos y los terceros con su registro
para calcular los 8 mejores terceros a nivel torneo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from src.simulation.match_engine import MatchEngine


@dataclass
class TeamRecord:
    """Registro acumulado de un equipo en la fase de grupos."""

    team: str
    points: int = 0
    gf: int = 0
    ga: int = 0
    played: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def add_result(self, scored: int, conceded: int) -> None:
        self.played += 1
        self.gf += scored
        self.ga += conceded
        if scored > conceded:
            self.points += 3
        elif scored == conceded:
            self.points += 1


@dataclass
class GroupResult:
    """Salida de un grupo simulado."""

    name: str
    first: TeamRecord
    second: TeamRecord
    third: TeamRecord
    standings: list[TeamRecord] = field(default_factory=list)


def _sort_key(rec: TeamRecord, jitter: float) -> tuple:
    """Clave de orden FIFA (mayor es mejor); ``jitter`` rompe empates."""
    return (rec.points, rec.gd, rec.gf, jitter)


def simulate_group(
    name: str, teams: list[str], engine: MatchEngine
) -> GroupResult:
    """Simula un grupo round-robin y devuelve su clasificación ordenada.

    Args:
        name: Identificador del grupo (p. ej. "A").
        teams: Las 4 selecciones del grupo.
        engine: Motor de partidos para muestrear marcadores.

    Returns:
        ``GroupResult`` con 1ro, 2do, 3ro y standings completos.
    """
    records = {t: TeamRecord(t) for t in teams}

    for home, away in combinations(teams, 2):
        hs, as_ = engine.sample_score(home, away, neutral=True)
        records[home].add_result(hs, as_)
        records[away].add_result(as_, hs)

    jitters = {t: engine.rng.random() for t in teams}
    ordered = sorted(
        records.values(),
        key=lambda r: _sort_key(r, jitters[r.team]),
        reverse=True,
    )
    return GroupResult(
        name=name,
        first=ordered[0],
        second=ordered[1],
        third=ordered[2],
        standings=ordered,
    )
