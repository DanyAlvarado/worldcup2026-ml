"""'Pedigrí' mundialista: experiencia histórica de cada selección.

Cuenta de forma causal (acumulada hasta la fecha del partido) la participación y
el éxito de cada equipo en Copas del Mundo. Selecciones con más recorrido tienden
a rendir mejor en fases decisivas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

WORLD_CUP_TOURNAMENT = "FIFA World Cup"


@dataclass
class PedigreeTracker:
    """Acumula partidos y victorias en Copa del Mundo por equipo.

    Igual que el Elo, se actualiza recorriendo el histórico cronológicamente:
    primero se lee el pedigrí vigente (feature), luego se incorpora el partido.
    """

    wc_matches: dict[str, int] = field(default_factory=dict)
    wc_wins: dict[str, int] = field(default_factory=dict)

    def get_pedigree(self, team: str) -> tuple[int, int]:
        """Partidos y victorias acumuladas en Mundiales hasta ahora."""
        return self.wc_matches.get(team, 0), self.wc_wins.get(team, 0)

    def update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        tournament: str,
    ) -> None:
        """Incorpora un partido al historial si es de Copa del Mundo."""
        if tournament != WORLD_CUP_TOURNAMENT:
            return
        for team in (home, away):
            self.wc_matches[team] = self.wc_matches.get(team, 0) + 1
        if home_score > away_score:
            self.wc_wins[home] = self.wc_wins.get(home, 0) + 1
        elif away_score > home_score:
            self.wc_wins[away] = self.wc_wins.get(away, 0) + 1
