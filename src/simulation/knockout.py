"""Fase eliminatoria del Mundial 2026 (32 equipos: dieciseisavos -> final).

Un partido de eliminación no admite empate: si el marcador muestreado es igual,
se resuelve con una moneda sesgada por la fuerza relativa (proxy de prórroga +
penales). El bracket parte de 32 equipos (24 clasificados directos + 8 mejores
terceros).
"""
from __future__ import annotations

from src.simulation.match_engine import MatchEngine


def play_knockout_match(home: str, away: str, engine: MatchEngine) -> str:
    """Resuelve un partido eliminatorio y devuelve el ganador.

    Args:
        home: Primer equipo del emparejamiento.
        away: Segundo equipo.
        engine: Motor de partidos.

    Returns:
        Nombre del equipo ganador.
    """
    hs, as_ = engine.sample_score(home, away, neutral=True)
    if hs > as_:
        return home
    if as_ > hs:
        return away
    # Empate -> prórroga/penales: moneda sesgada por probabilidad de victoria.
    probs = engine.win_probabilities(home, away, neutral=True)
    total = probs["home_win"] + probs["away_win"]
    p_home = probs["home_win"] / total if total > 0 else 0.5
    return home if engine.rng.random() < p_home else away


def run_bracket(qualified_32: list[str], engine: MatchEngine) -> dict[str, list[str]]:
    """Ejecuta el cuadro eliminatorio completo desde 32 equipos.

    Args:
        qualified_32: Lista ordenada de 32 clasificados (emparejamiento
            secuencial: 0 vs 1, 2 vs 3, ...).
        engine: Motor de partidos.

    Returns:
        Diccionario con los equipos que ALCANZARON cada etapa:
        ``round16`` (16), ``quarter`` (8), ``semi`` (4), ``final`` (2),
        ``champion`` (1). Son 5 rondas eliminatorias (log2(32)).
    """
    if len(qualified_32) != 32:
        raise ValueError(f"Se esperaban 32 equipos, llegaron {len(qualified_32)}")

    rounds: dict[str, list[str]] = {}
    # Cada etiqueta recibe los GANADORES de la ronda previa, es decir, los
    # equipos que avanzan a esa etapa (32->16 octavos, ..., 2->1 campeón).
    labels = ["round16", "quarter", "semi", "final", "champion"]
    current = list(qualified_32)

    for label in labels:
        winners = [
            play_knockout_match(current[i], current[i + 1], engine)
            for i in range(0, len(current), 2)
        ]
        rounds[label] = winners
        current = winners

    return rounds  # rounds["champion"] tiene un único elemento
