"""Aprendizaje online a partir de resultados reales del Mundial 2026.

Estrategia (segura y reversible):
    - El ``TeamState`` entrenado con el histórico (``team_state.joblib``) es la
      BASE inmutable: nunca se sobrescribe.
    - Cada resultado real ingresado por el usuario se anexa a un log CSV
      (``LIVE_RESULTS_CSV``).
    - El estado "vivo" se reconstruye SIEMPRE como: base (copia profunda) +
      replay cronológico del log. Esto hace la operación idempotente: borrar o
      editar una fila del log y reconstruir deja el estado exactamente como si
      esa fila nunca/así hubiera existido, sin residuos.

Como los modelos (Poisson, ensamble) predicen a partir del ``TeamState``,
actualizar el estado mejora las predicciones al instante, sin reentrenar.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import pandas as pd

from src.config import LIVE_RESULTS_CSV, TEAM_STATE_PATH
from src.data.cleaner import canonicalize_team
from src.features.builder import TeamState
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Esquema del log de resultados en vivo. El usuario debe aportar todo esto para
# que los trackers (Elo/forma/pedigrí) aprendan correctamente.
LIVE_COLUMNS = [
    "date",         # fecha del partido (YYYY-MM-DD)
    "home_team",    # equipo local (o "equipo A" si es cancha neutral)
    "away_team",    # equipo visitante (o "equipo B")
    "home_score",   # goles del local
    "away_score",   # goles del visitante
    "tournament",   # competición (normalmente "FIFA World Cup")
    "neutral",      # True si se juega en sede neutral (sin ventaja de localía)
    "stage",        # fase: "Group", "Round of 32", ... (informativo)
]


def empty_live_df() -> pd.DataFrame:
    """DataFrame vacío con el esquema del log en vivo."""
    return pd.DataFrame(columns=LIVE_COLUMNS)


def load_live_results() -> pd.DataFrame:
    """Carga el log de resultados reales (vacío si aún no existe)."""
    if not LIVE_RESULTS_CSV.exists():
        return empty_live_df()
    df = pd.read_csv(LIVE_RESULTS_CSV)
    for col in LIVE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[LIVE_COLUMNS]


def save_live_results(df: pd.DataFrame) -> None:
    """Persiste el log completo (sobrescribe con el contenido dado)."""
    LIVE_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[LIVE_COLUMNS].to_csv(LIVE_RESULTS_CSV, index=False)
    logger.info("Log en vivo guardado: %d resultados", len(df))


@dataclass
class MatchResultInput:
    """Resultado de un partido ingresado por el usuario, ya validado."""

    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str = "FIFA World Cup"
    neutral: bool = True
    stage: str = "Group"

    def as_row(self) -> dict:
        """Convierte a fila de log con nombres de equipo canonicalizados."""
        return {
            "date": self.date,
            "home_team": canonicalize_team(self.home_team),
            "away_team": canonicalize_team(self.away_team),
            "home_score": int(self.home_score),
            "away_score": int(self.away_score),
            "tournament": self.tournament,
            "neutral": bool(self.neutral),
            "stage": self.stage,
        }


def validate_result(
    home: str,
    away: str,
    home_score,
    away_score,
    known_teams: set[str] | None = None,
) -> list[str]:
    """Valida una entrada de resultado. Devuelve lista de errores (vacía = OK).

    Args:
        home, away: Selecciones.
        home_score, away_score: Goles (deben ser enteros >= 0).
        known_teams: Si se aporta, exige que ambos equipos existan en el modelo.
    """
    errors: list[str] = []
    if home == away:
        errors.append("Los dos equipos deben ser distintos.")
    for label, score in (("local", home_score), ("visitante", away_score)):
        try:
            s = int(score)
            if s < 0:
                errors.append(f"Los goles del {label} no pueden ser negativos.")
        except (TypeError, ValueError):
            errors.append(f"Los goles del {label} deben ser un número entero.")
    if known_teams is not None:
        for t in (home, away):
            if canonicalize_team(t) not in known_teams:
                errors.append(f"El equipo '{t}' no está en el modelo.")
    return errors


def append_result(result: MatchResultInput) -> pd.DataFrame:
    """Anexa un resultado al log y lo persiste. Devuelve el log actualizado."""
    df = load_live_results()
    df = pd.concat([df, pd.DataFrame([result.as_row()])], ignore_index=True)
    save_live_results(df)
    return df


def delete_result(index: int) -> pd.DataFrame:
    """Elimina la fila ``index`` del log y reescribe. Devuelve el log nuevo."""
    df = load_live_results()
    if 0 <= index < len(df):
        df = df.drop(df.index[index]).reset_index(drop=True)
        save_live_results(df)
    return df


def _replay_row(state: TeamState, row: pd.Series) -> None:
    """Aplica un resultado a los tres trackers del estado (en orden causal)."""
    home = canonicalize_team(str(row["home_team"]))
    away = canonicalize_team(str(row["away_team"]))
    hs, as_ = int(row["home_score"]), int(row["away_score"])
    tournament = str(row.get("tournament", "FIFA World Cup"))
    neutral = bool(row.get("neutral", True))

    state.elo.update(home, away, hs, as_, tournament, neutral)
    state.form.update(home, away, hs, as_)
    state.pedigree.update(home, away, hs, as_, tournament)


def build_live_state(base_state: TeamState, live_df: pd.DataFrame) -> TeamState:
    """Reconstruye el estado vivo = copia de la base + replay del log.

    No modifica ``base_state`` (trabaja sobre una copia profunda), por lo que la
    operación es totalmente reversible y repetible.

    Args:
        base_state: Estado entrenado con el histórico (inmutable).
        live_df: Log de resultados reales del Mundial.

    Returns:
        Un ``TeamState`` nuevo con el conocimiento del histórico + lo ocurrido.
    """
    state = copy.deepcopy(base_state)
    if live_df is None or live_df.empty:
        return state
    ordered = live_df.sort_values("date", kind="stable")
    for _, row in ordered.iterrows():
        try:
            _replay_row(state, row)
        except Exception as exc:  # noqa: BLE001 - una fila mala no debe romper todo
            logger.warning("Fila de log ignorada (%s): %s", exc, dict(row))
    return state


def elo_deltas(
    base_state: TeamState, live_state: TeamState
) -> dict[str, float]:
    """Cambio de Elo por equipo entre la base y el estado vivo (para la UI)."""
    deltas: dict[str, float] = {}
    teams = set(base_state.elo.ratings) | set(live_state.elo.ratings)
    for t in teams:
        before = base_state.elo.get(t)
        after = live_state.elo.get(t)
        if abs(after - before) > 1e-9:
            deltas[t] = after - before
    return deltas


def load_base_state() -> TeamState:
    """Carga el estado base entrenado desde disco."""
    import joblib

    return joblib.load(TEAM_STATE_PATH)
