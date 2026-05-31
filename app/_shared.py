"""Utilidades compartidas por las páginas de Streamlit.

Carga (con caché) los modelos entrenados y expone helpers para construir el
motor de partidos y el simulador. El estado de los equipos es el **estado vivo**:
la base entrenada con el histórico + el replay de los resultados reales del
Mundial 2026 que el usuario haya ingresado (aprendizaje online).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# Permite importar ``src`` cuando Streamlit corre desde la raíz del proyecto.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    ENSEMBLE_MODEL_PATH,
    LIVE_RESULTS_CSV,
    POISSON_MODEL_PATH,
    TEAM_STATE_PATH,
)
from src.features.builder import TeamState  # noqa: E402
from src.live.updater import (  # noqa: E402
    build_live_state,
    load_live_results,
)
from src.simulation.match_engine import MatchEngine  # noqa: E402
from src.simulation.monte_carlo import MonteCarloSimulator  # noqa: E402


def artifacts_exist() -> bool:
    """True si los tres artefactos de modelo están presentes."""
    return all(
        p.exists()
        for p in (ENSEMBLE_MODEL_PATH, POISSON_MODEL_PATH, TEAM_STATE_PATH)
    )


@st.cache_resource(show_spinner="Cargando modelos entrenados...")
def load_models() -> tuple:
    """Carga (cacheado) ensemble, poisson y el team_state BASE (histórico)."""
    ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
    poisson = joblib.load(POISSON_MODEL_PATH)
    base_state = joblib.load(TEAM_STATE_PATH)
    return ensemble, poisson, base_state


def _live_log_signature() -> float:
    """Marca de tiempo del log en vivo (cambia => invalida la caché del estado)."""
    return LIVE_RESULTS_CSV.stat().st_mtime if LIVE_RESULTS_CSV.exists() else 0.0


@st.cache_resource(show_spinner="Aplicando resultados del Mundial...")
def get_live_state(_signature: float) -> TeamState:
    """Estado vivo = base + replay del log. Cacheado por firma del log.

    El argumento ``_signature`` (prefijo ``_`` para que Streamlit no intente
    hashear el contenido) fuerza el recálculo cuando el CSV de resultados cambia.
    """
    _, _, base_state = load_models()
    return build_live_state(base_state, load_live_results())


def current_state() -> TeamState:
    """Devuelve el estado de equipos vigente (con los resultados ingresados)."""
    return get_live_state(_live_log_signature())


def make_engine(seed: int = 42) -> MatchEngine:
    """Crea un ``MatchEngine`` usando el estado vivo y RNG sembrado."""
    ensemble, poisson, _ = load_models()
    return MatchEngine(
        team_state=current_state(),
        poisson=poisson,
        ensemble=ensemble,
        rng=np.random.default_rng(seed),
    )


def make_simulator(seed: int = 42) -> MonteCarloSimulator:
    """Crea un ``MonteCarloSimulator`` listo para correr (estado vivo)."""
    return MonteCarloSimulator(engine=make_engine(seed))


def require_models() -> bool:
    """Muestra aviso si faltan modelos. Devuelve True si están listos."""
    if not artifacts_exist():
        st.error(
            "⚠️ No se encontraron modelos entrenados.\n\n"
            "Ejecuta primero:\n```bash\npython -m src.models.train\n```"
        )
        return False
    return True


def list_teams() -> list[str]:
    """Lista de selecciones con rating Elo conocido, ordenadas por fuerza.

    Usa el estado vivo para que el orden refleje los resultados ingresados.
    """
    ratings = current_state().elo.ratings
    return sorted(ratings, key=lambda t: ratings[t], reverse=True)
