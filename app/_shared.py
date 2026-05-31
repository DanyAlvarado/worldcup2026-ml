"""Utilidades compartidas por las páginas de Streamlit.

Carga (con caché) los modelos entrenados y expone helpers para construir el
motor de partidos y el simulador. Si los artefactos no existen, guía al usuario
para entrenar.
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
    POISSON_MODEL_PATH,
    TEAM_STATE_PATH,
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
    """Carga (cacheado) ensemble, poisson y team_state."""
    ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
    poisson = joblib.load(POISSON_MODEL_PATH)
    team_state = joblib.load(TEAM_STATE_PATH)
    return ensemble, poisson, team_state


def make_engine(seed: int = 42) -> MatchEngine:
    """Crea un ``MatchEngine`` con RNG sembrado."""
    ensemble, poisson, team_state = load_models()
    return MatchEngine(
        team_state=team_state,
        poisson=poisson,
        ensemble=ensemble,
        rng=np.random.default_rng(seed),
    )


def make_simulator(seed: int = 42) -> MonteCarloSimulator:
    """Crea un ``MonteCarloSimulator`` listo para correr."""
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
    """Lista de selecciones con rating Elo conocido, ordenadas por fuerza."""
    _, _, team_state = load_models()
    ratings = team_state.elo.ratings
    return sorted(ratings, key=lambda t: ratings[t], reverse=True)
