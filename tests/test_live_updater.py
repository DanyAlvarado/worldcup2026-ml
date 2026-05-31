"""Tests del aprendizaje online (ingreso de resultados reales)."""
import copy

import pandas as pd

from src.features.builder import FeatureBuilder
from src.live.updater import (
    MatchResultInput,
    build_live_state,
    elo_deltas,
    validate_result,
)


def _base_state():
    """Estado base mínimo a partir de unos pocos partidos sintéticos."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
            "home_team": ["Argentina", "Saudi Arabia", "Argentina"],
            "away_team": ["Saudi Arabia", "Brazil", "Brazil"],
            "home_score": [3, 0, 2],
            "away_score": [0, 2, 2],
            "tournament": ["Friendly"] * 3,
            "neutral": [True, True, True],
        }
    )
    return FeatureBuilder().build(matches), matches


def test_result_updates_elo_in_expected_direction():
    state, _ = _base_state()
    base = state if hasattr(state, "elo") else None
    # build returns a DataFrame; obtener el TeamState desde un builder fresco
    builder = FeatureBuilder()
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-01-01"]),
            "home_team": ["Argentina"],
            "away_team": ["Saudi Arabia"],
            "home_score": [3],
            "away_score": [0],
            "tournament": ["Friendly"],
            "neutral": [True],
        }
    )
    builder.build(matches)
    base_state = builder.team_state

    arg0 = base_state.elo.get("Argentina")
    sau0 = base_state.elo.get("Saudi Arabia")

    log = pd.DataFrame([
        MatchResultInput("2026-06-12", "Saudi Arabia", "Argentina", 2, 1).as_row()
    ])
    live = build_live_state(base_state, log)

    assert live.elo.get("Argentina") < arg0
    assert live.elo.get("Saudi Arabia") > sau0


def test_build_live_state_does_not_mutate_base():
    builder = FeatureBuilder()
    builder.build(pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01"]),
        "home_team": ["Argentina"], "away_team": ["Brazil"],
        "home_score": [1], "away_score": [0],
        "tournament": ["Friendly"], "neutral": [True],
    }))
    base_state = builder.team_state
    snapshot = copy.deepcopy(base_state.elo.ratings)

    log = pd.DataFrame([
        MatchResultInput("2026-06-12", "Brazil", "Argentina", 3, 0).as_row()
    ])
    build_live_state(base_state, log)

    # La base no debe cambiar (operación sobre copia profunda).
    assert base_state.elo.ratings == snapshot


def test_empty_log_returns_equivalent_state():
    builder = FeatureBuilder()
    builder.build(pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01"]),
        "home_team": ["Argentina"], "away_team": ["Brazil"],
        "home_score": [1], "away_score": [0],
        "tournament": ["Friendly"], "neutral": [True],
    }))
    base_state = builder.team_state
    live = build_live_state(base_state, pd.DataFrame())
    assert live.elo.ratings == base_state.elo.ratings
    assert elo_deltas(base_state, live) == {}


def test_validate_result_catches_errors():
    assert validate_result("Brazil", "Brazil", 1, 0) != []      # mismo equipo
    assert validate_result("Brazil", "Argentina", -1, 0) != []  # goles negativos
    assert validate_result("Brazil", "Argentina", 2, 1) == []   # válido
