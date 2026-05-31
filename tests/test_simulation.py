"""Tests de la lógica de simulación y de los modelos."""
import numpy as np
import pandas as pd

from src.models.poisson_model import PoissonDixonColes
from src.simulation.group_stage import TeamRecord


def test_team_record_points():
    r = TeamRecord("A")
    r.add_result(2, 0)  # victoria
    r.add_result(1, 1)  # empate
    r.add_result(0, 3)  # derrota
    assert r.points == 4
    assert r.played == 3
    assert r.gd == -1


def test_poisson_score_matrix_is_probability():
    p = PoissonDixonColes()
    m = p.score_matrix(elo_diff=100.0, is_home_real=1.0)
    assert abs(m.sum() - 1.0) < 1e-6
    assert (m >= 0).all()


def test_poisson_proba_sums_to_one():
    p = PoissonDixonColes()
    X = pd.DataFrame({"elo_diff": [50.0, -200.0], "neutral": [1.0, 0.0]})
    proba = p.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_stronger_team_more_likely_to_win():
    p = PoissonDixonColes()
    X = pd.DataFrame({"elo_diff": [300.0], "neutral": [1.0]})
    proba = p.predict_proba(X)[0]
    assert proba[0] > proba[2]  # P(local gana) > P(visita gana)


def test_bracket_has_five_rounds_from_32():
    """El cuadro de 32 equipos produce 5 rondas: 16, 8, 4, 2, 1 (campeón)."""
    import numpy as np

    from src.simulation.knockout import run_bracket

    class _StubEngine:
        rng = np.random.default_rng(0)

        def sample_score(self, home, away, neutral=True):
            return (2, 1)  # gana siempre el primero -> determinista

    teams = [f"T{i}" for i in range(32)]
    rounds = run_bracket(teams, _StubEngine())
    assert [len(rounds[k]) for k in
            ("round16", "quarter", "semi", "final", "champion")] == [16, 8, 4, 2, 1]
    assert rounds["champion"] == ["T0"]
