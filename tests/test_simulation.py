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


def test_official_bracket_structure():
    """El bracket oficial produce exactamente los tamaños correctos por ronda."""
    import numpy as np
    from src.simulation.bracket_2026 import simulate_bracket_2026
    from src.simulation.group_stage import GroupResult, TeamRecord

    class _StubEngine:
        rng = np.random.default_rng(0)
        def sample_score(self, home, away, neutral=True):
            return (1, 0)  # el local siempre gana
        def win_probabilities(self, home, away, neutral=True):
            return {"home_win": 0.6, "draw": 0.2, "away_win": 0.2}

    # Construir resultados de grupo sintéticos
    letters = list("ABCDEFGHIJKL")
    group_results = {}
    thirds = []
    for i, g in enumerate(letters):
        base = i * 4
        teams = [f"{g}{j}" for j in range(4)]
        recs = [TeamRecord(t, points=9-j*3, gf=3-j, ga=j) for j, t in enumerate(teams)]
        gr = GroupResult(name=g, first=recs[0], second=recs[1],
                         third=recs[2], standings=recs)
        group_results[g] = gr
        thirds.append(recs[2])

    # Los 8 mejores terceros (primeros 8)
    best = thirds[:8]
    result = simulate_bracket_2026(group_results, best, _StubEngine())

    assert len(result["round32"]) == 16, "Round of 32 debe tener 16 ganadores"
    assert len(result["round16"]) == 8,  "Round of 16 debe tener 8 ganadores"
    assert len(result["quarter"]) == 4,  "Cuartos debe tener 4 ganadores"
    assert len(result["semi"])    == 2,  "Semis debe tener 2 ganadores"
    assert len(result["final"])   == 2,  "Final debe tener 2 finalistas"
    assert len(result["champion"]) == 1, "Un único campeón"
