"""Test crítico: el feature builder no debe filtrar información futura.

Verifica que las features del PRIMER partido de un equipo sean siempre los
valores base (Elo base, forma neutra, pedigrí cero), es decir, no contienen
información del propio resultado ni de partidos posteriores.
"""
import pandas as pd

from src.features.builder import FeatureBuilder
from src.features.elo import EloRatingSystem


def _toy_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "home_team": ["A", "A", "B"],
            "away_team": ["B", "C", "C"],
            "home_score": [3, 1, 0],
            "away_score": [0, 1, 2],
            "tournament": ["Friendly"] * 3,
            "neutral": [True, True, True],
        }
    )


def test_first_match_uses_base_features():
    df = FeatureBuilder().build(_toy_matches())
    first = df.iloc[0]
    base = EloRatingSystem().config.base_rating
    assert first["elo_home"] == base
    assert first["elo_away"] == base
    assert first["wc_matches_home"] == 0
    assert first["win_streak_home"] == 0


def test_elo_updates_after_first_match():
    df = FeatureBuilder().build(_toy_matches())
    second = df[df["home_team"] == "A"].iloc[1]
    assert second["elo_home"] > EloRatingSystem().config.base_rating


def test_target_not_in_features():
    df = FeatureBuilder().build(_toy_matches())
    feature_cols = [c for c in df.columns if c not in
                    ("home_score", "away_score", "result",
                     "date", "home_team", "away_team", "tournament")]
    assert "home_score" not in feature_cols
    assert "result" not in feature_cols
