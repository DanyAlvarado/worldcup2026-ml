"""Tests del sistema Elo."""
from src.features.elo import EloRatingSystem


def test_base_rating_for_unknown_team():
    elo = EloRatingSystem()
    assert elo.get("Atlantis") == elo.config.base_rating


def test_winner_gains_loser_loses():
    elo = EloRatingSystem()
    pre_h, pre_a = elo.update("A", "B", 2, 0, "Friendly", neutral=True)
    assert pre_h == pre_a == 1500.0
    assert elo.get("A") > 1500.0
    assert elo.get("B") < 1500.0


def test_zero_sum_update():
    elo = EloRatingSystem()
    elo.update("A", "B", 1, 0, "Friendly", neutral=True)
    assert round(elo.get("A") + elo.get("B"), 6) == 3000.0


def test_home_advantage_raises_effective_rating():
    elo = EloRatingSystem()
    h_eff, a_eff = elo.pre_match_ratings("A", "B", neutral=False)
    assert h_eff > a_eff
