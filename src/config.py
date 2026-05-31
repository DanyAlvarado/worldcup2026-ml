"""Configuración central del proyecto Mundial 2026 ML.

Contiene rutas, constantes del dominio, hiperparámetros y la definición de los
grupos del torneo. Centralizar esto evita "números mágicos" dispersos y facilita
la reproducibilidad (todas las semillas viven aquí).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Archivos esperados (copiados desde la caché de kagglehub por src/data/loader.py)
RESULTS_CSV = DATA_RAW / "results.csv"
SHOOTOUTS_CSV = DATA_RAW / "shootouts.csv"

PROCESSED_FEATURES = DATA_PROCESSED / "features.parquet"

# Log de resultados REALES del Mundial 2026 ingresados por el usuario.
# Se usa para el aprendizaje online (ver src/live/updater.py).
LIVE_RESULTS_CSV = DATA_PROCESSED / "live_results_2026.csv"

# Artefactos de modelo
POISSON_MODEL_PATH = MODELS_DIR / "poisson_dixon_coles.joblib"
ENSEMBLE_MODEL_PATH = MODELS_DIR / "lgbm_1x2.joblib"
TEAM_STATE_PATH = MODELS_DIR / "team_state.joblib"

# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Hiperparámetros del sistema Elo
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EloConfig:
    """Parámetros del sistema Elo estilo World Football Elo Ratings."""

    base_rating: float = 1500.0
    k_factor: float = 40.0          # K base (ajustado por importancia del torneo)
    home_advantage: float = 65.0    # puntos Elo de ventaja de localía
    goal_diff_scaling: bool = True  # escalar K por diferencia de goles


# Multiplicador de importancia del torneo (afecta K y la fase de "pedigrí")
TOURNAMENT_WEIGHT: dict[str, float] = {
    "FIFA World Cup": 60.0,
    "FIFA World Cup qualification": 40.0,
    "UEFA Euro": 50.0,
    "Copa América": 50.0,
    "African Cup of Nations": 40.0,
    "AFC Asian Cup": 40.0,
    "Confederations Cup": 40.0,
    "UEFA Nations League": 35.0,
    "Friendly": 20.0,
}
DEFAULT_TOURNAMENT_WEIGHT = 30.0

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
ROLLING_WINDOW = 5          # tamaño de ventana para medias móviles / forma
MIN_MATCHES_FOR_STATS = 3   # partidos mínimos antes de confiar en stats

# --------------------------------------------------------------------------- #
# Sedes Mundial 2026 (altitud en metros, coordenadas para Haversine)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Venue:
    city: str
    country: str
    lat: float
    lon: float
    altitude_m: float


VENUES_2026: dict[str, Venue] = {
    "Mexico City": Venue("Mexico City", "Mexico", 19.4326, -99.1332, 2240.0),
    "Guadalajara": Venue("Guadalajara", "Mexico", 20.6597, -103.3496, 1566.0),
    "Monterrey": Venue("Monterrey", "Mexico", 25.6866, -100.3161, 540.0),
    "Toronto": Venue("Toronto", "Canada", 43.6532, -79.3832, 76.0),
    "Vancouver": Venue("Vancouver", "Canada", 49.2827, -123.1207, 4.0),
    "Atlanta": Venue("Atlanta", "USA", 33.7490, -84.3880, 320.0),
    "Boston": Venue("Boston", "USA", 42.3601, -71.0589, 43.0),
    "Dallas": Venue("Dallas", "USA", 32.7767, -96.7970, 131.0),
    "Houston": Venue("Houston", "USA", 29.7604, -95.3698, 24.0),
    "Kansas City": Venue("Kansas City", "USA", 39.0997, -94.5786, 277.0),
    "Los Angeles": Venue("Los Angeles", "USA", 34.0522, -118.2437, 87.0),
    "Miami": Venue("Miami", "USA", 25.7617, -80.1918, 2.0),
    "New York": Venue("New York", "USA", 40.7128, -74.0060, 10.0),
    "Philadelphia": Venue("Philadelphia", "USA", 39.9526, -75.1652, 12.0),
    "San Francisco": Venue("San Francisco", "USA", 37.7749, -122.4194, 16.0),
    "Seattle": Venue("Seattle", "USA", 47.6062, -122.3321, 53.0),
}

HOST_NATIONS = ("United States", "Mexico", "Canada")
ALTITUDE_THRESHOLD_M = 1500.0  # a partir de aquí el efecto altitud es notable

# --------------------------------------------------------------------------- #
# Simulación
# --------------------------------------------------------------------------- #
N_SIMULATIONS = 10_000


@dataclass
class TournamentConfig:
    """Estructura oficial del Mundial 2026: 48 equipos, 12 grupos de 4."""

    n_groups: int = 12
    teams_per_group: int = 4
    # Clasifican: 2 primeros de cada grupo (24) + 8 mejores terceros = 32
    best_thirds_qualify: int = 8
    groups: dict[str, list[str]] = field(default_factory=dict)


# Sorteo OFICIAL del Mundial 2026, realizado el 5 de diciembre de 2025 en
# Washington D.C. (verificado con NBC Sports y DAZN). Los nombres usan la
# convención canónica del dataset tras src/data/cleaner.py:
#   "Korea Republic" -> "South Korea", "Türkiye" -> "Turkey",
#   "Czechia" y "DR Congo" tal cual existen en el histórico.
# Posiciones de anfitriones según el sorteo: México A1, Canadá B1, EE.UU. D1.
GROUPS_2026: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
