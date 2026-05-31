"""Features de entorno: localía, altitud y fatiga por viaje.

Estas variables capturan el contexto específico del Mundial 2026 (tres sedes,
grandes distancias, altitud de CDMX). Para el histórico de entrenamiento se
derivan del país anfitrión y la bandera ``neutral``; para la simulación 2026 se
calculan respecto a las sedes reales.
"""
from __future__ import annotations

from src.config import (
    ALTITUDE_THRESHOLD_M,
    HOST_NATIONS,
    VENUES_2026,
    Venue,
)
from src.utils.geo import haversine_km

# Aproximación del "centro geográfico" de cada selección para estimar viajes
# desde casa hasta las sedes (solo se usa una muestra representativa; se cae a
# un valor neutro si el equipo no está mapeado).
TEAM_HOME_COORDS: dict[str, tuple[float, float]] = {
    "Brazil": (-15.78, -47.93), "Argentina": (-34.60, -58.38),
    "France": (48.85, 2.35), "Spain": (40.42, -3.70),
    "England": (51.51, -0.13), "Germany": (52.52, 13.40),
    "Portugal": (38.72, -9.14), "Netherlands": (52.37, 4.90),
    "Belgium": (50.85, 4.35), "Croatia": (45.81, 15.98),
    "Mexico": (19.43, -99.13), "United States": (38.90, -77.04),
    "Canada": (45.42, -75.70), "Japan": (35.68, 139.69),
    "South Korea": (37.57, 126.98), "Morocco": (34.02, -6.83),
    "Senegal": (14.72, -17.47), "Ghana": (5.60, -0.19),
    "Uruguay": (-34.90, -56.19), "Colombia": (4.71, -74.07),
    "Nigeria": (9.08, 7.40), "Australia": (-35.28, 149.13),
}
_NEUTRAL_COORDS = (20.0, -90.0)  # punto medio aproximado de Norteamérica


def altitude_penalty(altitude_m: float, team_home_altitude_m: float) -> float:
    """Penalización de altitud relativa para el equipo visitante.

    Equipos acostumbrados al nivel del mar sufren más en CDMX (2240 m). Devuelve
    un valor en [0, 1] proporcional al desnivel por encima del umbral.

    Args:
        altitude_m: Altitud de la sede del partido.
        team_home_altitude_m: Altitud habitual del equipo.

    Returns:
        Penalización normalizada (0 = sin efecto).
    """
    if altitude_m < ALTITUDE_THRESHOLD_M:
        return 0.0
    delta = max(0.0, altitude_m - team_home_altitude_m)
    return min(1.0, delta / 2500.0)


def travel_distance_km(team: str, venue: Venue) -> float:
    """Distancia desde la sede 'casa' del equipo hasta la sede del partido."""
    lat, lon = TEAM_HOME_COORDS.get(team, _NEUTRAL_COORDS)
    return haversine_km(lat, lon, venue.lat, venue.lon)


def is_host(team: str) -> bool:
    """True si el equipo es una de las tres naciones anfitrionas."""
    return team in HOST_NATIONS


def venue_by_city(city: str) -> Venue | None:
    """Busca una sede 2026 por nombre de ciudad (None si no es sede)."""
    return VENUES_2026.get(city)
