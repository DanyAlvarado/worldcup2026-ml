"""Utilidades geográficas para estimar fatiga por viaje entre sedes."""
from __future__ import annotations

import math


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia del círculo máximo entre dos puntos (km).

    Args:
        lat1, lon1: Latitud/longitud del origen en grados decimales.
        lat2, lon2: Latitud/longitud del destino en grados decimales.

    Returns:
        Distancia en kilómetros.
    """
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))
