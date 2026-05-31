"""Normalización de nombres de selecciones.

Algunas selecciones aparecen en el histórico bajo nombres distintos por
renombramientos oficiales (p. ej. "Czech Republic" -> "Czechia" desde 2022).
Si no se unifican, el sistema Elo trataría a la misma entidad como dos equipos
y partiría su historia, dejando al equipo "nuevo" con un rating casi base.

Este módulo define un mapa de alias canónicos y una función para aplicarlo sobre
las columnas de equipos del histórico. Los nombres canónicos coinciden con los
usados en ``GROUPS_2026`` (config) para que la simulación encuentre el Elo
correcto de cada selección del Mundial 2026.
"""
from __future__ import annotations

import pandas as pd

# alias (como aparece en el dataset) -> nombre canónico
TEAM_ALIASES: dict[str, str] = {
    # República Checa: unificar la entidad sucesora.
    "Czech Republic": "Czechia",
    # Variantes ortográficas / de transliteración frecuentes.
    "Cote d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Republic of Ireland": "Ireland",
    "Congo DR": "DR Congo",
    "DR Congo ": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
}


def canonicalize_team(name: str) -> str:
    """Devuelve el nombre canónico de una selección (o el original si no hay alias)."""
    return TEAM_ALIASES.get(name, name)


def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la canonicalización a ``home_team`` y ``away_team`` (copia)."""
    out = df.copy()
    out["home_team"] = out["home_team"].map(canonicalize_team)
    out["away_team"] = out["away_team"].map(canonicalize_team)
    return out
