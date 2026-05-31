"""Carga y validación de datos crudos.

Descarga el dataset de Kaggle vía ``kagglehub``, copia los CSV a ``data/raw`` y
los carga con validación de esquema. Si los archivos ya existen localmente, evita
volver a descargar. El dataset original nombra los archivos ``Match_Results.csv``
y ``Penalty_Shootouts.csv``; aquí se normalizan a ``results.csv`` y
``shootouts.csv``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, RESULTS_CSV, SHOOTOUTS_CSV
from src.data.cleaner import normalize_team_names
from src.utils.logging import get_logger

logger = get_logger(__name__)

RESULTS_SCHEMA = {
    "date", "home_team", "away_team", "home_score",
    "away_score", "tournament", "city", "country", "neutral",
}

KAGGLE_RESULTS_DS = "muhammadehsan02/global-football-results-18722024"
# (origen en el dataset -> destino normalizado)
_FILE_MAP = {
    "Match_Results.csv": RESULTS_CSV,
    "Penalty_Shootouts.csv": SHOOTOUTS_CSV,
}


def ensure_raw_data() -> None:
    """Garantiza que ``results.csv`` y ``shootouts.csv`` estén en ``data/raw``.

    Descarga desde Kaggle solo si faltan. Lanza ``RuntimeError`` si la descarga
    falla y no hay copia local previa.
    """
    if RESULTS_CSV.exists() and SHOOTOUTS_CSV.exists():
        logger.info("Datos crudos ya presentes en %s", DATA_RAW)
        return

    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "kagglehub no está instalado y faltan los CSV locales. "
            "Instala con `pip install kagglehub`."
        ) from exc

    logger.info("Descargando dataset de Kaggle: %s", KAGGLE_RESULTS_DS)
    cache_path = Path(kagglehub.dataset_download(KAGGLE_RESULTS_DS))

    for src_name, dest in _FILE_MAP.items():
        src = cache_path / src_name
        if not src.exists():
            raise RuntimeError(f"No se encontró {src_name} en el dataset.")
        shutil.copy(src, dest)
        logger.info("Copiado %s -> %s", src_name, dest.name)


def _validate_schema(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: faltan columnas {missing}")


def load_results() -> pd.DataFrame:
    """Carga el histórico de partidos internacionales, validado y ordenado.

    Returns:
        DataFrame con ``date`` como datetime, ordenado cronológicamente, sin
        filas con marcadores nulos.
    """
    ensure_raw_data()
    df = pd.read_csv(RESULTS_CSV)
    _validate_schema(df, RESULTS_SCHEMA, "results.csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df = df.astype({"home_score": "int64", "away_score": "int64"})
    df["neutral"] = df["neutral"].astype(bool)
    df = normalize_team_names(df)
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "Cargados %d partidos (%s a %s)",
        len(df), df["date"].min().date(), df["date"].max().date(),
    )
    return df
