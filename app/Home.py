"""Página principal del dashboard Mundial 2026 ML.

Ejecutar con:
    streamlit run app/Home.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from _shared import artifacts_exist, list_teams, load_models

st.set_page_config(
    page_title="Mundial 2026 · ML Predictor",
    page_icon="🏆",
    layout="wide",
)

st.title("🏆 Mundial FIFA 2026 — Predictor de Machine Learning")
st.markdown(
    """
Pipeline **end-to-end** que predice partidos y simula el torneo completo
(48 equipos, 12 grupos) mediante **Monte Carlo** sobre modelos entrenados con
partidos internacionales reales (datos de Kaggle).

**Modelos:** Poisson bivariada (Dixon-Coles) para marcadores + LightGBM para
probabilidades 1X2, validados con *TimeSeriesSplit* (sin fuga de datos).
"""
)

if not artifacts_exist():
    st.warning(
        "⚠️ **Modelos no entrenados aún.** Ejecuta en la terminal:\n\n"
        "```bash\npython -m src.models.train\n```\n\n"
        "Esto descarga los datos de Kaggle, construye las features de forma "
        "causal, valida y persiste los modelos. Luego recarga esta página."
    )
    st.stop()

ensemble, poisson, team_state = load_models()
teams = list_teams()
ratings = team_state.elo.ratings

col1, col2, col3 = st.columns(3)
col1.metric("Selecciones en el modelo", len(teams))
col2.metric("Equipo #1 (Elo)", teams[0], f"{ratings[teams[0]]:.0f}")
col3.metric("Ventaja localía (Poisson)", f"{poisson.home_adv:.3f}")

st.subheader("📈 Top 20 selecciones por rating Elo")
top = pd.DataFrame(
    {"Selección": teams[:20], "Elo": [round(ratings[t], 1) for t in teams[:20]]}
)
top.index = range(1, len(top) + 1)
st.bar_chart(top.set_index("Selección"))
st.dataframe(top, use_container_width=True)

st.info(
    "Navega por las páginas de la izquierda: "
    "**📊 Estadísticas**, **⚔️ Simular Partido**, **🏆 Simular Torneo**."
)
