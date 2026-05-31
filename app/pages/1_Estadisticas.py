"""Página de estadísticas por selección: Elo, forma y pedigrí mundialista."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from _shared import list_teams, load_models, require_models

st.set_page_config(page_title="Estadísticas", page_icon="📊", layout="wide")
st.title("📊 Estadísticas por selección")

if not require_models():
    st.stop()

_, _, team_state = load_models()
teams = list_teams()

team = st.selectbox("Selecciona una selección", teams)

elo = team_state.elo.ratings[team]
form = team_state.form.features(team)
wc_m, wc_w = team_state.pedigree.get_pedigree(team)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rating Elo", f"{elo:.0f}")
c2.metric("Puntos/partido (últ. 5)", f"{form['form_ppg']:.2f}")
c3.metric("Racha", f"{int(form['win_streak']):+d}")
c4.metric("Partidos en Mundiales", wc_m)

c5, c6, c7 = st.columns(3)
c5.metric("Goles a favor (prom.)", f"{form['avg_gf']:.2f}")
c6.metric("Goles en contra (prom.)", f"{form['avg_ga']:.2f}")
c7.metric("Victorias en Mundiales", wc_w)

st.divider()
st.subheader("🔝 Comparativa de ratings Elo")

n = st.slider("Número de selecciones a mostrar", 5, 40, 20)
ratings = team_state.elo.ratings
df = pd.DataFrame(
    {"Selección": teams[:n], "Elo": [round(ratings[t], 1) for t in teams[:n]]}
).set_index("Selección")
st.bar_chart(df)
