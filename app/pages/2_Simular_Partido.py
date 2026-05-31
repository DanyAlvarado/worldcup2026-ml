"""Página interactiva head-to-head: probabilidades 1X2 y marcadores simulados."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import list_teams, make_engine, require_models

st.set_page_config(page_title="Simular Partido", page_icon="⚔️", layout="wide")
st.title("⚔️ Simular un partido")

if not require_models():
    st.stop()

teams = list_teams()
col_a, col_b = st.columns(2)
home = col_a.selectbox("Equipo A", teams, index=0)
away = col_b.selectbox("Equipo B", teams, index=1)
neutral = st.checkbox("Cancha neutral", value=True)

if home == away:
    st.warning("Elige dos selecciones distintas.")
    st.stop()

engine = make_engine()
probs = engine.win_probabilities(home, away, neutral=neutral)

st.subheader("Probabilidades 1X2 (blend Poisson + LightGBM)")
fig = go.Figure(go.Bar(
    x=[f"Gana {home}", "Empate", f"Gana {away}"],
    y=[probs["home_win"], probs["draw"], probs["away_win"]],
    marker_color=["#1f77b4", "#7f7f7f", "#d62728"],
    text=[f"{v:.1%}" for v in
          (probs["home_win"], probs["draw"], probs["away_win"])],
    textposition="auto",
))
fig.update_layout(yaxis_tickformat=".0%", height=380)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("🎲 Marcadores más probables")

n_sim = st.slider("Número de simulaciones", 1000, 20000, 5000, step=1000)
scores: Counter = Counter()
for _ in range(n_sim):
    scores[engine.sample_score(home, away, neutral=neutral)] += 1

top = scores.most_common(10)
df = pd.DataFrame(
    {
        "Marcador": [f"{h}-{a}" for (h, a), _ in top],
        "Probabilidad": [c / n_sim for _, c in top],
    }
).set_index("Marcador")
st.bar_chart(df)
st.dataframe(
    df.assign(
        Probabilidad=lambda d: (d["Probabilidad"] * 100).round(1).astype(str) + "%"
    ),
    use_container_width=True,
)
