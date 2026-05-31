"""Página de simulación Monte Carlo del torneo completo."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import streamlit as st

from _shared import make_simulator, require_models
from src.config import GROUPS_2026

st.set_page_config(page_title="Simular Torneo", page_icon="🏆", layout="wide")
st.title("🏆 Simular el Mundial 2026 (Monte Carlo)")

if not require_models():
    st.stop()

with st.expander("Ver composición de los 12 grupos"):
    cols = st.columns(4)
    for i, (name, teams) in enumerate(GROUPS_2026.items()):
        cols[i % 4].markdown(
            f"**Grupo {name}**\n\n" + "\n".join(f"- {t}" for t in teams)
        )

n_sims = st.select_slider(
    "Número de simulaciones Monte Carlo",
    options=[500, 1000, 2500, 5000, 10000],
    value=2500,
)
seed = st.number_input("Semilla (reproducibilidad)", value=42, step=1)

if st.button("🚀 Ejecutar simulación", type="primary"):
    sim = make_simulator(seed=int(seed))
    with st.spinner(f"Simulando {n_sims:,} torneos..."):
        results = sim.run(n_sims=int(n_sims))
    df = results.to_dataframe()

    st.success(f"Listo: {n_sims:,} torneos simulados.")

    st.subheader("🥇 Probabilidad de ser campeón (Top 15)")
    top = df.head(15)
    fig = px.bar(
        top, x="P(Campeón)", y="team", orientation="h",
        text=top["P(Campeón)"].map(lambda v: f"{v:.1%}"),
        color="P(Campeón)", color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=520,
                      xaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Probabilidades por ronda (todas las selecciones)")
    pct_cols = [c for c in df.columns if c.startswith("P(")]
    styled = df.copy()
    for c in pct_cols:
        styled[c] = (styled[c] * 100).round(1).astype(str) + "%"
    st.dataframe(styled, use_container_width=True, height=500)
else:
    st.info("Configura los parámetros y pulsa **Ejecutar simulación**.")
