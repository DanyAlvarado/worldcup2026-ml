"""Página de aprendizaje online: ingresar resultados reales del Mundial 2026.

El usuario registra los partidos que ya ocurrieron. Cada resultado se anexa a un
log y el estado de los equipos (Elo, forma, pedigrí) se reconstruye como
base histórica + replay del log. Las predicciones de las demás páginas mejoran
automáticamente, sin reentrenar los modelos.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date

import pandas as pd
import streamlit as st

from _shared import current_state, list_teams, load_models, require_models
from src.config import GROUPS_2026
from src.live.updater import (
    MatchResultInput,
    append_result,
    delete_result,
    elo_deltas,
    load_base_state,
    load_live_results,
    validate_result,
)

st.set_page_config(page_title="Ingresar Resultados", page_icon="📝", layout="wide")
st.title("📝 Ingresar resultados del Mundial (aprendizaje online)")

if not require_models():
    st.stop()

st.markdown(
    """
Registra aquí los partidos **que ya se jugaron**. El modelo aprende de cada
resultado: actualiza el **Elo**, la **forma reciente** y el **pedigrí** de las
selecciones, y todas las predicciones (estadísticas, partidos y simulación del
torneo) se recalculan al instante.

> 🔒 El conocimiento histórico nunca se pierde: el estado se reconstruye como
> *histórico + resultados ingresados*. Puedes borrar cualquier fila y todo
> vuelve a su sitio.
"""
)

# Selecciones del Mundial primero (las 48), luego el resto del mundo.
wc_teams = sorted({t for teams in GROUPS_2026.values() for t in teams})
all_teams = list_teams()
ordered_teams = wc_teams + [t for t in all_teams if t not in wc_teams]
known = set(all_teams)

STAGES = [
    "Group", "Round of 32", "Round of 16",
    "Quarter-final", "Semi-final", "Third place", "Final",
]

# --------------------------------------------------------------------------- #
# Formulario de ingreso
# --------------------------------------------------------------------------- #
st.subheader("➕ Registrar un partido")

with st.form("nuevo_resultado", clear_on_submit=True):
    c1, c2, c3 = st.columns([2, 1, 2])
    home = c1.selectbox("Equipo local (A)", ordered_teams, index=0)
    c2.markdown(
        "<div style='text-align:center;padding-top:34px;font-weight:600'>VS</div>",
        unsafe_allow_html=True,
    )
    away = c3.selectbox("Equipo visitante (B)", ordered_teams, index=1)

    c4, c5, c6, c7 = st.columns(4)
    home_score = c4.number_input("Goles A", min_value=0, max_value=30, value=0, step=1)
    away_score = c5.number_input("Goles B", min_value=0, max_value=30, value=0, step=1)
    match_date = c6.date_input("Fecha", value=date(2026, 6, 11))
    stage = c7.selectbox("Fase", STAGES, index=0)

    c8, c9 = st.columns(2)
    neutral = c8.checkbox(
        "Cancha neutral (sin ventaja de localía)", value=True,
        help="Desmárcalo solo si 'Equipo A' juega realmente como local "
             "(p. ej. México, EE.UU. o Canadá en su país).",
    )
    tournament = c9.text_input("Competición", value="FIFA World Cup")

    submitted = st.form_submit_button("💾 Guardar resultado", type="primary")

if submitted:
    errors = validate_result(home, away, home_score, away_score, known)
    if errors:
        for e in errors:
            st.error(e)
    else:
        result = MatchResultInput(
            date=str(match_date),
            home_team=home,
            away_team=away,
            home_score=int(home_score),
            away_score=int(away_score),
            tournament=tournament,
            neutral=bool(neutral),
            stage=stage,
        )
        append_result(result)
        st.cache_resource.clear()  # invalida estado vivo cacheado
        st.success(
            f"✅ Guardado: {home} {int(home_score)}–{int(away_score)} {away}. "
            "El modelo ya aprendió de este resultado."
        )
        st.rerun()

# --------------------------------------------------------------------------- #
# Log de resultados ingresados + gestión
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("📒 Resultados ya registrados")

live_df = load_live_results()

if live_df.empty:
    st.info("Aún no has ingresado ningún resultado. Registra el primero arriba.")
    st.stop()

view = live_df.copy()
view.index = range(1, len(view) + 1)
st.dataframe(
    view.rename(columns={
        "date": "Fecha", "home_team": "Local", "away_team": "Visitante",
        "home_score": "GL", "away_score": "GV", "tournament": "Competición",
        "neutral": "Neutral", "stage": "Fase",
    }),
    use_container_width=True,
)

cdel, cclear = st.columns([2, 1])
with cdel:
    row_to_delete = st.number_input(
        "Nº de fila a eliminar", min_value=1, max_value=len(live_df), value=1, step=1
    )
    if st.button("🗑️ Eliminar fila"):
        delete_result(int(row_to_delete) - 1)
        st.cache_resource.clear()
        st.success(f"Fila {int(row_to_delete)} eliminada. Estado recalculado.")
        st.rerun()

with cclear:
    if st.button("⚠️ Borrar TODO el historial en vivo"):
        from src.config import LIVE_RESULTS_CSV

        LIVE_RESULTS_CSV.unlink(missing_ok=True)
        st.cache_resource.clear()
        st.success("Historial en vivo borrado. Vuelta al estado base.")
        st.rerun()

# --------------------------------------------------------------------------- #
# Impacto en el modelo: cambios de Elo
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("📈 Impacto en el modelo (cambio de Elo por los resultados)")

base_state = load_base_state()
live_state = current_state()
deltas = elo_deltas(base_state, live_state)

if not deltas:
    st.info("Sin cambios de Elo todavía.")
else:
    rows = []
    for team, d in sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True):
        rows.append({
            "Selección": team,
            "Elo antes": round(base_state.elo.get(team), 1),
            "Elo ahora": round(live_state.elo.get(team), 1),
            "Δ Elo": round(d, 1),
        })
    impact = pd.DataFrame(rows)
    st.dataframe(impact, use_container_width=True, height=min(420, 60 + 35 * len(rows)))
    st.bar_chart(impact.set_index("Selección")["Δ Elo"])
