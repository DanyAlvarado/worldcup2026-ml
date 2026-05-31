# 🏆 Mundial FIFA 2026 — Pipeline ML End-to-End

Predicción de partidos y simulación Monte Carlo del Mundial 2026 (48 equipos,
12 grupos) entrenado con el histórico de partidos internacionales (Kaggle).

## Arquitectura

```
src/
├── config.py            # Constantes, sedes, grupos, hiperparámetros
├── data/loader.py       # Descarga Kaggle + validación de esquema
├── features/            # Elo, forma, pedigrí, entorno + builder CAUSAL
├── models/              # Poisson Dixon-Coles, LightGBM, validación temporal
├── simulation/          # Fase de grupos, knockout, Monte Carlo
└── utils/               # Geo (Haversine), logging
app/                     # Dashboard Streamlit (3 páginas)
tests/                   # pytest (incluye test anti-leakage)
```

### Modelos

| Modelo | Salida | Rol |
|---|---|---|
| **Poisson Dixon-Coles** | Marcador exacto | Muestreo de goles para la simulación |
| **LightGBM** | P(1X2) | Probabilidades calibradas |
| **Blend 50/50** | P(1X2) | Predicción mostrada en la UI |

### Anti data-leakage

El `FeatureBuilder` recorre los partidos **en orden cronológico**: lee las
features pre-partido (Elo, forma, pedigrí) *antes* de actualizar el estado con
el resultado. La validación usa `TimeSeriesSplit`. Ver `tests/test_no_leakage.py`.

## Uso

```bash
# 1. Instalar dependencias
pip install -e ".[dev]"

# 2. Entrenar (descarga datos de Kaggle, valida y persiste modelos)
python -m src.models.train

# 3. Tests
pytest -q

# 4. Lanzar el dashboard
streamlit run app/Home.py
```

### Métricas de evaluación
Log Loss, Brier Score (calibración multiclase) y accuracy, reportadas por fold
temporal durante el entrenamiento.

## Datos
- `muhammadehsan02/global-football-results-18722024` (resultados 1872–2024).
- El dataset nombra los archivos `Match_Results.csv` / `Penalty_Shootouts.csv`;
  el loader los normaliza a `results.csv` / `shootouts.csv`.
- Para xG/posesión reales se puede integrar `excel4soccer/espn-soccer-data`
  ampliando un módulo `src/features/match_stats.py`.
