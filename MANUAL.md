# 📖 Manual de Usuario — Predictor Mundial FIFA 2026

Pipeline de Machine Learning para predecir partidos y simular el Mundial 2026
completo (48 equipos, 12 grupos) mediante Monte Carlo.

---

## 1. ¿Qué hace esta aplicación?

| Pregunta que responde | Dónde |
|---|---|
| ¿Qué tan fuerte es cada selección? | Página **📊 Estadísticas** |
| Si juegan X vs Y, ¿quién gana y con qué marcador? | Página **⚔️ Simular Partido** |
| ¿Quién tiene más chance de ser campeón? | Página **🏆 Simular Torneo** |
| ¿Y si registro lo que ya pasó en el Mundial? | Página **📝 Ingresar Resultados** |

Bajo el capó hay **dos modelos** entrenados con ~47,000 partidos internacionales
reales (1872–2024, datos de Kaggle):
- **Poisson Dixon-Coles** → predice el *marcador exacto* (3-1, 0-0, etc.).
- **LightGBM / Gradient Boosting** → predice *probabilidades* (gana/empata/pierde).

La predicción final que ves es una **mezcla 50/50** de ambos.

---

## 2. Requisitos previos

- **Python 3.10 o superior** instalado.
- Conexión a internet (solo la primera vez, para descargar los datos).
- macOS/Linux/Windows con terminal.

> 💡 En Mac, si usas LightGBM puede pedir `libomp`. Si no lo tienes, el sistema
> usa automáticamente un modelo equivalente de scikit-learn. No tienes que hacer
> nada: funciona igual.

---

## 3. Instalación (una sola vez)

Abre una terminal y ejecuta:

```bash
cd ~/Claude/worldcup2026-ml      # entra al proyecto
pip install -e ".[dev]"          # instala todo lo necesario
```

Esto instala pandas, scikit-learn, LightGBM, Streamlit, Plotly, etc.

---

## 4. Entrenar los modelos (una sola vez)

```bash
python -m src.models.train
```

Qué ocurre (verás mensajes en pantalla):
1. **Descarga** el dataset de Kaggle (la primera vez) y lo guarda en `data/raw/`.
2. **Construye las features** de forma causal (Elo, forma, pedigrí…).
3. **Valida** ambos modelos con validación temporal y muestra métricas.
4. **Guarda** los modelos entrenados en `models/`.

Tarda ~1–2 minutos. Al terminar verás algo como:

```
RESUMEN | LightGBM logloss=0.9214 acc=0.576 | Poisson logloss=0.9107
```

> ✅ Solo necesitas hacer esto una vez. Los modelos quedan guardados en disco.

---

## 5. Abrir la aplicación visual

```bash
streamlit run app/Home.py
```

Se abrirá tu navegador en `http://localhost:8501`. Si no se abre solo, copia esa
dirección en el navegador.

Para **cerrar** la app: vuelve a la terminal y pulsa `Ctrl + C`.

---

## 6. Guía de cada pantalla

### 🏠 Inicio (Home)
Resumen general: número de selecciones, equipo #1 del ranking Elo y un gráfico
con el **Top 20** de selecciones más fuertes.

### 📊 Estadísticas
1. Elige una selección en el menú desplegable.
2. Verás sus tarjetas: **Rating Elo**, puntos por partido recientes, racha,
   partidos y victorias en Mundiales, promedio de goles a favor/contra.
3. Abajo, un gráfico comparativo del Elo (ajusta cuántas selecciones mostrar
   con el deslizador).

**Cómo leerlo:** Elo más alto = mejor equipo. Referencias: ~2200 (Argentina,
élite), ~2000 (potencias), ~1700 (selección media), ~1500 (debutante).

### ⚔️ Simular Partido
1. Elige **Equipo A** y **Equipo B**.
2. (Opcional) Marca/desmarca **"Cancha neutral"** — desmárcalo para dar ventaja
   de local al Equipo A.
3. Verás:
   - Un gráfico de barras con **% de victoria / empate / derrota**.
   - Los **10 marcadores más probables** (ej. "2-1 → 9.4%").
4. Ajusta el número de simulaciones con el deslizador (más = más preciso, más
   lento).

### 🏆 Simular Torneo
1. (Opcional) Despliega **"Ver composición de los 12 grupos"** para revisar el
   sorteo oficial 2026.
2. Elige el **número de simulaciones** (500 a 10,000). Más simulaciones = números
   más estables, pero tarda más. **2,500 es un buen punto de partida.**
3. (Opcional) Cambia la **semilla** para obtener una corrida distinta
   (la misma semilla siempre da el mismo resultado = reproducible).
4. Pulsa **🚀 Ejecutar simulación**.
5. Resultados:
   - Gráfico con la **probabilidad de ser campeón** (Top 15).
   - Tabla completa con la probabilidad de cada selección de llegar a
     dieciseisavos, octavos, cuartos, semis, final y título.

**Tiempo aproximado:** 10,000 simulaciones tardan ~1 minuto.

### 📝 Ingresar Resultados (aprendizaje online)
Registra los partidos del Mundial **que ya se jugaron** para que el modelo
aprenda de la realidad y afine sus predicciones.

1. En **"Registrar un partido"** elige Equipo A (local) y Equipo B (visitante).
2. Escribe los **goles** de cada uno y la **fecha**.
3. Elige la **fase** (Grupos, Octavos, ...) — es informativa.
4. **Cancha neutral:** déjalo marcado salvo que A juegue de local de verdad
   (México, EE.UU. o Canadá en su país). Esto importa para el modelo.
5. Pulsa **💾 Guardar resultado**.

Qué pasa al guardar:
- El Elo, la forma reciente y el pedigrí de ambos equipos se actualizan.
- **Todas** las demás páginas (Estadísticas, Simular Partido, Simular Torneo)
  usan de inmediato el estado actualizado. No hay que reentrenar nada.

Gestión del historial:
- Abajo ves la tabla **"Resultados ya registrados"**.
- Puedes **eliminar** una fila por su número, o **borrar todo** el historial en
  vivo para volver al estado base.
- El panel **"Impacto en el modelo"** muestra cuánto subió/bajó el Elo de cada
  selección por los resultados que ingresaste.

> 🔒 **Tu historial nunca corrompe el modelo.** El estado se reconstruye siempre
> como *histórico entrenado + resultados ingresados*. Borrar una fila revierte
> su efecto exactamente, sin residuos.


---

## 7. Preguntas frecuentes

**¿Por qué nadie tiene más de ~35% de ser campeón?**
Porque el fútbol es impredecible. Incluso el mejor equipo puede perder un partido
único. La incertidumbre es real y el modelo la respeta.

**Cambié de opinión sobre los grupos, ¿puedo editarlos?**
Sí. Edita el diccionario `GROUPS_2026` en `src/config.py` y guarda. No necesitas
reentrenar; solo recarga la página del torneo.

**¿Los datos están actualizados?**
El histórico llega hasta julio 2024. Los grupos son el **sorteo oficial** del
5 de diciembre de 2025. Para incorporar partidos más recientes, reemplaza
`data/raw/results.csv` y reentrena.

**Salió un aviso de "modelos no entrenados".**
Ejecuta el paso 4 (`python -m src.models.train`) y recarga.

**¿`python` no se reconoce?**
Prueba con `python3` en todos los comandos.

---

## 8. Resumen de comandos

```bash
# Instalar (una vez)
pip install -e ".[dev]"

# Entrenar (una vez)
python -m src.models.train

# Ejecutar pruebas (opcional)
python -m pytest -q

# Lanzar la app
streamlit run app/Home.py
```

También disponibles como atajos: `make install`, `make train`, `make test`,
`make app`.

---

## 9. Glosario rápido

| Término | Significado |
|---|---|
| **Elo** | Puntuación de fuerza de cada selección, sube/baja según resultados. |
| **xG / forma** | Rendimiento reciente (últimos 5 partidos). |
| **Pedigrí** | Experiencia histórica en Copas del Mundo. |
| **Monte Carlo** | Simular el torneo miles de veces para estimar probabilidades. |
| **Dixon-Coles** | Modelo que predice marcadores realistas. |
| **Log Loss / Brier** | Métricas de qué tan bien calibradas están las probabilidades (más bajo = mejor). |
| **Semilla (seed)** | Número que hace reproducible una simulación aleatoria. |
