# ICFES Analytics – Sprint 2

Clustering y series de tiempo para datos ICFES (Saber 11 / Saber Pro). Incluye:
- Clusterización (particional, densidad, jerárquica) con seis gráficos compactos.
- Serie de tiempo: agregación, diagnóstico (STL + ADF), ARIMA y RNN mínima (NumPy, sin TensorFlow).
- Módulos en `src/` reutilizables desde notebooks o scripts.

## Datos (datasets grandes y Git)
- Los CSV del Sprint 1 pueden superar el límite de 100 MB de GitHub; por eso NO se incluyen en el repo.
- Coloca los archivos CSV (por ejemplo `Dataset1–Saber11(2020-2)_LIMPIO.csv` y `Dataset2–SaberPro(2021–2024)_LIMPIO.csv`) en la carpeta del proyecto (o en `data/`).
- Este repo ignora `*.csv` y `data/` por defecto (ver `.gitignore`).
- El notebook detecta automáticamente los archivos limpios por patrón: `Dataset1*LIMPIO.csv` y `Dataset2*LIMPIO.csv` en el directorio actual.

## Requisitos
- Python 3.10–3.12 (Windows recomendado).
- pip y venv (incluido con Python) para gestionar dependencias.

## Setup del proyecto con venv + pip (Windows)
1. Abre PowerShell en la carpeta del proyecto.
2. Crea y activa un entorno virtual:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
3. Instala las dependencias:
  ```powershell
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
4. Registra el kernel de Jupyter:
  ```powershell
  python -m ipykernel install --user --name icfes-analytics
  ```

## Estructura
```
Proyecto/
├─ src/
│  └─ icfes_analytics/
│     ├─ __init__.py
│     ├─ clustering.py        # 6 métodos y gráficos
│     ├─ timeseries.py        # parser, agregación, STL/ADF, ARIMA
│     ├─ rnn_numpy.py         # RNN mínima (NumPy)
│     └─ plots.py             # estilo BTC
├─ Sprint2_ICFES.ipynb        # notebook principal (ejecutado)
├─ requirements.txt           # dependencias (pip)
└─ README.md
```

## Uso de los módulos
Importa desde `icfes_analytics`:
```python
from icfes_analytics import (
    apply_btc_style,
    parse_periodo_flexible, aggregate_series,
    fit_arima_small_grid, plot_arima_forecast,
    forecast_one_step_numpy,
)
```

### Clustering
```python
from icfes_analytics.clustering import run_six_clustering_plots
# X_scaled: ndarray con features estandarizadas (StandardScaler)
resumen = run_six_clustering_plots(X_scaled, n_clusters=5)
print(resumen)
```

### Series de tiempo
```python
# df: DataFrame con columnas 'periodo' y 'punt_global'
agg = aggregate_series(df, period_col='periodo', value_col='punt_global')
apply_btc_style()
# Plot simple
agg.plot(figsize=(12,4))
```

#### ARIMA
```python
order_sel, res_sel, y_pred, y_true, train, test, metrics = fit_arima_small_grid(agg)
print("ARIMA", order_sel, metrics)
plot_arima_forecast(train, test, y_pred, order_sel)
```

#### RNN (NumPy)
```python
y_pred, y_true, metrics = forecast_one_step_numpy(agg, freq='QS-MAR', window=4, hidden_size=16, epochs=600)
print(metrics)
```

## Ejecutar el notebook
1. Abre VS Code y el notebook `Sprint2_ICFES.ipynb`.
2. Selecciona el kernel `icfes-analytics` (si lo registraste) o el intérprete del venv `.venv`.
3. Ejecuta las celdas en orden. El notebook ya usa los módulos de `src/`.

### Nota sobre Git LFS (opcional)
Si necesitas versionar datasets grandes, considera usar [Git LFS](https://git-lfs.github.com/). Para este proyecto académico recomendamos mantener los CSV fuera del repo y documentar su ubicación local.