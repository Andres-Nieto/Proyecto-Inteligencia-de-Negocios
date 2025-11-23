# 🎓 **ICFES Analytics – Proyecto de Inteligencia de Negocios**
Sistema completo de análisis, clusterización, predicción y recomendaciones académicas basado en datos de **ICFES Saber 11** y **Saber Pro**, implementado en Python mediante un pipeline profesional de *Machine Learning + API + Dashboard*.

## 👥 Autores
- Oscar Daniel Casallas Lozano – 2220221011
- Andres Fernando Nieto Rodríguez - 2220231083
- David Santiago Manchola Serna - 2220221093

---

# 📑 **Tabla de Contenidos**
- [🎓 **ICFES Analytics – Proyecto de Inteligencia de Negocios**](#-icfes-analytics--proyecto-de-inteligencia-de-negocios)
  - [👥 Autores](#-autores)
- [📑 **Tabla de Contenidos**](#-tabla-de-contenidos)
- [🎯 **Descripción del Proyecto**](#-descripción-del-proyecto)
- [🧪 Sprint 1: Busca y limpieza de datasets](#-sprint-1-busca-y-limpieza-de-datasets)
- [📂 **Datasets**](#-datasets)
    - [📌 **Fuente**](#-fuente)
- [🏗️ **Arquitectura del Proyecto**](#️-arquitectura-del-proyecto)
- [🧰 **Requisitos Previos**](#-requisitos-previos)
    - [✔️ Software necesario](#️-software-necesario)
- [⚙️ Instalación y Configuración](#️-instalación-y-configuración)
  - [1. Clonar el repositorio](#1-clonar-el-repositorio)
  - [2. Descarga de datasets](#2-descarga-de-datasets)
    - [📁 Ubicación esperada](#-ubicación-esperada)
  - [3. Crear Entorno Virtual](#3-crear-entorno-virtual)
  - [4. Instalar Dependencias](#4-instalar-dependencias)
  - [5. Configurar Kernel de Jupyter (para Sprint 2)](#5-configurar-kernel-de-jupyter-para-sprint-2)
  - [6. Descargar y Ubicar Datasets](#6-descargar-y-ubicar-datasets)
  - [7. Entrenar Modelos (obligatorio antes de Sprint 3 y 4)](#7-entrenar-modelos-obligatorio-antes-de-sprint-3-y-4)
  - [8. Iniciar el Dashboard](#8-iniciar-el-dashboard)
- [🧪 Sprint 2: Análisis y Clusterización](#-sprint-2-análisis-y-clusterización)
  - [🎯 Objetivos](#-objetivos)
  - [🛠️ Tecnologías Utilizadas](#️-tecnologías-utilizadas)
  - [📊 Ejecución del Notebook](#-ejecución-del-notebook)
  - [📈 Módulos Disponibles](#-módulos-disponibles)
    - [🔹 Clustering](#-clustering)
    - [🔹 Series Temporales](#-series-temporales)
    - [🔹 RNN desde Cero en NumPy](#-rnn-desde-cero-en-numpy)
  - [📊 Resultados Clave del Sprint 2](#-resultados-clave-del-sprint-2)
- [🌐 Sprint 3: API de Recomendaciones](#-sprint-3-api-de-recomendaciones)
  - [🎯 Objetivos](#-objetivos-1)
  - [🛠️ Tecnologías Utilizadas](#️-tecnologías-utilizadas-1)
  - [🚀 Iniciar la API](#-iniciar-la-api)
  - [📚 Documentación Interactiva](#-documentación-interactiva)
  - [🔌 Endpoints Disponibles](#-endpoints-disponibles)
  - [🧠 Lógica de Recomendaciones](#-lógica-de-recomendaciones)
  - [📊 Resultados Clave del Sprint 3](#-resultados-clave-del-sprint-3)
- [📊 Sprint 4: Dashboard Interactivo](#-sprint-4-dashboard-interactivo)
  - [🎯 Objetivos](#-objetivos-2)
  - [🛠️ Tecnologías Utilizadas](#️-tecnologías-utilizadas-2)
  - [🎨 Funciones del Dashboard](#-funciones-del-dashboard)
    - [1️⃣ Predicción Individual](#1️⃣-predicción-individual)
    - [2️⃣ Gráfico Radar Comparativo](#2️⃣-gráfico-radar-comparativo)
    - [3️⃣ Estadísticas Globales](#3️⃣-estadísticas-globales)
    - [4️⃣ Historial Completo](#4️⃣-historial-completo)
    - [5️⃣ Búsqueda por Estudiante](#5️⃣-búsqueda-por-estudiante)
    - [6️⃣ Limpieza de Historial](#6️⃣-limpieza-de-historial)
    - [7️⃣ Estado de la API](#7️⃣-estado-de-la-api)
- [🔄 Flujo de Trabajo Completo](#-flujo-de-trabajo-completo)
- [📈 Resultados y Conclusiones](#-resultados-y-conclusiones)
  - [🔹 Resultados Técnicos](#-resultados-técnicos)
  - [🔹 Conclusiones Académicas](#-conclusiones-académicas)

---

# 🎯 **Descripción del Proyecto**
Este proyecto implementa un sistema integral capaz de:

- 🔍 Analizar datos educativos del ICFES  
- 🧠 Aplicar técnicas de **clustering** para descubrir perfiles estudiantiles  
- 📈 Modelar series temporales (ARIMA y RNN)  
- 🤖 Generar predicciones automáticas  
- 🎓 Recomendar carreras y áreas de refuerzo  
- 📊 Visualizar resultados mediante un **dashboard interactivo**  

Todo organizado en 4 sprints:  
| Sprint | Objetivo | Tecnologías |
|--------|----------|-------------|
| **Sprint 1** | Busca y limpieza de datasets | pandas, numpy, matplotlib, seaborn |
| **Sprint 2** | Análisis y clustering | Scikit-learn, Statsmodels, NumPy |
| **Sprint 3** | API REST para predicciones | FastAPI, Uvicorn, Pickle |
| **Sprint 4** | Dashboard interactivo | Streamlit, Plotly |

---
# 🧪 Sprint 1: Busca y limpieza de datasets
# 📂 **Datasets**
### 📌 **Fuente**
Datos reales del ICFES:
- **Saber 11 – 2020-2** *https://www.datos.gov.co/Educaci-n/PUNTAJEGLOBALPROMEDIO-SABERPRO-11-2020-2-PORDE/tagd-tfeb*
  • Dataset 1 – Saber 11 (2020-2): contiene los resultados de los exámenes aplicados
a estudiantes de educación media (colegios), lo que representa el punto de entrada
al sistema de educación superior.

- **Saber Pro – 2021 a 2024** *https://www.kaggle.com/datasets/williamrrubio/data-icfes*
  • Dataset 2 – Saber Pro (2021–2024): reúne los resultados de los exámenes
aplicados a estudiantes universitarios próximos a graduarse, lo que refleja el punto
de salida del proceso formativo.

---

# 🏗️ **Arquitectura del Proyecto**
```
Proyecto/
│
├── src/icfes_analytics/          # 📦 Módulos analíticos (Sprint 2)
│   ├── clustering.py             # Algoritmos de clustering
│   ├── timeseries.py             # Análisis de series temporales
│   ├── rnn_numpy.py              # RNN implementada en NumPy puro
│   ├── plots.py                  # Utilidades de visualización
│   ├── forecasting.py
│   └── __init__.py
│
├── api/app/                      # 🌐 API REST (Sprint 3)
│   ├── main.py                   # Endpoints de FastAPI
│   ├── services.py               # Lógica de negocio
│   ├── schemas.py                # Modelos Pydantic
│   └── models_loader.py          # Carga de modelos ML
│
├── dashboard/                    # 📊 Dashboard (Sprint 4)
│   └── app.py                    # Aplicación Streamlit
│
├── models/                       # 🤖 Modelos entrenados
│   ├── scaler.pkl                # StandardScaler ajustado
│   ├── kmeans.pkl                # Modelo K-Means
│   └── feature_cols.pkl          # Lista de features
│
├── data/                         # 📁 Datasets (no incluidos en el repo)
│   ├── Dataset1–Saber11(2020-2)_LIMPIO.csv
│   └── Dataset2–SaberPro(2021–2024)_LIMPIO.csv
│
├── Sprint2_ICFES.ipynb           # 📓 Notebook principal (Sprint 2)
├── train_save_models.py          # 🎓 Script de entrenamiento
├── requirements.txt              # 📋 Dependencias unificadas
└── README.md                     # 📖 Este archivo
```


---

# 🧰 **Requisitos Previos**
### ✔️ Software necesario
- Python **3.10 – 3.12**
- pip actualizado
- Git
- Windows / Linux / macOS


# ⚙️ Instalación y Configuración

## 1. Clonar el repositorio
```
git clone https://github.com/Andres-Nieto/Sprint2-Proyecto-Inteligencia-de-Negocios.git
cd Sprint2-Proyecto-Inteligencia-de-Negocios
```
## 2. Descarga de datasets
🔗 *Enlace a Google Drive (datasets limpios)*  
*(https://drive.google.com/drive/folders/1O49JVxhRDbB1oaLek9JYvX1UWl59MEmo)*

### 📁 Ubicación esperada
```
data/
├── Dataset1–Saber11(2020-2)_LIMPIO.csv
└── Dataset2–SaberPro(2021–2024)_LIMPIO.csv
```
⚠️ Nota importante: Los datasets NO están incluidos en el repositorio debido a su tamaño (>100 MB). El archivo .gitignore excluye automáticamente *.csv y la carpeta data/.

## 3. Crear Entorno Virtual
Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Instalar Dependencias
```
pip install -r requirements.txt
```

## 5. Configurar Kernel de Jupyter (para Sprint 2)
```
python -m ipykernel install --user --name icfes-analytics
```

## 6. Descargar y Ubicar Datasets
- Descarga los CSV desde Google Drive
- Crea la carpeta data/ en la raíz del proyecto
- Copia los archivos CSV dentro de data/

## 7. Entrenar Modelos (obligatorio antes de Sprint 3 y 4)
```
python train_save_models.py
```
Esto generará los archivos en models/:
- scaler.pkl - Normalizador de features
- kmeans.pkl - Modelo de clustering
- feature_cols.pkl - Lista de columnas utilizadas

## 8. Iniciar el Dashboard
```bash
# Encender la API
uvicorn api.app.main:app --reload

# Encender el dashboard
streamlit run dashboard/app.py
```

Disponible en: http://localhost:8501

---

# 🧪 Sprint 2: Análisis y Clusterización
## 🎯 Objetivos
- Aplicar técnicas de clustering (particional, jerárquico y por densidad).
- Analizar series temporales de puntajes.
- Implementar modelos de pronóstico: ARIMA y RNN desde cero con NumPy.

## 🛠️ Tecnologías Utilizadas
- **Clustering:** K-Means, DBSCAN, Hierarchical Clustering (Scikit-learn)
- **Series temporales:** STL Decomposition, Test ADF, ARIMA (Statsmodels)
- **Deep learning:** RNN implementada manualmente con NumPy
- **Visualización:** Matplotlib, Seaborn

## 📊 Ejecución del Notebook
1. Abrir VS Code en la carpeta del proyecto.
2. Abrir el archivo: `Sprint2_ICFES.ipynb`
3. Seleccionar kernel: **icfes-analytics**
4. Ejecutar todas las celdas en orden.

## 📈 Módulos Disponibles

### 🔹 Clustering
```python
from icfes_analytics.clustering import run_six_clustering_plots

X_scaled  # array con features estandarizadas

resumen = run_six_clustering_plots(X_scaled, n_clusters=5)
print(resumen)
```

Genera automáticamente los siguientes gráficos:
- K-Means  
- DBSCAN  
- Hierarchical Clustering (Ward)  
- Silhouette Analysis  
- Elbow Method  
- Dendrograma  

---

### 🔹 Series Temporales
```python
from icfes_analytics.timeseries import (
    aggregate_series,
    fit_arima_small_grid,
    plot_arima_forecast
)

# Agregar datos por periodo
agg = aggregate_series(df, period_col='periodo', value_col='punt_global')

# Ajustar modelo ARIMA
order, result, y_pred, y_true, train, test, metrics = fit_arima_small_grid(agg)
print(f"Mejor modelo ARIMA{order}: RMSE={metrics['rmse']:.2f}")

# Visualizar pronóstico
plot_arima_forecast(train, test, y_pred, order)
```

---

### 🔹 RNN desde Cero en NumPy
```python
from icfes_analytics.rnn_numpy import forecast_one_step_numpy

y_pred, y_true, metrics = forecast_one_step_numpy(
    agg,
    freq='QS-MAR',
    window=4,
    hidden_size=16,
    epochs=600
)
print(f"RNN Metrics: {metrics}")
```

## 📊 Resultados Clave del Sprint 2
- Identificación de **5 perfiles académicos** mediante clustering.
- ARIMA alcanzó un **RMSE ≈ 15 puntos**.
- RNN logró capturar **patrones no lineales**.
- Se generaron **6 gráficos automáticos** de análisis.

---

# 🌐 Sprint 3: API de Recomendaciones

## 🎯 Objetivos
- Construir API REST para predicciones en tiempo real.  
- Crear sistema de recomendaciones académicas.  
- Mantener historial de consultas en memoria.

## 🛠️ Tecnologías Utilizadas
- FastAPI  
- Uvicorn  
- Pydantic  
- Pickle  

## 🚀 Iniciar la API
```bash
uvicorn api.app.main:app --reload
```

URL base:  
http://127.0.0.1:8000

## 📚 Documentación Interactiva
- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`

## 🔌 Endpoints Disponibles
| Endpoint | Método | Descripción |
|---------|--------|-------------|
| `/health` | GET | Estado de la API |
| `/predict` | POST | Predice clúster y recomendaciones |
| `/cluster` | POST | Igual que predict + guarda historial |
| `/history` | GET | Últimas 50 consultas |
| `/summary` | GET | Resumen por clúster |
| `/student/{id}` | GET | Historial por estudiante |
| `/clear-history` | DELETE | Limpia memoria |
| `/forecast` | POST | Prediccion | 

## 🧠 Lógica de Recomendaciones
Basada en:  
- Clúster asignado  
- Fortaleza principal  
- Carreras sugeridas según perfil  
- Áreas de refuerzo  

## 📊 Resultados Clave del Sprint 3
- API con **7 endpoints funcionales**
- Tiempo de respuesta **< 50 ms**
- Documentación automática
- Arquitectura lista para despliegue en la nube

---

# 📊 Sprint 4: Dashboard Interactivo

## 🎯 Objetivos
- Crear interfaz visual para usuarios finales  
- Integrar API del Sprint 3  
- Mostrar estadísticas y comparativas en tiempo real  

## 🛠️ Tecnologías Utilizadas
- Streamlit  
- Plotly  
- Requests  

## 🎨 Funciones del Dashboard

### 1️⃣ Predicción Individual
Muestra:
- Clúster asignado  
- Fortaleza principal  
- Carreras sugeridas  
- Áreas de refuerzo  

### 2️⃣ Gráfico Radar Comparativo
Compara:
- Estudiante (azul)
- Promedio nacional (rojo)

### 3️⃣ Estadísticas Globales
- Promedios por área  
- Total de consultas  
- Filtros por rango de puntaje  

### 4️⃣ Historial Completo  
- 50 últimas consultas  

### 5️⃣ Búsqueda por Estudiante  

### 6️⃣ Limpieza de Historial

### 7️⃣ Estado de la API  

---

# 🔄 Flujo de Trabajo Completo
```
SPRINT 2 → Clustering + Series Temporales
        ↓
train_save_models.py → Entrena y guarda modelos
        ↓
SPRINT 3 → API con 8 endpoints
        ↓
SPRINT 4 → Dashboard conectado a la API
```

---

# 📈 Resultados y Conclusiones

## 🔹 Resultados Técnicos
| Métrica | Valor |
|--------|-------|
| Modelos clustering | K-Means, DBSCAN, Jerárquico |
| Número clústeres | 5 |
| RMSE ARIMA | ~15 |
| Endpoints API | 8 |
| Tiempo API | < 50 ms |
| Visualizaciones | 5 tipos |

## 🔹 Conclusiones Académicas
- Clustering reveló **5 perfiles académicos**  
- ARIMA útil para cortoplazo  
- RNN útil para patrones no lineales  
- Arquitectura escalable  
- Dashboard accesible y claro  
