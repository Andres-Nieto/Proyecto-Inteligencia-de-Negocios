from typing import Dict, List
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import os
from functools import lru_cache
@lru_cache(maxsize=1)
def load_data():
    return pd.read_csv("data/Dataset2-SaberPro(2021-2024)_LIMPIO.csv")
from src.icfes_analytics.forecasting import forecast_next_period

# --- FIX PARA IMPORTAR icfes_analytics DESDE /src (muy importante) ---
BASE_DIR = Path(__file__).resolve().parents[2]     # Proyecto/
SRC_DIR = BASE_DIR / "src"

if SRC_DIR.exists():
    sys.path.append(str(SRC_DIR))


#Se cargan los modelos entrenados, se procesan los datos de los estudiantes
# Se generan las recomendaciones automáticas

# Variables globales
DATA = {}
SCALER = None
KMEANS = None
FEATURE_COLS: List[str] = []
HISTORY: List[Dict] = []

# Nombres para mostrar al usuario
FRIENDLY = {
    "punt_global": "Puntaje Global",
    "mod_razona_cuantitat_punt": "Matemáticas",
    "mod_lectura_critica_punt": "Lectura Crítica",
    "mod_ingles_punt": "Inglés",
}

# Mapeo entre los nombres que recibe la API -> columnas del dataset real
API_TO_DATASET = {
    "punt_global": "punt_global",
    "punt_matematicas": "mod_razona_cuantitat_punt",
    "punt_lectura_critica": "mod_lectura_critica_punt",
    "punt_ingles": "mod_ingles_punt",
}

# Carga de modelos entrenados (scaler, kmeans, columnas)
def load_cluster_models():
    """
    Se ejecuta al inciar la API.
    Carga el scaler, kmeans y las columnas desde /models. (entrenados con train_save_models.py)
    """
    global SCALER, KMEANS, FEATURE_COLS
    models_dir = Path(__file__).resolve().parents[2] / "models"
    SCALER = joblib.load(models_dir / "scaler.pkl")
    KMEANS = joblib.load(models_dir / "kmeans.pkl")
    FEATURE_COLS = joblib.load(models_dir / "feature_cols.pkl")
    print("OK - Modelos cargados en memoria correctamente:", FEATURE_COLS)


def set_data_source(datasets: Dict):
    """Guarda datasets en memoria Saber 11 y Saber Pro (por si luego se usan para estadísticas).
    Y luego carga los modelos de clúster al iniciar la API."""
    global DATA
    DATA = datasets
    # Carga los modelos apenas arranca la API
    load_cluster_models()


# ---------------------- Utilidades de datasets limpios ----------------------
def list_datasets() -> Dict:
    """Retorna metadatos de los datasets cargados en memoria."""
    meta: Dict[str, Dict] = {}
    for name, df in DATA.items():
        try:
            meta[name] = {
                "rows": int(len(df)),
                "cols": int(df.shape[1]),
                "columns": df.columns.tolist(),
            }
        except Exception:
            meta[name] = {"rows": 0, "cols": 0, "columns": []}
    return meta


def get_dataset_sample(name: str, limit: int = 100, columns: List[str] | None = None) -> Dict:
    """Devuelve un sample (head) del dataset solicitado con columnas opcionales."""
    if name not in DATA:
        return {"error": f"Dataset '{name}' no disponible", "available": list(DATA.keys())}
    df = DATA[name]
    if columns:
        cols = [c for c in columns if c in df.columns]
        if cols:
            df = df.loc[:, cols]
    limit = max(1, min(int(limit), 1000))
    sample = df.head(limit).to_dict(orient="records")
    return {
        "name": name,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "sample": sample,
        "limit": limit,
    }


# Conversión y procesamiento de puntajes enviados a la API
def _scores_to_dataframe(scores: Dict) -> pd.DataFrame:
    """
    Crea un DataFrame con las columnas usadas en el entrenamiento,
    mapeando desde los nombres de entrada (API) hacia los nombres reales.
    """
    row = {}
    for api_key, ds_col in API_TO_DATASET.items():
        # Si existe en la entrada, úsalo; sino, 0
        value = scores.get(api_key, 0)
        if value is None or value == "":
            value = 0
        row[ds_col] = float(value)
    return pd.DataFrame([row], columns=FEATURE_COLS)

# Calculo de clúster y centroide
def _cluster_and_centroid(x_df: pd.DataFrame):
    """Escala los datos del estudiante, predice el clúster y devuelve su centroide real."""
    if SCALER is None or KMEANS is None:
        load_cluster_models()
    x_scaled = SCALER.transform(x_df)
    cid = int(KMEANS.predict(x_scaled)[0])
    centroid = KMEANS.cluster_centers_[cid]
    means, stds = SCALER.mean_, SCALER.scale_
    centroid_real = centroid * stds + means
    return cid, centroid, centroid_real, x_scaled

# Funciones auxiliares para ordernar y generar recomendaciones
def _rank_features(vec, cols):
    """Ordena las materias de mayor a menor valor."""
    idx = np.argsort(vec)[::-1]
    return [(cols[i], float(vec[i])) for i in idx]


def _careers_for_feature(feat_name: str) -> List[str]:
    """Sugiere carreras en función del área más fuerte."""
    if "cuantitat" in feat_name or "matematicas" in feat_name:
        return ["Ingeniería", "Economía", "Estadística", "Arquitectura"]
    if "lectura" in feat_name:
        return ["Derecho", "Comunicación Social", "Ciencias Políticas", "Psicología"]
    if "ingles" in feat_name:
        return ["Negocios Internacionales", "Idiomas", "Turismo", "Relaciones Públicas"]
    return ["Becas", "Programas de investigación", "Educación superior destacada"]


def _reinforcements_from_gaps(x_df: pd.DataFrame, centroid_real: np.ndarray) -> List[str]:
    """Sugiere refuerzos comparando los puntajes del estudiante con su cluster (centroide) y genera recomendaciones."""
    student = x_df.iloc[0].values
    gaps = centroid_real - student
    suggestions = []

    for i, col in enumerate(FEATURE_COLS):
        name = FRIENDLY.get(col, col)
        if gaps[i] > 15:
            if "lectura" in col:
                suggestions.append(f"Reforzar {name} con análisis de textos y comprensión crítica.")
            elif "cuantitat" in col or "matematicas" in col:
                suggestions.append(f"Mejorar {name} practicando razonamiento lógico y ejercicios guiados.")
            elif "ingles" in col:
                suggestions.append(f"Reforzar {name} con práctica auditiva y vocabulario técnico.")
            elif "global" in col:
                suggestions.append(f"Elevar {name} manteniendo equilibrio general.")
    return suggestions or ["Mantén tu nivel actual, sigue fortaleciendo tus competencias."]


# Funciones principales de predicción y recomendacion
def predict_cluster(scores: Dict) -> int:
    """Predice únicamente el clúster al que pertenece el estudiante."""
    x_df = _scores_to_dataframe(scores)
    cid, _, _, _ = _cluster_and_centroid(x_df)
    return cid


def recommend_from_cluster(scores: Dict, id_student: str = None) -> Dict:
    """
    Usa el clúster y el centroide para generar:
    - cluster_id
    - main_strength (sin contar puntaje global)
    - recommended_careers
    - reinforcement_suggestions
    """
    x_df = _scores_to_dataframe(scores)
    cid, centroid, centroid_real, x_scaled = _cluster_and_centroid(x_df)

    # Filtrar las materias reales (sin puntaje global)
    feature_subset = [f for f in FEATURE_COLS if "global" not in f]

    # Convertir el DataFrame a dict solo con materias
    student_values = {k: v for k, v in x_df.iloc[0].to_dict().items() if k in feature_subset}

    # Materia más fuerte
    max_feature = max(student_values, key=student_values.get)
    main_strength = FRIENDLY.get(max_feature, max_feature)
    careers = _careers_for_feature(max_feature)

    # Materias más débiles
    min_feature = min(student_values, key=student_values.get)
    weak_subject = FRIENDLY.get(min_feature, min_feature)

    # Recomendación de refuerzo
    reinforcements = _reinforcements_from_gaps(x_df, centroid_real)
    if reinforcements == ["Mantén tu nivel actual, sigue fortaleciendo tus competencias."]:
        reinforcements = [f"Refuerza {weak_subject} mediante práctica adicional y tutorías."]

    #Guardar en el historial con ID de estudiante
    save_to_history({
        "id_student": id_student,
        "puntajes": scores,
        "cluster": cid,
        "perfil": f"Perfil del estudiante clasificado en el clúster {cid}",
        "recomendaciones": reinforcements,
        "main_strength": main_strength,
        "recommended_careers": careers,
    })
    return {
        "cluster_id": cid,
        "main_strength": main_strength,
        "recommended_careers": careers,
        "recommendations": reinforcements,
        "profile": f"Perfil del estudiante clasificado en el clúster {cid}",
    }

# Historial de consultas 
HISTORY: List[Dict] = []

def save_to_history(entry: Dict):
    HISTORY.append(entry)

def get_history():
    return HISTORY[-50:]

def get_student_history(student_id: str):
    results = [e for e in HISTORY if e.get("id_student") == student_id]
    return results[-50:]

def get_summary():
    summary = {}
    for e in HISTORY:
        c = e.get("cluster")
        if c is not None:
            summary[c] = summary.get(c, 0) + 1
    return dict(sorted(summary.items()))

def clear_history():
    HISTORY.clear()
    return {"message": "Historial eliminado correctamente."}


def get_forecast(group_col: str, group_value: str, target_col: str):

    df = load_data()   # <-- CARGA UNA VEZ Y QUEDA EN MEMORIA

    df[group_col] = df[group_col].astype(str)
    df_filtered = df[df[group_col] == group_value]
    if df_filtered.empty:
        return {"error": "No hay datos suficientes para esta universidad."}

    result = forecast_next_period(df_filtered, group_col, target_col)

    # Si hubo error
    if "error" in result:
        return result

    # Nuevo historial simplificado
    history = result["historical"]

    print("TARGET RECIBIDO:", target_col)

    return {
        "historical": history,
        "forecast": result["forecast"],
        "next_year": result["next_year"]

    }
