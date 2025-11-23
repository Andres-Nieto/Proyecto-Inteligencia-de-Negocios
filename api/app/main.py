from fastapi import FastAPI
from pathlib import Path
import pandas as pd

from .schemas import (
    ForecastRequest, ScoresInput, PredictResponse, HealthResponse,
    ClusterRequest, ClusterResponse
)

from .services import (
    set_data_source,
    get_history, save_to_history,
    predict_cluster, recommend_from_cluster
)
from api.app import services

app = FastAPI(title="ICFES Recommendation API", version="1.0")

# Startup: cargar datasets y modelos cuando inicia la API
@app.on_event("startup")
def load_data():
    data_dir = Path(__file__).resolve().parents[2] / "data"
    saber11 = list(data_dir.glob("Dataset1*LIMPIO.csv"))
    saberpro = list(data_dir.glob("Dataset2*LIMPIO.csv"))

    data = {}
    if saber11:  data["saber11"]  = pd.read_csv(saber11[0])
    if saberpro: data["saberpro"] = pd.read_csv(saberpro[0])

    print("✅ Datos cargados en memoria:", list(data.keys()))
    set_data_source(data)

#  Health 
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "detail": "API running successfully"}

#  Predict ( Basado en clúster/centroide ) Devuelve recomendaciones automáticas
@app.post("/predict", response_model=PredictResponse)
def predict(data: ScoresInput):
    """
    A partir de los puntajes del estudiante:
      - detecta el clúster y su centroide,
      - identifica la fortaleza principal (main_strength),
      - sugiere carreras acordes,
      - propone refuerzos en áreas por debajo del centroide.
    """
    info = recommend_from_cluster(data.dict(),id_student=data.id_student)
    return {
        "id_student": data.id_student,
        "cluster_id": info["cluster_id"],
        "main_strength": info["main_strength"],
        "recommended_careers": info["recommended_careers"],
        "reinforcement_suggestions": info["recommendations"],
    }

# Cluster : Recibe puntajes y devuelve clúster + recomendaciones 
@app.post("/cluster", response_model=ClusterResponse)
def cluster(data: ClusterRequest):
    info = recommend_from_cluster(data.dict(), id_student=data.id_student)
    cluster_id = info["cluster_id"]

    # Guardamos la consulta incluyendo el id del estudiante
    save_to_history({
        "puntajes": data.dict(),
        "cluster": cluster_id,
        "perfil": info["profile"],
        "recomendaciones": info["recommendations"],
        "top_features": info.get("top_features", []),
    })

    return {"cluster": cluster_id, "recommendations": info["recommendations"]}



#  History : Devuelve el historial de consultas
@app.get("/history", summary="Historial de consultas")
def history():
    return get_history()


# Student history: Devuelve historial por estudiante 
@app.get("/student/{id_student}", summary="Historial por estudiante")
def student_history(id_student: str):
    from .services import get_student_history
    return get_student_history(id_student)

# Summary: Devuelve cuantos estudiantes hay por cluster 
@app.get("/summary", summary="Resumen de clusters")
def summary():
    from .services import get_summary
    return get_summary()

# Clear history: Limpia el historial de consultas 
@app.delete("/clear-history", summary="Eliminar historial de consultas")
def delete_history():
    from .services import clear_history
    return clear_history()


#  Datasets limpios: listar y samplear
@app.get("/datasets", summary="Listar datasets limpios disponibles")
def datasets():
    from .services import list_datasets
    return {"datasets": list_datasets()}


@app.get("/dataset/{name}", summary="Sample de un dataset limpio")
def dataset_sample(name: str, limit: int = 100, columns: str | None = None):
    from .services import get_dataset_sample
    cols = [c.strip() for c in columns.split(",")] if columns else None
    return get_dataset_sample(name, limit=limit, columns=cols)

@app.post("/forecast")
def forecast(req: ForecastRequest):
    return services.get_forecast(
        req.group_col,
        req.group_value,
        req.target_col
    )
