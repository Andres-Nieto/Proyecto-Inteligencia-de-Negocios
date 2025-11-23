from pydantic import BaseModel
from typing import Optional, List

# En este archivo se definen los esquemas (modelos de datos) que utiliza FastAPI para validar la información que entra y sale de la API.
# Esto permite tener un control sobre los tipos de datos y las estructuras JSON.

class ScoresInput(BaseModel):
    """
    Modelo que representa los datos de entrada para realizar predicciones
    de rendimiento académico de un estudiante.
    """
    id_student: Optional[str] = None
    punt_global: Optional[float]
    punt_matematicas: Optional[float]
    punt_lectura_critica: Optional[float]
    punt_ingles: Optional[float]
    estrato: Optional[int] = None


class PredictResponse(BaseModel):
    """
    Modelo que define la estructura de la respuesta del endpoint /predict.
    Contiene la fortaleza principal, carreras recomendadas y refuerzos sugeridos.
    """
    id_student: Optional[str]
    main_strength: str
    recommended_careers: List[str]
    reinforcement_suggestions: List[str]


class HealthResponse(BaseModel):
    """
    Modelo utilizado para el endpoint /health.
    Simplemente informa el estado actual del servicio.
    """
    status: str
    detail: str


class ClusterRequest(BaseModel):
    """
    Modelo de entrada para el endpoint /cluster.
    Recibe los puntajes principales con los que se predice el grupo o cluster del estudiante.
    """
    id_student: Optional[str] = None
    punt_global: float
    punt_matematicas: float
    punt_lectura_critica: float
    punt_ingles: Optional[float]


class ClusterResponse(BaseModel):
    """
    Modelo de salida del endpoint /cluster.
    Devuelve el número del cluster asignado y las recomendaciones asociadas a ese grupo.
    """
    cluster: int
    recommendations: List[str]

class ForecastRequest(BaseModel):
    group_col: str   # ciudad, sede, colegio
    group_value: str
    target_col: str  # punt_global, punt_matematicas, etc