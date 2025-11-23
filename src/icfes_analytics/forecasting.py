import pandas as pd
import numpy as np

def forecast_next_period(df, group_col, target_col):

    # Validar columnas necesarias
    if "periodo" not in df.columns:
        return {"error": "El dataset no contiene la columna 'periodo'."}

    if target_col not in df.columns:
        return {"error": f"La columna '{target_col}' no existe en el dataset."}

    # Limpiar
    df = df.dropna(subset=[target_col]).copy()
    df["periodo"] = df["periodo"].astype(int)

    # Agrupar por año
    df_grouped = df.groupby("periodo")[target_col].mean().reset_index()
    df_grouped = df_grouped.sort_values("periodo")

    # Necesita mínimo 2 años
    if df_grouped.shape[0] < 2:
        return {"error": "No hay suficientes datos históricos para generar la predicción."}

    # Variables para regresión
    x = df_grouped["periodo"].values
    y = df_grouped[target_col].values

    # Regresión lineal con polyfit
    m, b = np.polyfit(x, y, 1)

    next_year = df_grouped["periodo"].max() + 1
    predicted_value = m * next_year + b

    return {
        "historical": [
            {"periodo": int(row["periodo"]), "value": float(row[target_col])}
            for _, row in df_grouped.iterrows()
        ],
        "forecast": float(predicted_value),
        "next_year": int(next_year)
    }
