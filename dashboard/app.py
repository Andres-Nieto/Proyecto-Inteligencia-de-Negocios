import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="ICFES Dashboard", layout="wide")
st.title("ICFES – Dashboard de Estadísticas y Recomendaciones")

# ------------------------- Utilidades API -------------------------

def api_get(path: str) -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=15)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_delete(path: str) -> Dict[str, Any]:
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ------------------------- Sidebar (Filtros / Input) -------------------------
with st.sidebar:
    st.header("Entrada de Puntajes")
    st.caption("Ingresa puntajes para generar recomendaciones.")
    id_student = st.text_input("ID estudiante (opcional)")
    punt_global = st.number_input("Puntaje Global", min_value=0, max_value=500, value=0, step=1, format="%d")
    punt_matematicas = st.number_input("Matemáticas", min_value=0, max_value=500, value=0, step=1, format="%d")
    punt_lectura_critica = st.number_input("Lectura Crítica", min_value=0, max_value=500, value=0, step=1, format="%d")
    punt_ingles = st.number_input("Inglés", min_value=0, max_value=500, value=0, step=1, format="%d")

    st.markdown("---")
    do_predict = st.button("Generar Recomendaciones")

# ------------------------- Layout principal -------------------------
col_left, col_right = st.columns([1.25, 1])

# ------------------------- Columna Izquierda -------------------------
with col_left:
    st.subheader("Perfil y Recomendaciones")
    if do_predict:
        payload = {
            "id_student": id_student or None,
            "punt_global": int(punt_global),
            "punt_matematicas": int(punt_matematicas),
            "punt_lectura_critica": int(punt_lectura_critica),
            "punt_ingles": int(punt_ingles),
        }
        resp = api_post("/predict", payload)
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.success("Predicción generada correctamente")
            st.write(f"ID estudiante: {resp.get('id_student', id_student or '—')}")
            cluster_id = resp.get("cluster_id")  # Se usa internamente pero no se muestra
            st.write(f"Fortaleza principal: {resp.get('main_strength','N/D')}")
            st.write("Carreras sugeridas:")
            for c in resp.get("recommended_careers", []):
                st.markdown(f"- {c}")
            st.write("Refuerzos sugeridos:")
            for r in resp.get("reinforcement_suggestions", []):
                st.markdown(f"- {r}")

            # ------------------------- Radar Comparativo (Global Histórico) -------------------------
            hist_all = api_get("/history")
            if isinstance(hist_all, list) and hist_all:
                df_hist_all = pd.DataFrame(hist_all)
                if "puntajes" in df_hist_all.columns:
                    punt_cols_all = df_hist_all["puntajes"].apply(pd.Series)
                    punt_cols_all = punt_cols_all.loc[:, [c for c in punt_cols_all.columns if c not in df_hist_all.columns]]
                    df_hist_all = pd.concat([df_hist_all.drop(columns=["puntajes"]), punt_cols_all], axis=1)
                score_cols_radar = [c for c in ["punt_global","punt_matematicas","punt_lectura_critica","punt_ingles"] if c in df_hist_all.columns]
                if score_cols_radar:
                    global_means = df_hist_all[score_cols_radar].mean()
                    student_scores = [payload.get(col, 0) for col in score_cols_radar]
                    categories = [
                        "Global" if c == "punt_global" else
                        "Matemáticas" if c == "punt_matematicas" else
                        "Lectura Crítica" if c == "punt_lectura_critica" else
                        "Inglés" if c == "punt_ingles" else c
                        for c in score_cols_radar
                    ]
                    st.markdown("---")
                    st.write("### Comparación Estudiante vs Promedio Histórico")
                    fig_radar = go.Figure()
                    # Colores personalizados: Estudiante (rojo) vs Promedio (azul)
                    fig_radar.add_trace(go.Scatterpolar(
                        r=student_scores,
                        theta=categories,
                        fill='toself',
                        name='Estudiante',
                        line_color='#FF4B4B',
                        fillcolor='rgba(255,75,75,0.40)'
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=global_means.values,
                        theta=categories,
                        fill='toself',
                        name='Promedio Histórico',
                        line_color='#1f77b4',
                        fillcolor='rgba(31,119,180,0.30)'
                    ))
                    max_axis = max(300, int(max(student_scores + list(global_means.values))))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max_axis])), showlegend=True)
                    st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.caption("Historial vacío; genera más consultas para ver comparación.")


    # ------------------------- Estadísticas -------------------------
    st.markdown("---")
    st.subheader("Estadísticas del Historial")
    hist_raw = api_get("/history")
    if isinstance(hist_raw, list) and hist_raw:
        df_hist_full = pd.DataFrame(hist_raw)
        # Expandir puntajes
        if "puntajes" in df_hist_full.columns:
            punt_cols_full = df_hist_full["puntajes"].apply(pd.Series)
            punt_cols_full = punt_cols_full.loc[:, [c for c in punt_cols_full.columns if c not in df_hist_full.columns]]
            df_hist_full = pd.concat([df_hist_full.drop(columns=["puntajes"]), punt_cols_full], axis=1)

        # Filtro simple por rango de puntaje global
        col_r1, col_r2 = st.columns(2)
        if "punt_global" in df_hist_full.columns:
            min_pg, max_pg = int(df_hist_full["punt_global"].min()), int(df_hist_full["punt_global"].max())
            if min_pg == max_pg:
                col_r2.caption(f"Rango Puntaje Global (único valor: {min_pg})")
                range_pg = (min_pg, max_pg)
            else:
                range_pg = col_r2.slider("Rango Puntaje Global", min_pg, max_pg, (min_pg, max_pg))
        else:
            range_pg = (0, 9999)
        df_filt = df_hist_full.copy()
        if "punt_global" in df_filt.columns:
            df_filt = df_filt[(df_filt["punt_global"] >= range_pg[0]) & (df_filt["punt_global"] <= range_pg[1])]
        st.caption(f"Registros filtrados: {len(df_filt)}")

        # Estadísticas generales (sin segmentación por clúster)
        score_cols = [c for c in ["punt_global","punt_matematicas","punt_lectura_critica","punt_ingles"] if c in df_filt.columns]
        if score_cols:
            st.write("Promedios históricos de puntajes:")
            st.dataframe(pd.DataFrame(df_filt[score_cols].mean().round(1), columns=["promedio"]))

        # KPIs generales
        total_consultas = len(df_filt)
        col_kpi1, col_kpi2 = st.columns(2)
        col_kpi1.metric("Consultas totales", total_consultas)
        if score_cols:
            col_kpi2.metric("Materias analizadas", len(score_cols))

        # Distribución promedio general
        if score_cols:
            avg_scores = df_filt[score_cols].mean().round(1)
            fig_avg = go.Figure()
            fig_avg.add_trace(go.Bar(x=avg_scores.index, y=avg_scores.values, text=avg_scores.values, textposition="outside"))
            fig_avg.update_layout(title="Promedio General de Puntajes (Historial Filtrado)")
            st.plotly_chart(fig_avg, use_container_width=True)
    else:
        st.info("Sin datos en el historial para generar estadísticas.")

    st.markdown("---")
    st.subheader("Historial (últimas 50 consultas)")
    hist = api_get("/history")
    if isinstance(hist, list) and hist:
        df_hist = pd.DataFrame(hist)
        if "puntajes" in df_hist.columns:
            punt_cols = df_hist["puntajes"].apply(pd.Series)
            punt_cols = punt_cols.loc[:, [c for c in punt_cols.columns if c not in df_hist.columns]]
            df_hist = pd.concat([df_hist.drop(columns=["puntajes"]), punt_cols], axis=1)
        st.dataframe(df_hist.tail(50), use_container_width=True)
    else:
        st.caption("Sin historial disponible.")

    # (Sección de datasets movida a ancho completo más abajo)

# ------------------------- Columna Derecha -------------------------
with col_right:
    st.subheader("Búsqueda por Estudiante")
    search_id = st.text_input("ID para historial individual")
    if st.button("Buscar historial"):
        if not search_id:
            st.warning("Ingresa un ID válido.")
        else:
            res = api_get(f"/student/{search_id}")
            if isinstance(res, list) and res:
                df_stu = pd.DataFrame(res)
                if "puntajes" in df_stu.columns:
                    punt_cols = df_stu["puntajes"].apply(pd.Series)
                    punt_cols = punt_cols.loc[:, [c for c in punt_cols.columns if c not in df_stu.columns]]
                    df_stu = pd.concat([df_stu.drop(columns=["puntajes"]), punt_cols], axis=1)
                st.write(f"Consultas registradas: {len(df_stu)}")
                st.dataframe(df_stu.tail(20), use_container_width=True)
            else:
                st.info("No hay registros para ese estudiante.")

    st.markdown("---")
    st.subheader("Mantenimiento")
    if st.button("Limpiar historial"):
        r = api_delete("/clear-history")
        if r.get("message"):
            st.success(r["message"])
        else:
            st.error(r.get("error", "Error desconocido"))

    st.markdown("---")
    st.caption(f"API base: {API_BASE}")
    health = api_get("/health")
    if health.get("status") == "ok":
        st.success("API OK")
    else:
        st.error("API no disponible")

# --- Exploración de datasets limpios desde la API (ancho completo) ---
st.markdown("---")
with st.container():
    st.markdown("### Datasets limpios")

    meta = api_get("/datasets")
    if isinstance(meta, dict) and "datasets" in meta and meta["datasets"]:
        ds_names = list(meta["datasets"].keys())

        # Controles principales (ancho completo)
        c_top1, c_top2, c_top3, c_top4 = st.columns([1.6, 1, 1, 1])
        with c_top1:
            ds_sel = st.selectbox("Selecciona un dataset", ds_names, index=0)
        rows = int(meta["datasets"][ds_sel].get("rows", 0))
        cols_total = int(meta["datasets"][ds_sel].get("cols", 0))
        cols_meta = meta["datasets"][ds_sel].get("columns", [])
        # Solo columnas de puntajes
        score_cols = [
            c for c in cols_meta
            if c == "punt_global" or c.startswith("punt_") or (c.startswith("mod_") and c.endswith("_punt"))
        ]
        # Restricción específica para SaberPro: quitar 'competen', 'comuni' y 'razona'
        if ds_sel.lower() == "saberpro":
            allowed_saberpro = {"mod_lectura_critica_punt", "mod_ingles_punt", "mod_razona_cuantitat_punt", "punt_global"}
            score_cols = [c for c in score_cols if c in allowed_saberpro]

        # Opciones de vista
        c_opt1, c_opt2, c_opt3 = st.columns([1, 1, 2])
        with c_opt1:
            limit = st.slider("Filas a mostrar", 10, 500, 100, step=10)
        with c_opt2:
            use_friendly = st.checkbox("Nombres amigables", value=True)

        # Consumir API
        params = f"?limit={limit}"
        if score_cols:
            params += "&columns=" + ",".join(score_cols)
        data = api_get(f"/dataset/{ds_sel}{params}")

        if isinstance(data, dict) and "sample" in data:
            df_sample = pd.DataFrame(data["sample"]) if data.get("sample") else pd.DataFrame(columns=score_cols)

            # Renombrar columnas a etiquetas amigables (solo visual)
            friendly_map = {
                "punt_global": "Puntaje Global",
                "punt_matematicas": "Matemáticas",
                "mod_razona_cuantitat_punt": "Matemáticas",
                "punt_lectura_critica": "Lectura Crítica",
                "mod_lectura_critica_punt": "Lectura Crítica",
                "punt_ingles": "Inglés",
                "mod_ingles_punt": "Inglés",
                "punt_c_naturales": "Ciencias Naturales",
                "punt_sociales_ciudadanas": "Sociales y Ciudadanas",
            }
            df_view = df_sample.copy()
            if use_friendly:
                df_view = df_view.rename(columns=friendly_map)

            # Pestañas: Tabla / Resumen
            tab1, tab2 = st.tabs(["Tabla", "Resumen"])
            with tab1:
                st.dataframe(df_view, use_container_width=True, height=520)
                csv = df_view.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Descargar CSV (sample)",
                    data=csv,
                    file_name=f"{ds_sel}_scores_sample.csv",
                    mime="text/csv"
                )
                st.caption("Vista limitada a columnas de puntajes para mayor legibilidad.")

            with tab2:
                if not df_sample.empty:
                    means = df_sample.mean(numeric_only=True).dropna()
                    if not means.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=[friendly_map.get(c, c) for c in means.index],
                            y=means.values,
                            text=[round(v,1) for v in means.values],
                            textposition="outside"
                        ))
                        max_val = float(means.max())
                        headroom = max(15.0, max_val * 0.12)  # espacio extra para que no se corte el texto superior
                        fig.update_layout(
                            title="Promedio de puntajes (muestra)",
                            yaxis_title="Puntaje",
                            yaxis=dict(range=[0, max_val + headroom]),
                            margin=dict(t=80, b=40, l=40, r=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No hay datos numéricos para resumir.")
                else:
                    st.info("No hay filas en la muestra actual.")
        else:
            if isinstance(data, dict) and data.get("error"):
                st.warning(data["error"])
            else:
                st.warning("No fue posible cargar el sample.")
    else:
        st.info("API sin datasets cargados.")

st.markdown("---")
st.caption("Sprint 4 – Dashboard interactivo para recomendaciones ICFES")

# ------------------------- Distribución – Histograma -------------------------
st.markdown("---")
with st.container():
    st.markdown("### Distribución – Histograma (puntajes)")

    meta = api_get("/datasets")
    if isinstance(meta, dict) and "datasets" in meta and meta["datasets"]:
        ds_names = list(meta["datasets"].keys())
        c1, c2, c3 = st.columns([1.4, 1, 1])
        with c1:
            ds_sel = st.selectbox("Dataset", ds_names, index=0, key="dist_ds")
        with c2:
            limit = st.slider("Filas (muestra)", 100, 1000, 500, step=50, help="La API limita a 1000 filas.")
        with c3:
            dropna = st.checkbox("Omitir NaN", value=True)

        cols_meta = meta["datasets"][ds_sel].get("columns", [])
        score_cols_all = [
            c for c in cols_meta
            if c == "punt_global" or c.startswith("punt_") or (c.startswith("mod_") and c.endswith("_punt"))
        ]
        if ds_sel.lower() == "saberpro":
            allowed_saberpro = {"mod_lectura_critica_punt", "mod_ingles_punt", "mod_razona_cuantitat_punt", "punt_global"}
            score_cols_all = [c for c in score_cols_all if c in allowed_saberpro]

        params = f"?limit={limit}&columns=" + ",".join(score_cols_all) if score_cols_all else f"?limit={limit}"
        data = api_get(f"/dataset/{ds_sel}{params}")
        df_scores = pd.DataFrame(data.get("sample", [])) if isinstance(data, dict) else pd.DataFrame()
        if dropna and not df_scores.empty:
            df_scores = df_scores.dropna(how="all")

        if df_scores.empty or not score_cols_all:
            st.info("No hay datos suficientes para graficar.")
        else:
            for c in score_cols_all:
                if c in df_scores.columns:
                    df_scores[c] = pd.to_numeric(df_scores[c], errors="coerce")

            # Construir nombres amigables para el selector
            friendly_map = {
                "punt_global": "Puntaje Global",
                "punt_matematicas": "Matemáticas",
                "mod_razona_cuantitat_punt": "Matemáticas",
                "punt_lectura_critica": "Lectura Crítica",
                "mod_lectura_critica_punt": "Lectura Crítica",
                "punt_ingles": "Inglés",
                "mod_ingles_punt": "Inglés",
                "punt_c_naturales": "Ciencias Naturales",
                "punt_sociales_ciudadanas": "Sociales y Ciudadanas",
            }
            # Preferir columnas 'punt_' cuando hay duplicados de la misma materia
            label_to_col = {}
            for c in [c for c in score_cols_all if c in df_scores.columns]:
                label = friendly_map.get(c, c)
                if label not in label_to_col:
                    label_to_col[label] = c
                else:
                    if c.startswith("punt_") and not label_to_col[label].startswith("punt_"):
                        label_to_col[label] = c

            labels_options = list(label_to_col.keys())
            colh1, colh2, colh3 = st.columns([1.2, 1, 1])
            with colh1:
                chosen_label = st.selectbox("Variable", labels_options, key="hist_label") if labels_options else None
                col_sel = label_to_col.get(chosen_label) if chosen_label else None
            with colh2:
                bins = st.slider("Bins", 5, 100, 30)
            if not col_sel:
                st.info("No hay variables disponibles para graficar.")
            else:
                series = df_scores[col_sel].dropna()
                vmin, vmax = (float(series.min()), float(series.max())) if not series.empty else (0.0, 1.0)
                with colh3:
                    range_sel = st.slider("Rango", min_value=float(vmin), max_value=float(vmax), value=(float(vmin), float(vmax)))

                s_filtered = series[(series >= range_sel[0]) & (series <= range_sel[1])]
                axis_label = chosen_label or col_sel
                fig_h = px.histogram(s_filtered, nbins=bins, labels={"value": axis_label}, marginal="rug")
                fig_h.update_layout(xaxis_title="Puntaje", yaxis_title="Frecuencia")
                st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("API sin datasets cargados.")
