import streamlit as st
import pandas as pd
import time
from datetime import datetime
from transformers import pipeline
import matplotlib.pyplot as plt

# CONFIGURACIÓN INICIAL
st.set_page_config(page_title="Evolución Emociones", layout="wide")
st.title("Evolución de emociones en comentarios")

# Ruta del CSV
csv_path = "sentiment_analysis.csv"

# Cargar modelo de sentimiento / emoción
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion")

emotion_model = load_model()

# DataFrame global para almacenar resultados
if "df_all" not in st.session_state:
    st.session_state.df_all = pd.DataFrame(columns=["timestamp", "text", "emotion"])

# FUNCIÓN PRINCIPAL
def process_batch(df, start, end):
    """Procesa un lote de comentarios del CSV"""
    batch = df.iloc[start:end]
    if batch.empty:
        return None

    # Analizar emociones con RoBERTa
    results = emotion_model(batch["text"].tolist())
    emotions = [r["label"] for r in results]

    # Agregar timestamp
    processed = pd.DataFrame({
        "timestamp": [datetime.now()] * len(batch),
        "text": batch["text"].tolist(),
        "emotion": emotions
    })
    return processed


# LOOP STREAMING
st.write("El sistema lee 10 comentarios cada 10 segundos...")

df_csv = pd.read_csv(csv_path)
batch_size = 10

# Contenedor de Streamlit para gráficos
chart_placeholder = st.empty()

for i in range(0, len(df_csv), batch_size):
    new_data = process_batch(df_csv, i, i + batch_size)
    if new_data is not None:
        st.session_state.df_all = pd.concat([st.session_state.df_all, new_data])

        # Contar frecuencia de emociones por timestamp
        df_plot = (
            st.session_state.df_all.groupby(["timestamp", "emotion"])
            .size()
            .reset_index(name="count")
        )

        # Pivot para gráfico de líneas
        pivot_df = df_plot.pivot(index="timestamp", columns="emotion", values="count").fillna(0)

        # Dibujar gráfico
        chart_placeholder.line_chart(pivot_df)

    time.sleep(10)

if not df_csv.empty:
    df_grouped = df_csv.groupby("emotion").size()
    df_percent = 100 * df_grouped / df_grouped.sum()
    st.bar_chart(df_percent)



