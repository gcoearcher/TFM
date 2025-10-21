import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
import time

# --- Cargar dataset ---
df = pd.read_csv("C:\\Users\\Usuario9\\OneDrive\\Documentos\\Master Big Data\\UPC - BIG DATA\\TFM\\TFM-main\\TFM-main\\datasets\\Kaggle\\csv\\DailyDialog.csv", dtype=str)

# --- Estandarizar columnas ---
df = df.rename(columns={
    "comment": "text",
    "tweet": "text",
    "sentiment": "emotion",
    "Emotion": "emotion"
})

# --- Extraer textos ---
texts = df["text"].astype(str).tolist()

# Cargar clasificador RoBERTa
@st.cache_resource
def load_model():
    return pipeline("text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=False)

classifier = load_model()

# Configuración de la ventana deslizante
WINDOW_SIZE = 300  # <- aquí alargamos la ventana
STEP_SIZE = 150    # <- desplazamiento de la ventana

# Simulación de ingestión batch
def sliding_window_batches(data, window=WINDOW_SIZE, step=STEP_SIZE):
    for i in range(0, len(data), step):
        yield data[i:i+window]

# App Streamlit
st.title("📊 Evolución de emociones en tiempo real (RoBERTa)")

uploaded_file = st.file_uploader("Sube un dataset CSV/JSONL con columnas 'text' y 'emotion'", type=["csv", "jsonl"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_json(uploaded_file, lines=True)

    # Estandarizar columna de texto
    if "text" not in df.columns:
        if "tweet" in df.columns:
            df.rename(columns={"tweet": "text"}, inplace=True)
        elif "comment" in df.columns:
            df.rename(columns={"comment": "text"}, inplace=True)

    texts = df["text"].astype(str).tolist()

    # Procesamiento en batches
    emotion_evolution = []
    progress = st.progress(0)
    placeholder = st.empty()

    for idx, batch in enumerate(sliding_window_batches(texts)):
        # Clasificación con RoBERTa
        results = classifier(batch)
        emotions = [res['label'] for res in results]

        counter = Counter(emotions)
        emotion_evolution.append(counter)

        # Mostrar resultados en streamlit
        df_evo = pd.DataFrame(emotion_evolution).fillna(0)
        placeholder.line_chart(df_evo)

        progress.progress(min((idx+1)*STEP_SIZE/len(texts), 1.0))
        time.sleep(1)  # simula latencia de llegada de datos

    st.success("✅ Procesamiento completado")

# --- Modo Streamlit ---
def run_streamlit():
    import streamlit as st
    st.title("Evolución de emociones en tiempo real (RoBERTa + Sliding Window)")
    
    placeholder = st.empty()
    for batch in sliding_window_batches(texts):
        df = sliding_window_batches(batch)
        with placeholder.container():
            st.write("### Nuevo batch procesado:")
            st.dataframe(df)

# --- Detección del modo ---
if __name__ == "__main__":
    try:
        import streamlit.runtime.scriptrunner as stsr
        if stsr.get_script_run_ctx():
            run_streamlit()
    except ImportError:
        # Modo script normal
        for batch in sliding_window_batches(texts):
            df = sliding_window_batches(batch)
            print("Nuevo batch procesado:")
            print(df)