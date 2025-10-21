import pandas as pd

# ------------------------------
# 1. Cargar dataset genérico
# ------------------------------
def load_dataset(path: str):
    """
    Carga cualquier dataset en formato json, csv o parquet que contenga 
    al menos las columnas 'text' y 'emotion'.
    """
    if path.endswith(".json") or path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True, encoding="utf-8")
    elif path.endswith(".csv"):
        df = pd.read_csv(path, encoding="utf-8")
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Formato no soportado. Usa JSON, CSV o Parquet.")

    # Estandarizar columnas posibles para "text"
    text = ["text", "comment", "tweet", "review"]
    emotion = ["emotion", "Emotion", "sentiment", "label"]

    # Renombrar columna de texto
    for col in text:
        if col in df.columns:
            df = df.rename(columns={col: "text"})
            break

    # Renombrar columna de emociones
    for col in emotion:
        if col in df.columns:
            df = df.rename(columns={col: "emotion"})
            break

    # Verificar que ya existan ambas columnas
    if not all(col in df.columns for col in ["text", "emotion"]):
        raise ValueError("El dataset debe contener al menos una columna de texto y una de emoción.")

    # Asegurar formato string
    df["text"] = df["text"].astype(str)
    df["emotion"] = df["emotion"].astype(str)

    return df


# ------------------------------
# 2. Preprocesamiento genérico
# ------------------------------
def preprocess_dataset(df: pd.DataFrame, top_n=5):
    """
    Filtra el dataset a las N emociones más comunes y devuelve df filtrado.
    """
    top_emotions = df["emotion"].value_counts().head(top_n).index.tolist()
    df_filtered = df[df["emotion"].isin(top_emotions)].copy()
    return df_filtered

# ------------------------------
# 3. Ejemplo de uso
# ------------------------------
path = "C:\\Users\\Usuario9\\OneDrive\\Documentos\\Master Big Data\\UPC - BIG DATA\\TFM\\TFM-main\\TFM-main\\datasets\\Kaggle\\csv\\tweet_sentiment.csv"

df = load_dataset(path)
df_filtered = preprocess_dataset(df, top_n=5)

print("Shape original:", df.shape)
print("Shape filtrado:", df_filtered.shape)
print("Emociones más comunes:", df_filtered["emotion"].unique())
