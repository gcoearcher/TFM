import pandas as pd
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
from keybert import KeyBERT
from sklearn.cluster import KMeans
from collections import Counter
import streamlit as st

# 1. Cargar dataset

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

df = load_dataset("C:\\Users\\Usuario9\\OneDrive\\Documentos\\Master Big Data\\UPC - BIG DATA\\TFM\\TFM-main\\TFM-main\\datasets\\Kaggle\\csv\\DailyDialog.csv")
texts = df['text'].astype(str).tolist()

# Crear columna con la emoción principal
df['main_emotion'] = df['emotion'].astype(str)
#df["main_emotion"] = df["emotion"].apply(lambda x: x[0] if len(x) > 0 else "none")

# Filtrar las 5 emociones más comunes
top_emotions = df["main_emotion"].value_counts().head(5).index.tolist()
df_filtered = df[df["main_emotion"].isin(top_emotions)].copy()

# Reducir el número de textos para mejor visualización
# Tomar una muestra equilibrada de cada emoción (máximo 300 textos por emoción para HDBSCAN)
sample_size_per_emotion = 300

df_sampled = pd.DataFrame()
for emotion in top_emotions:
    emotion_data = df_filtered[df_filtered["main_emotion"] == emotion]
    if len(emotion_data) > sample_size_per_emotion:
        emotion_sample = emotion_data.sample(n=sample_size_per_emotion, random_state=42)
    else:
        emotion_sample = emotion_data
    df_sampled = pd.concat([df_sampled, emotion_sample], ignore_index=True)

print(f"Analizando {len(df_sampled)} textos de las emociones: {top_emotions}")

# 3. Generar embeddings y clustering
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(df_sampled["text"].tolist(), show_progress_bar=True)

# UMAP + HDBSCAN
umap_reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.0, random_state=42)
umap_2d = umap_reducer.fit_transform(embeddings)

hdb_clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30,     # Aumentar de 15 a 30
    min_samples=10,          # Añadir min_samples
    cluster_selection_epsilon=0.2  # Control de granularidad
)
cluster_labels = hdb_clusterer.fit_predict(umap_2d)

# 4. Extraer palabras clave con KeyBERT
print("\nExtrayendo palabras clave...")
kw_model = KeyBERT(model=model)
cluster_keywords = {}

for cluster_id in np.unique(cluster_labels):
    if cluster_id == -1:  # Saltar ruido
        continue
    cluster_docs = [df_sampled["text"].iloc[i] for i in range(len(cluster_labels)) if cluster_labels[i] == cluster_id]
    cluster_text = " ".join(cluster_docs)
    keywords = kw_model.extract_keywords(cluster_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=8)
    cluster_keywords[cluster_id] = [kw[0] for kw in keywords]

# 5. Resultados y visualización
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"\n RESULTADOS: {n_clusters} clusters encontrados")

# Información por cluster

for i in range(n_clusters):
    cluster_data = df_sampled[cluster_labels == i]
    emotion_dist = cluster_data["main_emotion"].value_counts()
    keywords = cluster_keywords.get(i, [])
    # Emoción principal = la más frecuente
    #main_emotion = emotion_dist.idxmax() if not emotion_dist.empty else "none"

    print(f"\n Cluster {i} ({len(cluster_data)} textos):")
    print(f"   Emociones: {dict(emotion_dist)}")
    print(f"   Keywords: {keywords}")

# 6. Visualización con palabras clave
plt.figure(figsize=(14, 10))
colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 1)))

for i in range(n_clusters):
    cluster_mask = cluster_labels == i
    plt.scatter(umap_2d[cluster_mask, 0], umap_2d[cluster_mask, 1], 
                c=[colors[i]], s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                label=f" Cluster {i}")
    
    # Añadir keywords en el centro del cluster
    cluster_points = umap_2d[cluster_mask]
    center_x, center_y = cluster_points.mean(axis=0)
    top_keywords = ", ".join(cluster_keywords.get(i, [])[:3])
    
    plt.text(center_x, center_y, top_keywords, fontsize=11, weight='bold',
             ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=3))

# Puntos de ruido
if -1 in cluster_labels:
    noise_mask = cluster_labels == -1
    plt.scatter(umap_2d[noise_mask, 0], umap_2d[noise_mask, 1], 
                c='black', s=30, alpha=0.4, marker='x', label='Ruido')


# === NIVEL 2: Subclustering emocional dentro de cada cluster ===

df_sampled["cluster_id"] = cluster_labels
subcluster_results = {}

for cluster_id in sorted(df_sampled["cluster_id"].unique()):
    if cluster_id == -1:  # ignoramos ruido
        continue

    # Datos del cluster actual
    cluster_data = df_sampled[df_sampled["cluster_id"] == cluster_id]
    cluster_embeddings = embeddings[cluster_data.index]

    # Definir número de subclusters (máx 3, al menos 2 si hay >1 emoción)
    n_unique_emotions = cluster_data["main_emotion"].nunique()
    k = min(3, n_unique_emotions) if n_unique_emotions > 1 else 1

    if k > 1:  # Solo aplicamos KMeans si hay más de una emoción
        kmeans = KMeans(n_clusters=k, random_state=42)
        sub_labels = kmeans.fit_predict(cluster_embeddings)
    else:
        sub_labels = [0] * len(cluster_data)

    # Añadimos columna de subcluster
    df_sampled.loc[cluster_data.index, "subcluster_id"] = sub_labels

    # Guardamos distribución emocional por subcluster
    subcluster_info = {}
    for sub_id in range(k):
        sub_emotions = df_sampled.loc[
            (df_sampled["cluster_id"] == cluster_id) & 
            (df_sampled["subcluster_id"] == sub_id), "main_emotion"
        ]
        subcluster_info[sub_id] = dict(Counter(sub_emotions))

    subcluster_results[cluster_id] = subcluster_info

# === 3. Mostrar resultados de subclustering ===
for cluster_id, subs in subcluster_results.items():
    print(f"\n Cluster {cluster_id}:")
    for sub_id, emo_dist in subs.items():
        print(f"   ▸ Subcluster {sub_id}: {emo_dist}")


plt.title("Clustering de Emociones con Palabras Clave (KeyBERT + MiniLM)", fontsize=16, pad=20)
plt.xlabel("UMAP Dimensión 1", fontsize=12)
plt.ylabel("UMAP Dimensión 2", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
