import pandas as pd
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
from keybert import KeyBERT
from sklearn.cluster import KMeans
from collections import Counter
import base64
import io

# 🔧 1. Cargar y preparar datos
def cargar_datos(ruta_csv, sample_size_per_emotion=300):
    df = pd.read_csv(ruta_csv, encoding="utf-8")
    df['main_emotion'] = df['sentiment'].astype(str)
    top_emotions = df["main_emotion"].value_counts().head(3).index.tolist()
    df_filtered = df[df["main_emotion"].isin(top_emotions)].copy()

    df_sampled = pd.DataFrame()
    for emotion in top_emotions:
        emotion_data = df_filtered[df_filtered["main_emotion"] == emotion]
        if len(emotion_data) > sample_size_per_emotion:
            emotion_sample = emotion_data.sample(n=sample_size_per_emotion, random_state=42)
        else:
            emotion_sample = emotion_data
        df_sampled = pd.concat([df_sampled, emotion_sample], ignore_index=True)

    return df_sampled, top_emotions

# 🔧 2. Embeddings y clustering
def generar_clusters(df_sampled):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(df_sampled["text"].tolist(), show_progress_bar=True)

    umap_reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.0, random_state=42)
    umap_2d = umap_reducer.fit_transform(embeddings)

    hdb_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=10,
        cluster_selection_epsilon=0.2
    )
    cluster_labels = hdb_clusterer.fit_predict(umap_2d)

    return model, umap_2d, cluster_labels

# 🔧 3. Extraer palabras clave
def extraer_keywords(df_sampled, cluster_labels, model):
    kw_model = KeyBERT(model=model)
    cluster_keywords = {}

    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue
        cluster_docs = [df_sampled["text"].iloc[i] for i in range(len(cluster_labels)) if cluster_labels[i] == cluster_id]
        cluster_text = " ".join(cluster_docs)
        keywords = kw_model.extract_keywords(cluster_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=8)
        cluster_keywords[cluster_id] = [kw[0] for kw in keywords]

    return cluster_keywords

# 🔧 4. Preparar resumen por cluster
def preparar_resumen(df_sampled, cluster_labels, cluster_keywords):
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    resumen_clusters = []

    for i in range(n_clusters):
        cluster_data = df_sampled[cluster_labels == i]
        emotion_dist = cluster_data["main_emotion"].value_counts().to_dict()
        keywords = cluster_keywords.get(i, [])
        resumen_clusters.append({
            "id": i,
            "size": len(cluster_data),
            "emotions": emotion_dist,
            "keywords": keywords
        })

    return n_clusters, resumen_clusters

# 🔧 5. Visualización en base64
def generar_grafico_clusters(umap_2d, cluster_labels, cluster_keywords):
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    plt.figure(figsize=(14, 10))
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 1)))

    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        plt.scatter(umap_2d[cluster_mask, 0], umap_2d[cluster_mask, 1], 
                    c=[colors[i]], s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                    label=f"Cluster {i}")
        
        cluster_points = umap_2d[cluster_mask]
        center_x, center_y = cluster_points.mean(axis=0)
        top_keywords = ", ".join(cluster_keywords.get(i, [])[:3])
        
        plt.text(center_x, center_y, top_keywords, fontsize=11, weight='bold',
                 ha='center', va='center', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=3))

    if -1 in cluster_labels:
        noise_mask = cluster_labels == -1
        plt.scatter(umap_2d[noise_mask, 0], umap_2d[noise_mask, 1], 
                    c='black', s=30, alpha=0.4, marker='x', label='Ruido')

    plt.legend()
    plt.title("Visualización de Clusters con Palabras Clave")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64



'''
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
plt.show()'''
