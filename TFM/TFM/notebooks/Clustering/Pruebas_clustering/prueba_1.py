import pandas as pd
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
from keybert import KeyBERT 

# Cargar dataset
df = pd.read_csv("C:\\Users\\Usuario9\\OneDrive\\Documentos\\Master Big Data\\UPC - BIG DATA\\TFM\\TFM-main\\TFM-main\\datasets\\Kaggle\\csv\\Emotion_classify_Data.csv", encoding="utf-8")
texts = df['text'].astype(str).tolist()

# Identificar columnas de emociones
emotion_columns = df.columns[df.columns.get_loc("Emotion"):]

# Función para extraer la primera emoción etiquetada
def get_main_emotion(row):
    emotions = row[emotion_columns]
    labels = emotions[emotions == 1].index.tolist()
    return labels[0] if labels else "neutral"

# Crear columna con la emoción principal
df["main_emotion"] = df.apply(get_main_emotion, axis=1)

# Filtrar las 4 emociones más comunes
top_emotions = df["main_emotion"].value_counts().head(4).index.tolist()
df_filtered = df[df["main_emotion"].isin(top_emotions)].copy()

print(f"Las 4 emociones principales son: {top_emotions}")
print(f"Número total de textos por emoción:")
print(df_filtered["main_emotion"].value_counts())

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

print(f"\nNúmero de textos después del muestreo:")
print(df_sampled["main_emotion"].value_counts())
print(f"Total de textos para análisis: {len(df_sampled)}")

# Usar df_sampled en lugar de df_filtered para el resto del análisis
df_filtered = df_sampled

# Generar embeddings solo para los datos filtrados
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(df_filtered["text"].tolist(), show_progress_bar=True)

# Reducir dimensiones con UMAP (parámetros optimizados para HDBSCAN)
umap_reducer = umap.UMAP(
    n_components=2, 
    metric="cosine",
    n_neighbors=15,    # Equilibrio para HDBSCAN
    min_dist=0.1,      # Permitir clusters más densos
    spread=1.0,        # Balance para HDBSCAN
    random_state=42
)
umap_2d = umap_reducer.fit_transform(embeddings)

# Añadir coordenadas UMAP al dataframe filtrado
df_filtered = df_filtered.copy()
df_filtered["x"] = umap_2d[:, 0]
df_filtered["y"] = umap_2d[:, 1]

# Aplicar HDBSCAN clustering (descubrimiento automático de clusters)
print("\nAplicando HDBSCAN...")
hdb_clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15,        # Mínimo 15 puntos por cluster
    metric='euclidean',         # Métrica en el espacio UMAP
    cluster_selection_method='eom'  # Excess of Mass
)

hdbscan_labels = hdb_clusterer.fit_predict(umap_2d)
df_filtered["hdbscan_cluster"] = hdbscan_labels

# Información sobre clusters HDBSCAN
n_clusters_hdb = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
n_noise = list(hdbscan_labels).count(-1)

print(f"HDBSCAN encontró {n_clusters_hdb} clusters")
print(f"Número de puntos de ruido: {n_noise}")

print(f"\n📊 HDBSCAN - {n_clusters_hdb} clusters + ruido:")
if n_clusters_hdb > 0:
    for i in range(n_clusters_hdb):
        cluster_data = df_filtered[df_filtered["hdbscan_cluster"] == i]
        emotion_dist = cluster_data["main_emotion"].value_counts()
        print(f"  Cluster {i}: {len(cluster_data)} puntos")
        print(f"    Emociones: {emotion_dist.to_dict()}")

if n_noise > 0:
    noise_data = df_filtered[df_filtered["hdbscan_cluster"] == -1]
    emotion_dist_noise = noise_data["main_emotion"].value_counts()
    print(f"  Ruido: {n_noise} puntos")
    print(f"    Emociones en ruido: {emotion_dist_noise.to_dict()}")


# KeyBERT para palabras clave
kw_model = KeyBERT(model=model)
cluster_keywords = {}

for cluster_id in np.unique(hdbscan_labels):
    if cluster_id == -1:  # Ruido
        continue
    cluster_docs = [df_filtered["text"].iloc[i] for i in range(len(hdbscan_labels)) if hdbscan_labels[i] == cluster_id]
    cluster_text = " ".join(cluster_docs)
    keywords = kw_model.extract_keywords(cluster_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    cluster_keywords[cluster_id] = [kw[0] for kw in keywords]

plt.figure(figsize=(10, 8))
unique_labels = np.unique(hdbscan_labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    mask = hdbscan_labels == label
    plt.scatter(umap_2d[mask, 0], umap_2d[mask, 1], 
                color=color.reshape(1,-1), label=f"Cluster {label}", alpha=0.6)

plt.legend()
plt.title("Clusters con UMAP+HDBSCAN")
plt.show()
    

# Mostrar resultados
for cluster, words in cluster_keywords.items():
    print(f"Cluster {cluster}: {words}")


# Crear visualización comparativa
fig, (ax_hdb, ax_kw) = plt.subplots(1, 2, figsize=(20, 8))

# ====== GRÁFICO HDBSCAN ======
colors_hdb = plt.cm.Set1(np.linspace(0, 1, max(n_clusters_hdb, 1)))
if n_clusters_hdb > 0:
    for i in range(n_clusters_hdb):
        cluster_data = df_filtered[df_filtered["hdbscan_cluster"] == i]
        ax_hdb.scatter(
            cluster_data["x"], cluster_data["y"],
            c=[colors_hdb[i]], label=f"Cluster {i} ({len(cluster_data)})",
            s=50, alpha=0.7, edgecolors='black', linewidth=0.5
        )

# Puntos de ruido en negro
if n_noise > 0:
    noise_data = df_filtered[df_filtered["hdbscan_cluster"] == -1]
    ax_hdb.scatter(
        noise_data["x"], noise_data["y"],
        c='black', label=f"Ruido ({n_noise})",
        s=30, alpha=0.5, marker='x'
    )

ax_hdb.set_title(f"HDBSCAN ({n_clusters_hdb} clusters)", fontsize=14, fontweight='bold')
ax_hdb.set_xlabel("UMAP Dimensión 1")
ax_hdb.set_ylabel("UMAP Dimensión 2")
ax_hdb.legend()
ax_hdb.grid(True, alpha=0.3)

# ====== GRÁFICO CON KEYWORDS ======
if n_clusters_hdb > 0:
    for i in range(n_clusters_hdb):
        cluster_data = df_filtered[df_filtered["hdbscan_cluster"] == i]
        ax_kw.scatter(
            cluster_data["x"], cluster_data["y"],
            c=[colors_hdb[i]], s=50, alpha=0.7, edgecolors='black', linewidth=0.5
        )
        # Calcular centro del cluster para colocar las keywords
        center_x = cluster_data["x"].mean()
        center_y = cluster_data["y"].mean()
        keywords = ", ".join(cluster_keywords.get(i, [])[:3])  # solo top 3
        ax_kw.text(center_x, center_y, keywords, fontsize=10, weight='bold',
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

ax_kw.set_title("Clusters con palabras clave (KeyBERT)", fontsize=14, fontweight='bold')
ax_kw.set_xlabel("UMAP Dimensión 1")
ax_kw.set_ylabel("UMAP Dimensión 2")
ax_kw.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()