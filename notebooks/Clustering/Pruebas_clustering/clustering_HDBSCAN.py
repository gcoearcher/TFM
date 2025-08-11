import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
from sklearn.cluster import KMeans

# Cargar el archivo
df = pd.read_csv("C:\\Users\\Usuario9\\OneDrive\\Documentos\\Master Big Data\\UPC - BIG DATA\\TFM\\TFM-main\\TFM-main\\datasets\\Kaggle\\csv\\GoEmotions\\goemotions_1.csv", encoding="utf-8")

# Identificar columnas de emociones
emotion_columns = df.columns[df.columns.get_loc("admiration"):]

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
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
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
    min_cluster_size=30,        # Mínimo 30 puntos por cluster
    min_samples=10,             # Mínimo 10 vecinos para ser punto core
    cluster_selection_epsilon=0.5,  # Threshold para fusión de clusters
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

# Aplicar K-Means para comparación (forzar 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(umap_2d)
df_filtered["kmeans_cluster"] = kmeans_labels

# Mostrar información detallada de ambos métodos
print("\n" + "="*50)
print("COMPARACIÓN DE MÉTODOS DE CLUSTERING")
print("="*50)

# Información HDBSCAN
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

# Información K-Means
print(f"\n📊 K-MEANS - 4 clusters forzados:")
for i in range(4):
    cluster_data = df_filtered[df_filtered["kmeans_cluster"] == i]
    emotion_dist = cluster_data["main_emotion"].value_counts()
    print(f"  Cluster {i}: {len(cluster_data)} puntos")
    print(f"    Emociones: {emotion_dist.to_dict()}")

# Crear visualización comparativa
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

# Gráfico 1: Emociones originales
colors_emotion = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
for i, emotion in enumerate(top_emotions):
    emotion_data = df_filtered[df_filtered["main_emotion"] == emotion]
    ax1.scatter(
        emotion_data["x"], emotion_data["y"], 
        c=colors_emotion[i], label=f"{emotion}", 
        s=50, alpha=0.7, edgecolors='black', linewidth=0.5
    )

ax1.set_title("Emociones Originales", fontsize=14, fontweight='bold')
ax1.set_xlabel("UMAP Dimensión 1")
ax1.set_ylabel("UMAP Dimensión 2")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: HDBSCAN clusters
colors_hdb = plt.cm.Set1(np.linspace(0, 1, max(n_clusters_hdb, 1)))
if n_clusters_hdb > 0:
    for i in range(n_clusters_hdb):
        cluster_data = df_filtered[df_filtered["hdbscan_cluster"] == i]
        ax2.scatter(
            cluster_data["x"], cluster_data["y"],
            c=[colors_hdb[i]], label=f"Cluster {i} ({len(cluster_data)})",
            s=50, alpha=0.7, edgecolors='black', linewidth=0.5
        )

# Puntos de ruido en negro
if n_noise > 0:
    noise_data = df_filtered[df_filtered["hdbscan_cluster"] == -1]
    ax2.scatter(
        noise_data["x"], noise_data["y"],
        c='black', label=f"Ruido ({n_noise})",
        s=30, alpha=0.5, marker='x'
    )

ax2.set_title(f"HDBSCAN ({n_clusters_hdb} clusters automáticos)", fontsize=14, fontweight='bold')
ax2.set_xlabel("UMAP Dimensión 1")
ax2.set_ylabel("UMAP Dimensión 2")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gráfico 3: K-Means clusters
colors_kmeans = ['#FF0000', '#0000FF', '#00FF00', '#FFA500']
for i in range(4):
    cluster_data = df_filtered[df_filtered["kmeans_cluster"] == i]
    ax3.scatter(
        cluster_data["x"], cluster_data["y"],
        c=colors_kmeans[i], label=f"Cluster {i} ({len(cluster_data)})",
        s=50, alpha=0.7, edgecolors='black', linewidth=0.5
    )

# Centroides K-Means
centroids = kmeans.cluster_centers_
ax3.scatter(
    centroids[:, 0], centroids[:, 1],
    c='white', marker='x', s=200, linewidths=3, label='Centroides'
)

ax3.set_title("K-Means (4 clusters forzados)", fontsize=14, fontweight='bold')
ax3.set_xlabel("UMAP Dimensión 1")
ax3.set_ylabel("UMAP Dimensión 2")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calcular métricas de calidad
from sklearn.metrics import adjusted_rand_score, silhouette_score

print("\n" + "="*50)
print("MÉTRICAS DE CALIDAD DE CLUSTERING")
print("="*50)

# Silhouette Score (mayor es mejor)
if n_clusters_hdb > 1:
    # Solo calcular para puntos que NO son ruido
    non_noise_mask = hdbscan_labels != -1
    if np.sum(non_noise_mask) > 1:
        sil_hdb = silhouette_score(umap_2d[non_noise_mask], hdbscan_labels[non_noise_mask])
        print(f"Silhouette Score HDBSCAN: {sil_hdb:.3f}")
    else:
        print("Silhouette Score HDBSCAN: No calculable (muy pocos clusters)")
else:
    print("Silhouette Score HDBSCAN: No calculable (menos de 2 clusters)")

sil_kmeans = silhouette_score(umap_2d, kmeans_labels)
print(f"Silhouette Score K-Means: {sil_kmeans:.3f}")

# Calcular pureza para cada método
def calculate_purity(true_labels, cluster_labels):
    """Calcula la pureza de clustering"""
    total = len(true_labels)
    purity_sum = 0
    
    for cluster in set(cluster_labels):
        if cluster == -1:  # Ignorar ruido en HDBSCAN
            continue
        cluster_mask = cluster_labels == cluster
        cluster_true_labels = true_labels[cluster_mask]
        if len(cluster_true_labels) > 0:
            most_common = np.bincount(pd.factorize(cluster_true_labels)[0]).max()
            purity_sum += most_common
    
    return purity_sum / total if total > 0 else 0

# Convertir emociones a números para el cálculo
emotion_to_num = {emotion: i for i, emotion in enumerate(top_emotions)}
true_labels = df_filtered["main_emotion"].map(emotion_to_num).values

if n_clusters_hdb > 0:
    purity_hdb = calculate_purity(true_labels, hdbscan_labels)
    print(f"Pureza HDBSCAN: {purity_hdb:.3f}")
else:
    print("Pureza HDBSCAN: No calculable")

purity_kmeans = calculate_purity(true_labels, kmeans_labels)
print(f"Pureza K-Means: {purity_kmeans:.3f}")

# Guardar resultados
df_filtered.to_csv("comparacion_clustering_hdbscan_vs_kmeans.csv", index=False)
print(f"\nResultados guardados en 'comparacion_clustering_hdbscan_vs_kmeans.csv'")
print(f"Se analizaron {len(df_filtered)} textos comparando HDBSCAN vs K-Means")