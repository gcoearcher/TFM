import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

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
# Tomar una muestra equilibrada de cada emoción (máximo 200 textos por emoción)
sample_size_per_emotion = 200

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

# Reducir dimensiones con UMAP (parámetros optimizados para mejor separación)
umap_reducer = umap.UMAP(
    n_components=2, 
    metric="cosine",
    n_neighbors=5,     # Muy reducido para formar clusters más compactos
    min_dist=0.5,      # Aumentado para mayor separación entre clusters
    spread=2.0,        # Aumentado para dispersar más los clusters
    random_state=42
)
umap_2d = umap_reducer.fit_transform(embeddings)

# Añadir coordenadas UMAP al dataframe filtrado
df_filtered = df_filtered.copy()  # Evitar warnings
df_filtered["x"] = umap_2d[:, 0]
df_filtered["y"] = umap_2d[:, 1]

# Aplicar K-Means clustering para forzar 4 clusters bien definidos
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(umap_2d)
df_filtered["cluster"] = cluster_labels

# Mostrar información sobre los clusters
print("\nInformación de clusters:")
for i in range(4):
    cluster_data = df_filtered[df_filtered["cluster"] == i]
    emotion_dist = cluster_data["main_emotion"].value_counts()
    print(f"Cluster {i}: {len(cluster_data)} puntos")
    print(f"  Emociones dominantes: {emotion_dist.head(2).to_dict()}")

# Obtener centroides de los clusters
centroids = kmeans.cluster_centers_

# Crear el gráfico con clusters definidos por K-Means
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Gráfico 1: Coloreado por emoción original
colors_emotion = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # Rojo, Azul, Verde, Naranja
for i, emotion in enumerate(top_emotions):
    emotion_data = df_filtered[df_filtered["main_emotion"] == emotion]
    ax1.scatter(
        emotion_data["x"], 
        emotion_data["y"], 
        c=colors_emotion[i], 
        label=f"{emotion} ({len(emotion_data)} textos)", 
        s=60,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.8
    )

ax1.set_title("Clustering por UMAP", fontsize=14, fontweight='bold')
ax1.set_xlabel("UMAP Dimensión 1", fontsize=12)
ax1.set_ylabel("UMAP Dimensión 2", fontsize=12)
ax1.legend(title="Emociones", bbox_to_anchor=(1.05, 1), loc="upper left")
ax1.grid(True, alpha=0.3)

# Gráfico 2: Coloreado por clusters de K-Means
colors_cluster = ['#FF0000', '#0000FF', '#00FF00', '#FFA500']  # Colores distintivos para clusters
for i in range(4):
    cluster_data = df_filtered[df_filtered["cluster"] == i]
    ax2.scatter(
        cluster_data["x"], 
        cluster_data["y"], 
        c=colors_cluster[i], 
        label=f"Cluster {i} ({len(cluster_data)} puntos)", 
        s=60,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.8
    )

# Añadir centroides de los clusters
ax2.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    c='black', 
    marker='x', 
    s=300, 
    linewidths=3,
    label='Centroides'
)

ax2.set_title("Clustering por K-Means", fontsize=14, fontweight='bold')
ax2.set_xlabel("UMAP Dimensión 1", fontsize=12)
ax2.set_ylabel("UMAP Dimensión 2", fontsize=12)
ax2.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Análisis de la correspondencia entre emociones y clusters
print("\nAnálisis de correspondencia Emoción-Cluster:")
correspondence_matrix = pd.crosstab(df_filtered["main_emotion"], df_filtered["cluster"], margins=True)
print(correspondence_matrix)

# Calcular pureza de clusters (qué tan homogéneos son los clusters)
print("\nPureza de cada cluster:")
for i in range(4):
    cluster_data = df_filtered[df_filtered["cluster"] == i]
    if len(cluster_data) > 0:
        dominant_emotion = cluster_data["main_emotion"].value_counts().iloc[0]
        purity = dominant_emotion / len(cluster_data) * 100
        dominant_emotion_name = cluster_data["main_emotion"].value_counts().index[0]
        print(f"Cluster {i}: {purity:.1f}% de pureza (dominado por '{dominant_emotion_name}')")

# Opcional: Guardar las coordenadas UMAP y clusters para análisis posteriores
df_filtered.to_csv("emociones_con_clusters_kmeans.csv", index=False)
print(f"\nArchivo con coordenadas UMAP y clusters guardado como 'emociones_con_clusters_kmeans.csv'")
print(f"Se analizaron {len(df_filtered)} textos en total con 4 clusters forzados")