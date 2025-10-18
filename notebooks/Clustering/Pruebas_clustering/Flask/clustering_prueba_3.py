from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')  # backend sin GUI


# 🔧 1. Cargar y preparar datos
def cargar_datos(ruta_csv, sample_size_per_emotion=200, top_n_emotions=4):
    import pandas as pd

    # Cargar el CSV
    df = pd.read_csv(ruta_csv, encoding="utf-8")

    # Normalizar nombres de columnas
    columnas = {col.lower(): col for col in df.columns}
    texto_col = next((columnas[c] for c in ["text", "comment"] if c in columnas), None)
    emocion_col = next((columnas[c] for c in ["emotion", "sentiment"] if c in columnas), None)

    # Verificar columnas obligatorias
    if texto_col is None or emocion_col is None:
        raise ValueError("El dataset debe contener una columna de texto ('text' o 'comment') y una de emoción ('emotion' o 'sentiment').")

    # Limpiar texto y emoción
    df[texto_col] = df[texto_col].astype(str).str.strip()
    df["main_emotion"] = df[emocion_col].astype(str).str.strip()

    # Filtrar emociones más comunes
    top_emotions = df["main_emotion"].value_counts().head(top_n_emotions).index.tolist()
    df_filtered = df[df["main_emotion"].isin(top_emotions)].copy()

    # Muestreo equilibrado
    df_sampled = pd.DataFrame()
    for emotion in top_emotions:
        emotion_data = df_filtered[df_filtered["main_emotion"] == emotion]
        if len(emotion_data) > sample_size_per_emotion:
            emotion_sample = emotion_data.sample(n=sample_size_per_emotion, random_state=42)
        else:
            emotion_sample = emotion_data
        df_sampled = pd.concat([df_sampled, emotion_sample], ignore_index=True)

    return df_sampled, top_emotions


# 🔧 2. Embeddings y reducción dimensional
def generar_embeddings(df_sampled):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df_sampled["text"].tolist(), show_progress_bar=True)

    umap_reducer = umap.UMAP(
        n_components=2, 
        metric="cosine",
        n_neighbors=5,
        min_dist=0.5,
        spread=2.0,
        random_state=42
    )
    umap_2d = umap_reducer.fit_transform(embeddings)
    return umap_2d

# 🔧 3. Clustering con K-Means
def aplicar_kmeans(df_sampled, umap_2d, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(umap_2d)
    df_sampled["x"] = umap_2d[:, 0]
    df_sampled["y"] = umap_2d[:, 1]
    df_sampled["cluster"] = cluster_labels
    centroids = kmeans.cluster_centers_
    return df_sampled, centroids

# 🔧 4. Resumen por cluster
def preparar_resumen(df_sampled):
    resumen_clusters = []
    for i in sorted(df_sampled["cluster"].unique()):
        cluster_data = df_sampled[df_sampled["cluster"] == i]
        emotion_dist = cluster_data["main_emotion"].value_counts().to_dict()
        resumen_clusters.append({
            "id": i,
            "size": len(cluster_data),
            "emotions": emotion_dist
        })
    return resumen_clusters

# 🔧 5. Visualización en base64
def generar_grafico_clusters(df_sampled, top_emotions, centroids):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    colors = ['#FF6B6B', "#B84ECD", '#45B7D1', "#A0FF7AB3"]
    emotion_colors = dict(zip(top_emotions, colors))

    for i, emotion in enumerate(top_emotions):
        emotion_data = df_sampled[df_sampled["main_emotion"] == emotion]
        ax1.scatter(
            emotion_data["x"], emotion_data["y"],
            c=colors[i], label=f"{emotion} ({len(emotion_data)} textos)",
            s=60, alpha=0.9, edgecolors='black', linewidth=0.8
        )

    ax1.set_title("Clustering de emociones por UMAP", fontsize=16, fontweight='bold')
    ax1.set_xlabel("UMAP Dimensión 1")
    ax1.set_ylabel("UMAP Dimensión 2")
    ax1.legend(title="Emociones", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    colors_cluster = ['#FF0000', '#0000FF', '#00FF00', '#FFA500']
    for i in range(4):
        cluster_data = df_sampled[df_sampled["cluster"] == i]
        ax2.scatter(
            cluster_data["x"], cluster_data["y"],
            c=colors_cluster[i], label=f"Cluster {i} ({len(cluster_data)} puntos)",
            s=60, alpha=0.7, edgecolors='black', linewidth=0.8
        )

    ax2.scatter(
        centroids[:, 0], centroids[:, 1],
        c='white', marker='x', s=300, linewidths=3, label='Centroides'
    )

    ax2.set_title("Clustering por K-Means", fontsize=14, fontweight='bold')
    ax2.set_xlabel("UMAP Dimensión 1")
    ax2.set_ylabel("UMAP Dimensión 2")
    ax2.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64

'''
# Opcional: Guardar las coordenadas UMAP para análisis posteriores
df_sampled.to_csv("emociones_con_coordenadas_umap_muestra.csv", index=False)
print(f"\nArchivo con coordenadas UMAP guardado como 'emociones_con_coordenadas_umap_muestra.csv'")
'''