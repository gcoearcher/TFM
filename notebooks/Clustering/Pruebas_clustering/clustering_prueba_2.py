import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from keybert import KeyBERT
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar CSV
df = pd.read_csv("C:\\Users\\Usuario9\\OneDrive\\Documentos\\Master Big Data\\UPC - BIG DATA\\TFM\\TFM-main\\TFM-main\\datasets\\Kaggle\\csv\\GoEmotions\\goemotions_1.csv", encoding="utf-8")

# Seleccionar las columnas de emociones
emotion_cols = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise'
]

# Filtrar textos con al menos una emoción (sin contar neutral)
df["has_emotion"] = df[emotion_cols].sum(axis=1) > 0
df_emotions = df[df["has_emotion"] == True].copy()

# Embeddings con Sentence Transformers
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(df_emotions["text"].tolist(), show_progress_bar=True)

# Reducir dimensiones con UMAP
umap_model = umap.UMAP(n_neighbors=10, n_components=2, metric="cosine")
reduced_embeddings = umap_model.fit_transform(embeddings)

# Agrupar con HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=30, metric="euclidean")
cluster_labels = clusterer.fit_predict(reduced_embeddings)

# Añadir columnas al DataFrame
df_emotions["cluster"] = cluster_labels

# Extraer palabras clave por cluster con KeyBERT
kw_model = KeyBERT(model=model)

print("\n=== Palabras clave por cluster ===")
for c in sorted(df_emotions["cluster"].unique()):
    if c == -1:
        continue  # Excluir outliers
    cluster_texts = df_emotions[df_emotions["cluster"] == c]["text"].tolist()
    joined_text = " ".join(cluster_texts)
    keywords = kw_model.extract_keywords(joined_text, top_n=8)
    print(f"\nCluster {c}: {[kw[0] for kw in keywords]}")

# Visualizar en 2D con UMAP
umap_2d = umap.UMAP(n_components=2, metric="cosine").fit_transform(embeddings)
df_emotions["x"] = umap_2d[:, 0]
df_emotions["y"] = umap_2d[:, 1]

# Filtrar solo clusters válidos
df_clusters = df_emotions[df_emotions["cluster"] != -1]

# Plot con seaborn
plt.figure(figsize=(10, 8))
palette = sns.color_palette("Spectral", n_colors=df_clusters["cluster"].nunique())
sns.scatterplot(
    data=df_clusters,
    x="x", y="y", hue="cluster",
    palette=palette, legend="full", s=40, alpha=0.8
)
plt.title("Clustering de 4 emociones principales")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
