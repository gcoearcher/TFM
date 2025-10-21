import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from keybert import KeyBERT
import matplotlib.pyplot as plt
import seaborn as sns

with open("/../TFM-main/datasets/paper+reviews/reviews.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraer los comentarios
reviews = []
for paper in data["paper"]:
    for review in paper["review"]:
        reviews.append({
            "paper_id": paper["id"],
            "review_id": review["id"],
            "text": review["text"],
            "confidence": review["confidence"],
            "evaluation": review["evaluation"]
        })

df = pd.DataFrame(reviews)

# Cargar el modelo de SentenceTransformer y generar embeddings
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

# Reducir la dimensionalidad de los embeddings con UMAP
umap_model = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine")
reduced_embeddings = umap_model.fit_transform(embeddings)

# Agrupar los embeddings reducidos con HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric="euclidean")
cluster_labels = clusterer.fit_predict(reduced_embeddings)

df["cluster"] = cluster_labels

# Extraer palabras clave de cada cluster usando KeyBERT
kw_model = KeyBERT(model=model)

for c in sorted(df["cluster"].unique()):
    cluster_texts = df[df["cluster"] == c]["text"].tolist()
    joined_text = " ".join(cluster_texts)
    if c != -1 and joined_text.strip():
        keywords = kw_model.extract_keywords(joined_text, top_n=10)
        print(f"\nCluster {c}: {[kw[0] for kw in keywords]}")


# Visualizar los clusters usando UMAP en 2D
umap_2d = umap.UMAP(n_components=2, metric="cosine").fit_transform(embeddings)
df["x"] = umap_2d[:,0]
df["y"] = umap_2d[:,1]

# Filtrar puntos sin ruido
df_clean = df[df["cluster"] != -1]

# Graficar con seaborn
plt.figure(figsize=(10, 8))
palette = sns.color_palette("tab10", len(df_clean["cluster"].unique()))

for i, cluster in enumerate(sorted(df_clean["cluster"].unique())):
    cluster_data = df_clean[df_clean["cluster"] == cluster]
    plt.scatter(cluster_data["x"], cluster_data["y"],
                label=f"Cluster {cluster}",
                s=30,
                alpha=0.7)

plt.title("Clustering de comentarios por emoción (sin ruido)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(title="Clusters")
plt.grid(True)
plt.tight_layout()
plt.show()
