from flask import Flask, render_template
from clustering_prueba_3 import (
    cargar_datos,
    generar_embeddings,
    aplicar_kmeans,
    preparar_resumen,
    generar_grafico_clusters
)


app = Flask(__name__)

@app.route("/")
def resultados():
    ruta_csv = "C:\\Users\\Usuario9\\OneDrive\\Documentos\\Master Big Data\\UPC - BIG DATA\\TFM\\TFM-main\\TFM\\TFM\\notebooks\\Clustering\\Pruebas_clustering\\Flask\\datasets\\goemotions_1.csv"  # Cambia esto según el dataset
    df_sampled, top_emotions = cargar_datos(ruta_csv)
    umap_2d = generar_embeddings(df_sampled)
    df_clustered, centroids = aplicar_kmeans(df_sampled, umap_2d)
    resumen_clusters = preparar_resumen(df_clustered)
    grafico_base64 = generar_grafico_clusters(df_clustered, top_emotions, centroids)

    return render_template("resultados.html",
                           resumen_clusters=resumen_clusters,
                           grafico_base64=grafico_base64)

if __name__ == "__main__":
    app.run(debug=False)

