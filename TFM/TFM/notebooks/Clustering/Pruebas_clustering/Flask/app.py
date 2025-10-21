from flask import Flask, render_template
from UMAP_HDBSCAN_KeyBERT_3 import (
    cargar_datos,
    generar_clusters,
    extraer_keywords,
    preparar_resumen,
    generar_grafico_clusters
)

app = Flask(__name__)

@app.route("/")
def resultados():
    # Ruta al dataset que quieres analizar
    ruta_csv = "C:\\Users\\Usuario9\\OneDrive\\Documentos\\Master Big Data\\UPC - BIG DATA\\TFM\\TFM-main\\TFM\\TFM\\notebooks\\Clustering\\Pruebas_clustering\\Flask\\datasets\\DailyDialog.csv"  # Cambia esto según el archivo que quieras usar

    # Paso 1: Cargar y preparar datos
    df_sampled, top_emotions = cargar_datos(ruta_csv)

    # Paso 2: Generar embeddings y clustering
    model, umap_2d, cluster_labels = generar_clusters(df_sampled)

    # Paso 3: Extraer palabras clave
    cluster_keywords = extraer_keywords(df_sampled, cluster_labels, model)

    # Paso 4: Preparar resumen por cluster
    n_clusters, resumen_clusters = preparar_resumen(df_sampled, cluster_labels, cluster_keywords)

    # Paso 5: Generar gráfico en base64
    grafico_base64 = generar_grafico_clusters(umap_2d, cluster_labels, cluster_keywords)

    # Renderizar resultados en la plantilla
    return render_template("resultados.html",
                           resumen_clusters=resumen_clusters,
                           grafico_base64=grafico_base64,
                           n_clusters=n_clusters)

if __name__ == "__main__":
    app.run(debug=True)
