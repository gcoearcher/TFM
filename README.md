# noFanalyzers

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Treball de fi de master

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         nofanalyzers and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── nofanalyzers   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes nofanalyzers a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


# TFM
---------------------------
P1
- Data Ingestion: src/P1/data_ingestion  -> carpeta datasets/{fuente}/{formato}
- Landing Zone: src/P1/landing zone  -> carpeta delta_lake/{fuente}/{formato}
---------------------------
P2
- Trusted Zone: src/P2/trusted_zone -> MongoDB tfm-trusted-zone
- Explotation Zone: src/P2/explotation_zone -> MongoDB tfm_explotation_zone
----------------------------
Database - MongoDB 
Necesita de la creación de una base de datos en local
- Trusted Zone : mongodb://localhost:27017/tfm-trusted-zone.tf-idf
- Explotation Zone: mongodb://localhost:27017/tfm_explotation_zone.join_positive_emotions
----------------------------
Jars -> src/P2/trusted_zone/jars (contenido en el repositorio)
- Se ha necesitado de los jars en local para poder ejecutar los conectores de mongo con pyspark.
----------------------------
Versions and Libraries
pytho==3.9
pyspark==3.1.3
delta-spark==1.0.0
java==11.0 
-----------------------------
Hay limitadores en el codigo por si se quiere accelerar la performance.
ApiCall Kaggle -> Da error en la primera ejecución (la seguna ya funciona), por un tema de la api.authorization()