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
