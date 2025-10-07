from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Test").getOrCreate()
print("Versión de Spark:", spark.version)