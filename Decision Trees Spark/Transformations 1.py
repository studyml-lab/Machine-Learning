from pyspark.ml.feature import OneHotEncoder, StringIndexer

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
print(spark)

df = spark.createDataFrame([(0, "a"),
    (1, "b"),
    (2, "c"),
    (3, "a"),
    (4, "a"),
    (5, "c")], ["id", "category"])

stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)

encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
encoded = encoder.transform(indexed)
encoded.show()