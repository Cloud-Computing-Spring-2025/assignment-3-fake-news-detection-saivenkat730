from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression

# Start Spark session
spark = SparkSession.builder.appName("FakeNews-Task4").getOrCreate()

# Load tokenized data (filtered_words as string)
df = spark.read.csv("output/task2_output.csv", header=True, inferSchema=True)

# Convert to array
df = df.withColumn("filtered_words", split(df["filtered_words"], " "))

# HashingTF
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
df_tf = hashingTF.transform(df)

# IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)

# Label indexing
indexer = StringIndexer(inputCol="label", outputCol="label_index")
df_final = indexer.fit(df_tfidf).transform(df_tfidf)

# Split data
train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

# Train Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_data)

# Predict
predictions = model.transform(test_data)

# Select and save results
predictions.select("id", "title", "label_index", "prediction") \
    .coalesce(1).write.csv("output/task4_output.csv", header=True, mode="overwrite")

# Stop session
spark.stop()