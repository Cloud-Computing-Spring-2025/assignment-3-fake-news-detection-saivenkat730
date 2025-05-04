from pyspark.sql import SparkSession
from pyspark.sql.functions import split, concat_ws, col
from pyspark.ml.feature import HashingTF, IDF, StringIndexer

# Initialize Spark session
spark = SparkSession.builder.appName("FakeNews-Task3").getOrCreate()

# Load tokenized output (where filtered_words is a space-separated string)
df = spark.read.csv("output/task2_output.csv", header=True, inferSchema=True)

# Convert filtered_words back to array
df = df.withColumn("filtered_words", split(df["filtered_words"], " "))

# Apply HashingTF
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
featurized_df = hashingTF.transform(df)

# Apply IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(featurized_df)
rescaled_df = idfModel.transform(featurized_df)

# Index the labels (FAKE → 0, REAL → 1)
indexer = StringIndexer(inputCol="label", outputCol="label_index")
final_df = indexer.fit(rescaled_df).transform(rescaled_df)

# Convert filtered_words array to string for saving
final_df = final_df.withColumn("filtered_words", concat_ws(" ", "filtered_words"))

# Convert features vector to string for saving
final_df = final_df.withColumn("features", col("features").cast("string"))

# Select final output columns
output_df = final_df.select("id", "filtered_words", "features", "label_index")

# Save to single CSV file
output_df.coalesce(1).write.csv("output/task3_output.csv", header=True, mode="overwrite")

# Stop Spark session
spark.stop()