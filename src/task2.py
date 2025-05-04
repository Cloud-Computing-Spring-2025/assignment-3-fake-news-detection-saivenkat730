from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# Initialize Spark session
spark = SparkSession.builder.appName("FakeNews-Task2").getOrCreate()

# Load dataset
df = spark.read.csv("data/fake_news_sample.csv", header=True, inferSchema=True)

# Lowercase the text column
df = df.withColumn("text", lower(col("text")))

# Tokenize the text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
words_df = tokenizer.transform(df)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned_df = remover.transform(words_df).select("id", "title", "filtered_words", "label")

# Convert array of words to string so it can be written to CSV
final_df = cleaned_df.withColumn("filtered_words", concat_ws(" ", "filtered_words"))

# Save to output as a single CSV file
final_df.coalesce(1).write.csv("output/task2_output.csv", header=True, mode="overwrite")

# Stop Spark session
spark.stop()