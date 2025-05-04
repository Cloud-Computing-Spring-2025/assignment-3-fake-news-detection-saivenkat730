from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("FakeNewsDetection").getOrCreate()
df = spark.read.csv("data/fake_news_sample.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("news_data")

df.show(5)
print(f"Total articles: {df.count()}")
df.select("label").distinct().show()

df.coalesce(1).write.csv("output/task1_output.csv", header=True, mode="overwrite")
