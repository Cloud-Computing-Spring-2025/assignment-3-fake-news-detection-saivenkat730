from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("FakeNewsDetection").getOrCreate()
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

df = spark.read.csv("output/task4_output.csv", header=True, inferSchema=True)

evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(df)
f1 = evaluator_f1.evaluate(df)

results = spark.createDataFrame([("Accuracy", accuracy), ("F1 Score", f1)], ["Metric", "Value"])
results.write.csv("output/task5_output.csv", header=True)