# Assignment-5-FakeNews-Detection
# Fake News Detection using Spark MLlib

## Overview
This project implements a Fake News Detection system using Apache Spark MLlib. It leverages machine learning techniques to classify news articles as either **Fake** or **Real** based on their textual content.

## Project Structure
![image](https://github.com/user-attachments/assets/76031f43-326a-4b47-abd9-4f1292024b18)


## Setup Instructions

1. Install Apache Spark & PySpark  
   Ensure Spark and Python are installed. Then install PySpark:

   ```bash
   pip install pyspark
   ```
   
Install Dependencies

```
pip install faker
pip install pandas
```

Execution Steps
```bash
spark-submit src/task1.py
spark-submit src/task2.py
spark-submit src/task3.py
spark-submit src/task4.py
spark-submit src/task5.py
```
---
Model Details

Model Used: Logistic Regression

Feature Extraction: TF-IDF Vectorizer

Evaluation Metric: Accuracy (can be extended to F1-Score, Precision, Recall)

---
Output 

outputs will be clearly  seen in the output folder



