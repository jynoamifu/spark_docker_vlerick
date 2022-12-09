# Import the necessary modules
from pyspark.sql import SparkSession
from pyspark import SparkConf
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print(os.environ['AWS_SECRET_ACCESS_KEY'])
print('='*80)

# Create a SparkSession object
BUCKET = "dmacademy-course-assets"
KEY1 = "vlerick/pre_release.csv"
KEY2 = "vlerick/after_release.csv"

config = {
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
    "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
}
conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Read the CSV file from the S3 bucket
variables = spark.read.csv(f"s3a://{BUCKET}/{KEY1}", header=True)
target = spark.read.csv(f"s3a://{BUCKET}/{KEY2}", header=True)

variables.show()
target.show()

# Create Pandas DataaFrame
variables = variables.toPandas()
target = target.toPandas()

# Inner Join the two tables
df_raw = pd.merge(variables, target, how='inner', on='movie_title')
print(df_raw.head())

