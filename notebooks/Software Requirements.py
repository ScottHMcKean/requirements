# Databricks notebook source
import pandas as pd

# Set up default logger
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
logging.getLogger("py4j.client_server").setLevel("ERROR")

# COMMAND ----------

table_name = 'shm.requirements.coal_dataset'

try:
    spark.read.table(table_name)
    log.info(f"Table {table_name} already exists.")
except Exception:
    df = pd.read_csv("hf://datasets/nguyenminh871/software_requirements/Coalation task.csv")
    df.columns = ['id', 'python_task', 'contract_task', 'java_task']
    spark_df = spark.createDataFrame(df)
    spark_df.write.format("delta").saveAsTable(table_name)
    log.info(f"Table {table_name} created and data loaded.")

# COMMAND ----------

req_coal = spark.read.table(table_name)
req_pure = spark.read.table('shm.requirements.pure_dataset')

# COMMAND ----------

display(req_coal)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM shm.requirements.pure_dataset
# MAGIC WHERE security = 1
# MAGIC AND reliability = 1

# COMMAND ----------

req_pure_pd.reliability.unique()

# COMMAND ----------

display(req_coal)

# COMMAND ----------


