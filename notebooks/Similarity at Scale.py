# Databricks notebook source
# MAGIC %md
# MAGIC This notebook shows how we are going to use User Defined Functions and Vector Search to scale our semantic similarity at scale.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd

import sys
sys.path.append('../src/')

# COMMAND ----------

df = spark.table('shm.requirements.pure_dataset')
display(df.limit(5))
print(f"{df.count()} rows")

# COMMAND ----------

def normalized_levenshtein(query, col_name):
  return 1 - (
    F.levenshtein(F.lit(query), F.col(col_name))
    /F.greatest(F.length(F.lit(query)),F.length(F.col(col_name)))
    )

query = 'The system shall create a single patient record for each patient.'
display(df.withColumn('similarity', normalized_levenshtein(query, 'sentence')))

# COMMAND ----------

# MAGIC %md
# MAGIC Doing all of these functions manually takes time, so we can omit them in favour of vector searches. A vector search requires a) an endpoint and b) an index with an embedding model. Each embedding model requires a single index.

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

# test our embedding deployments
embedding_deployments = {
  'gte-large': "databricks-gte-large-en",
  'bge-large': "databricks-bge-large-en",
  'bge-small': "bge_small_en_v1_5"
}

for embedding_name, embedding in embedding_deployments.items():
  response = deploy_client.predict(
    endpoint=embedding, 
    inputs={"input": ["Ich verstehe nur Bahnhof"]}
    )

  embeddings = [e['embedding'] for e in response.data]
  print(embedding_name)
  print(embeddings[0][0:3])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/index_creation.gif?raw=true" width="600px" style="float: right">
# MAGIC
# MAGIC You can view your endpoint on the [Vector Search Endpoints UI](#/setting/clusters/vector-search). Click on the endpoint name to see all indexes that are served by the endpoint.
# MAGIC
# MAGIC
# MAGIC ### Creating the Vector Search Index
# MAGIC
# MAGIC All we now have to do is to as Databricks to create the index. 
# MAGIC
# MAGIC Because it's a managed embedding index, we just need to specify the text column and our embedding foundation model (`GTE`).  Databricks will compute the embeddings for us automatically.
# MAGIC
# MAGIC This can be done using the API, or in a few clicks within the Unity Catalog Explorer menu.
# MAGIC

# COMMAND ----------

endpoint_name = "one-env-shared-endpoint-4"
from vs_utils import endpoint_exists, wait_for_vs_endpoint_to_be_ready

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if not endpoint_exists(vsc, endpoint_name):
    vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
    wait_for_vs_endpoint_to_be_ready(vsc, endpoint_name)

print(f"Endpoint named {endpoint_name} is ready.")

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE shm.requirements.pure_dataset SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC It takes a while to create indexes because we compute embeddings for every row. In this case, 3x11,000 rows.
# MAGIC ![](/Workspace/Users/scott.mckean@databricks.com/requirements_similarity/resources/vector_index_creation.png)

# COMMAND ----------

display(spark.table("shm.requirements.pure_dataset"))

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
from vs_utils import index_exists, wait_for_index_to_be_ready

source_table_fullname = f"shm.requirements.pure_dataset"

for embedding_name, embedding in embedding_deployments.items():

  vs_index_fullname = f"shm.requirements.pure_vs_index_{embedding_name.replace('-','_')}"

  if not index_exists(vsc, endpoint_name, vs_index_fullname):
    print(f"Creating index {vs_index_fullname} on endpoint {endpoint_name}...")
    vsc.create_delta_sync_index(
      endpoint_name=endpoint_name,
      index_name=vs_index_fullname,
      source_table_name=source_table_fullname,
      pipeline_type="TRIGGERED",
      primary_key="id",
      embedding_source_column='sentence', 
      embedding_model_endpoint_name='databricks-gte-large-en'
    )
    #Let's wait for the index to be ready and all our embeddings to be created and indexed
    wait_for_index_to_be_ready(vsc, endpoint_name, vs_index_fullname)
  else:
    #Trigger a sync to update our vs content with the new data saved in the table
    # wait_for_index_to_be_ready(vsc, endpoint_name, vs_index_fullname)
    vsc.get_index(endpoint_name, vs_index_fullname).sync()
    pass

  print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

df = spark.table('shm.requirements.coal_dataset')
display(df.limit(10))

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

query = 'The system shall provide the ability to capture medications entered by authorized users other than the prescriber.'

for embedding_name, embedding in embedding_deployments.items():
  print(embedding_name)
  results = vsc.get_index(endpoint_name, f"shm.requirements.pure_vs_index_{embedding_name.replace('-','_')}").similarity_search(
    query_text=query,
    columns=["sentence", "security"],
    num_results=3)
  display(results)

# COMMAND ----------

from langchain_community.vectorstores import DatabricksVectorSearch

vs_index = vsc.get_index(endpoint_name, f"shm.requirements.pure_vs_index_{embedding_name.replace('-','_')}")

vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column="sentence",
    columns=['id', 'security'],
).as_retriever(search_kwargs={'k':3, 'query_type':'ann'})

# COMMAND ----------

vector_search_as_retriever.invoke(query)
