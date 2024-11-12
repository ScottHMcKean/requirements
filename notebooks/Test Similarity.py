# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import nltk
nltk.download('punkt')

# COMMAND ----------

import sys
sys.path.append('../src/')
sys.path.append('src/')

# COMMAND ----------

from similarity import RequirementSimilarity
similarity = RequirementSimilarity(embedding_models={
    'minilm': 'all-MiniLM-L6-v2',
    'codebert': 'microsoft/codebert-base',
    'bge': 'BAAI/bge-large-en-v1.5',
    'lt-industry': 'dell-research-harvard/lt-un-data-fine-industry-en'
    })

# COMMAND ----------

text1 = "The system shall provide user authentication via email and password"
text2 = "Users must be able to log in using their email address and password"

# COMMAND ----------

similarity.get_combined_similarity(text1, text2)