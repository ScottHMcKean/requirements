# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import nltk
nltk.download('punkt')

# COMMAND ----------

import sys
sys.path.append('../src/')

# COMMAND ----------

from similarity import RequirementSimilarity
similarity = RequirementSimilarity()

# COMMAND ----------

query = "The system shall provide user authentication via email and password"
requirements = [
    "Users must be able to log in using their email address and password",
    "The system should implement secure user authentication",
    "Data must be encrypted during transmission",
    "The application should have a responsive design"
]

# COMMAND ----------

ranked_reqs = []
for req in requirements:
    score, detailed_scores = similarity.get_combined_similarity(query, req)
    ranked_reqs.append((req, score))

# COMMAND ----------

detailed_scores

# COMMAND ----------

ranked_reqs

# COMMAND ----------


