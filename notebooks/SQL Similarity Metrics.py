# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION shm.requirements.normalized_levenshtein_distance(s1 STRING, s2 STRING)
# MAGIC RETURNS DOUBLE
# MAGIC RETURN 
# MAGIC   CASE 
# MAGIC     WHEN LENGTH(s1) = 0 AND LENGTH(s2) = 0 THEN 0.0
# MAGIC     ELSE CAST(levenshtein(s1, s2) AS DOUBLE) / CAST(GREATEST(LENGTH(s1), LENGTH(s2)) AS DOUBLE)
# MAGIC   END;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT shm.requirements.normalized_levenshtein_distance(
# MAGIC   "The system shall provide user authentication via email and password",
# MAGIC   "Users must be able to log in using their email address and password"
# MAGIC ) AS levenshtein_distance

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ai_similarity(
# MAGIC   "The system shall provide user authentication via email and password", 
# MAGIC   "Users must be able to log in using their email address and password"
# MAGIC   ) AS semantic_similarity
