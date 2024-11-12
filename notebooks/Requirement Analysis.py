# Databricks notebook source
# MAGIC %md
# MAGIC # Requirement Similarity Analysis
# MAGIC
# MAGIC This notebook provides an example of similarity analysis for requirements. It takes two datasets and uses both natural language processing and embeddings to measure the similarity between a new requirement and a table of existing requirements.
# MAGIC
# MAGIC We analyze two types of similarity - textual similarity and semantic similarity. Textual similarity is the similarity between the words (or tokens) in the requirements. Semantic similarity is the similarity between the meaning (i.e. semantics) of the requirements.
# MAGIC
# MAGIC This approach to similarity is general and applicable to many problems like operational comments, reviews, product descriptions, etc.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd

# We need this for textual tokenization 
import nltk
nltk.download('punkt')

# Set up the relative path
import sys
sys.path.append('../src/')
from similarity import RequirementSimilarity

# Set up default logger
import logging
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.client_server").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md
# MAGIC The first dataset we use is the software requirements from here:
# MAGIC https://huggingface.co/datasets/nguyenminh871/software_requirements
# MAGIC
# MAGIC This is a small dataset with three columns that have similar requirements - one for a python project, one for a smart contracts project, and one for a java project. We use the python and Java requirements from this dataset to do a detailed analysis of each distance metric and see how they quantify requirement similarity.

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

# MAGIC %md
# MAGIC Let's start by looking at these individual metrics, using the methods in the RequirementSimilarity class.

# COMMAND ----------

df = spark.table('shm.requirements.coal_dataset').toPandas()
req_similarity = RequirementSimilarity()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Textual Similarity
# MAGIC The goal of analyzing textual similarity is to find requirements that are exactly the same or very similar. For textual similarity, we use the Jaccard similarity, Levenshtein distance, ROUGE and BLEU. 
# MAGIC
# MAGIC ### 

# COMMAND ----------

text1 = "The system shall provide user authentication via email and password"
text2 = "Users must be able to log in using their email address and password"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Jaccard Similarity
# MAGIC Jaccard similarity measures similarity between finite sets by comparing their intersection to their union. It's good for a simple comparison of word overlap between texts, with a range from 0 (completely different) to 1 (identical).
# MAGIC
# MAGIC The example below uses Jaccard Similarity to show that 15% of the tokens intersect.

# COMMAND ----------

# raw code
from typing import List
def tokenize(text: str) -> List[str]:
    """Basic tokenizeing"""
    tokens = nltk.word_tokenize(text.lower())
    return tokens
tokens1 = set(tokenize(text1))
tokens2 = set(tokenize(text2))
intersection = tokens1.intersection(tokens2)
union = tokens1.union(tokens2)
print(len(intersection) / len(union))

# within the object
print(req_similarity.jaccard_sim(text1, text2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Levenshtein Distance
# MAGIC Levenshtein distance counts the minimum number of single-character edits (insertions, deletions, or substitutions) needed to transform one string into another. It's particularly good at catching typos and minor variations in requirements text. Since the raw score depends on string length, it needs to be normalized to provide a consistent similarity metric between 0 and 1.
# MAGIC
# MAGIC The example below uses Levenshtein distance to show that 63% of the longest string must be edited to match the shorter string.

# COMMAND ----------

# raw code
from Levenshtein import distance as levenshtein_distance
dist = levenshtein_distance(text1, text2)
max_len = max(len(text1), len(text2))
print(1 - (dist / max_len))

req_similarity.levenshtein_sim(text1, text2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROUGE
# MAGIC ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a text similarity metric originally designed for evaluating automatic summarization. It focuses on recall by measuring how much of the reference text is captured, making it ideal for ensuring completeness of content. With multiple variants (ROUGE-N, ROUGE-L, ROUGE-W) and the ability to provide both precision and recall metrics (we use the normalized F1-score), ROUGE effectively captures different aspects of similarity. As a well-established NLP metric with widespread implementation support, it's particularly valuable when you need to verify that all aspects of a requirement are thoroughly covered.

# COMMAND ----------

# raw code
from rouge_score import rouge_scorer
import numpy as np
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
scores = scorer.score(text1, text2)
rouge_scores = [
    scores['rouge1'].fmeasure,
    scores['rouge2'].fmeasure,
    scores['rougeL'].fmeasure
]
print("ROUGE SCORE")
print(np.mean(rouge_scores))

req_similarity.rouge_sim(text1, text2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### BLEU
# MAGIC BLEU focuses on precision by measuring how accurate the candidate text is compared to the reference. It uses a brevity penalty to prevent very short matches and combines different n-gram precisions (usually up to 4-grams). While originally designed for machine translation evaluation, BLEU is more suitable when accuracy is important, as it tends to be better at ensuring precise matching and avoiding false positives.
# MAGIC
# MAGIC The example shows how difficult it is to get a positive BLEU score on semantic similarity. For that reason we will probably give it low weight in the final ranking, but the lack of false positives is useful.

# COMMAND ----------

from nltk.translate.bleu_score import sentence_bleu
reference = [tokenize(text1)]
candidate = tokenize(text2)
print(sentence_bleu(reference, candidate))

req_similarity.bleu_sim(text1, text2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Semantic Similarity
# MAGIC
# MAGIC Semantic similarity measures how close requirements are in meaning, rather than exact wording. To capture this meaning computationally, we use embeddings - dense vector representations of text that encode semantic information. These embeddings are created using transformer-based encoding models like BERT, GPT, and Sentence-BERT, which are neural networks that condense words into numerical semantics rather than treating them as independent tokens.
# MAGIC
# MAGIC The power of embeddings comes from how they represent similar concepts close together in vector space, which we measure using cosine similarity. For example, terms like "authenticate" and "log in" will have similar embeddings, as will phrases like "shall" and "must", or "via email" and "using their email address". This allows us to find requirements that express the same ideas even when worded quite differently.
# MAGIC
# MAGIC When choosing an embedding model, we need to consider the tradeoffs between speed, accuracy, and domain specificity. Smaller models like MiniLM are faster and good for prototyping, while larger models like MPNet and GTE provide better accuracy at the cost of speed. The domain match is also important - CodeBERT may work well for software requirements, while models like SPECTER2 and BGE might be better suited for technical documentation. Each model has its own strengths and weaknesses, so testing different options is important. Additionally, each model uses a different tokenizer and vocabulary, which can affect the results.
# MAGIC

# COMMAND ----------

from sentence_transformers import SentenceTransformer

embedding_models=[
    'all-MiniLM-L6-v2',
    'microsoft/codebert-base',
    'BAAI/bge-large-en-v1.5',
    'dell-research-harvard/lt-un-data-fine-industry-en'
    ]

sentence_transformers = [
  SentenceTransformer(model) for model in embedding_models
  ]


# COMMAND ----------


