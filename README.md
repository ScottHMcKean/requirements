# Requirement Similarity

This repository provides an example of similarity analysis for requirements. It takes two datasets and uses both natural language processing and embeddings to measure the similarity between a new requirement and a table of existing requirements. 

We analyze two types of similarity - textual similarity and semantic similarity. Textual similarity is the similarity between the words in the requirements.Semantic similarity is the similarity between the meaning of the requirements.

One important consideration - we need to normalize all these metrics so they return scores between 0 and 1. 
The beauty of normalization is that we can create a weighted ensemble score by simply taking a weighted sum of the metrics.

## Setup

- Activate a virtual environment with `python -m venv .req_similarity`
- Install the dependencies with `pip install -r requirements.txt`
- Activate the virtual environment with `source .req_similarity/bin/activate`
- Download the NLTK Punkt tokenizer with `python -m nltk.downloader punkt`

## Datasets

The first dataset we use is the software requirements from here:
https://huggingface.co/datasets/nguyenminh871/software_requirements

This is quite a small dataset, but is nice because it has three columns with similar requirements - one for a python project, one for a smart contracts project, and one for a java project. This lets us see how well each distance metric can differentiate requirements, as well as use different embedding models and see how they compare.

The second dataset is the PURE requirements dataset:
https://www.kaggle.com/datasets/computerscience3/public-requirementspure-dataset

This is a convenience load of the XML files from the PURE dataset, paper and full dataset here:
https://ieeexplore.ieee.org/document/8049173
https://zenodo.org/records/1414117

This dataset is much larger, and contains requirements for a variety of different projects and standards, chunked into numerous requirements per file (see the id column). We can use this dataset to show how we'd use a vector database to search through the requirements at scale.

## Textual Similarity Methods

The goal of analyzing textual similarity is to find requirements that are exactly the same or very similar. For textual similarity, we use the Jaccard similarity, Levenshtein distance, ROUGE and BLEU. BLEU and ROUGE provide complementary insights (e.g. high BLEU + low ROUGE indicates accurate but incomplete matches), with ROUGE better for finding complete coverage and BLEU better for precise matches. BLEU and Jaccard similarity are based on token overlap, so we need to tokenize the requirements before we can calculate these metrics.

### Jaccard Similarity
Jaccard similarity measures similarity between finite sets by comparing their intersection to their union. It's good for a simple comparison of word overlap between texts, with a range from 0 (completely different) to 1 (identical).

### Levenshtein Distance
Levenshtein distance counts the minimum number of single-character edits (insertions, deletions, or substitutions) needed to transform one string into another. It's particularly good at catching typos and minor variations in requirements text. Since the raw score depends on string length, it needs to be normalized to provide a consistent similarity metric between 0 and 1.

### ROUGE
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a text similarity metric originally designed for evaluating automatic summarization. It focuses on recall by measuring how much of the reference text is captured, making it ideal for ensuring completeness of content. With multiple variants (ROUGE-N, ROUGE-L, ROUGE-W) and the ability to provide both precision and recall metrics (we use the normalized F1-score), ROUGE effectively captures different aspects of similarity. As a well-established NLP metric with widespread implementation support, it's particularly valuable when you need to verify that all aspects of a requirement are thoroughly covered.

### BLEU
BLEU focuses on precision by measuring how accurate the candidate text is compared to the reference. It uses a brevity penalty to prevent very short matches and combines different n-gram precisions (usually up to 4-grams). While originally designed for machine translation evaluation, BLEU is more suitable when accuracy is important, as it tends to be better at ensuring precise matching and avoiding false positives.

## Semantic Similarity Methods

Semantic similarity measures how close requirements are in meaning, rather than exact wording. To capture this meaning computationally, we use embeddings - dense vector representations of text that encode semantic information. These embeddings are created using transformer-based encoding models like BERT, GPT, and Sentence-BERT, which are neural networks that condense words into numerical semantics rather than treating them as independent tokens.

The power of embeddings comes from how they represent similar concepts close together in vector space, which we measure using cosine similarity. For example, terms like "authenticate" and "log in" will have similar embeddings, as will phrases like "shall" and "must", or "via email" and "using their email address". This allows us to find requirements that express the same ideas even when worded quite differently.

When choosing an embedding model, we need to consider the tradeoffs between speed, accuracy, and domain specificity. Smaller models like MiniLM are faster and good for prototyping, while larger models like BGE and GTE provide better accuracy at the cost of speed. The domain match is also important - CodeBERT may work well for software requirements, while models like SPECTER2 and BGE might be better suited for technical documentation. Each model has its own strengths and weaknesses, so testing different options is important. Additionally, each model uses a different tokenizer and vocabulary, which can affect the results.

## Ensemble Scoring

We normalize all the metrics so they return scores between 0 and 1, and then create a weighted ensemble score by simply taking a weighted sum of the metrics. The goal here is to store the weights as a hyperparameter so we can easily tune the solution based on SME feedback.
