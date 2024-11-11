# Requirement Similarity

An example of using NLP and embeddings to analyze requirements. It takes two datasets and uses both natural language processing and embeddings to measure the similarity between a new requirement and a table of existing requirements. 

We analyze two types of similarity- textual similarity and semantic similarity.
Textual similarity is the similarity between the words in the requirements.
Semantic similarity is the similarity between the meaning of the requirements.

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

For exact matching, we use the Jaccard similarity, Levenshtein distance, ROUGE and BLEU. 
The goal of analyzing textual similarity is to find requirements that are exactly the same or very similar.

### Jaccard Similarity
- Measures similarity between finite sets by comparing their intersection to their union
- Good for a simple comparison of word overlap between texts
- Range: 0 (completely different) to 1 (identical)

### Levenshtein Distance
- Counts minimum number of single-character edits needed to change one string into another
- Good for catching typos and minor variations in requirements
- Raw score needs to be normalized for text length

### ROUGE
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation) was originally designed for evaluating automatic summarization, but it's quite useful for general text similarity.
- Focuses on recall (how much reference text is captured)
- Better for measuring if all important content is present
- Multiple variants (ROUGE-N, ROUGE-L, ROUGE-W)
- Originally for summarization evaluation
- More suitable when completeness is important
- ROUGE is particularly useful for similarity because it captures different aspects of similarity through its variants
- It provides both precision and recall, giving a more complete picture of similarity, but we use the normalized F1-score as a single similarity metric
- It's well-established in NLP research and has implementations in many languages.
- ROUGE might be better if you want to ensure all aspects of a requirement are covered

### BLEU
- Focuses on precision (how accurate the candidate text is)
- Uses a brevity penalty to prevent very short matches
- Combines different n-gram precisions (usually up to 4-grams)
- Originally for machine translation evaluation
- More suitable when accuracy is important
- BLEU might be better if you want to ensure precise matching and avoid false positives

Using both could provide complementary insights:
High BLEU + Low ROUGE = Accurate but incomplete match
Low BLEU + High ROUGE = Complete but imprecise match
High both = Very similar requirements
Low both = Very different requirements
The best choice often depends on your specific use case:
If you're looking for requirements that completely cover a reference requirement → ROUGE
If you're looking for requirements that precisely match → BLEU
3. If you want comprehensive similarity analysis → Use both

BLEU and Jaccard similarity are based on token overlap, so we need to tokenize the requirements before we can calculate these metrics.

## Semantic Similarity Methods

For semantic similarity, we use the cosine similarity between the embeddings of the requirements and test different embedding models. 
The goal of analyzing semantic similarity is to find requirements that are similar in meaning, but might have different wording.

Most embedding models (like BERT, GPT, etc.) come with their own specialized tokenizers that:
1. Know the model's vocabulary
2. Handle special tokens (like [CLS], [SEP], <s>, </s>)
3. Apply specific tokenization rules

## Ensemble Scoring

We normalize all the metrics so they return scores between 0 and 1, and then create a weighted ensemble score by simply taking a weighted sum of the metrics.
The goal here is to store the weights as a hyperparameter so we can easily tune the solution based on SME feedback.
