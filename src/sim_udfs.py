import numpy as np
from typing import List, Tuple, Dict, Union
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk

def tokenize(text: str) -> List[str]:
    """Basic tokenizing"""
    tokens = nltk.word_tokenize(text.lower())
    return tokens

def levenshtein_sim(text1: str, text2: Union[str, pd.Series]) -> Union[float, pd.Series]:
    """Normalized Levenshtein similarity between a string and either another string or a pandas Series"""
    from Levenshtein import distance as levenshtein_distance
    
    if isinstance(text2, pd.Series):
        # Calculate distances for each row in text2
        distances = text2.apply(lambda x: levenshtein_distance(text1, x))
        max_lens = text2.str.len().combine(pd.Series([len(text1)]*len(text2)), max)
        return 1 - (distances / max_lens)


def jaccard_sim(text1: str, text2: str) -> float:
    """Jaccard similarity"""
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

def rouge_sim(text1: str, text2: str) -> float:
    """ROUGE similarity (average of ROUGE-1, ROUGE-2, ROUGE-L)"""
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = rouge_scorer_obj.score(text1, text2)
    rouge_scores = [
        scores['rouge1'].fmeasure,
        scores['rouge2'].fmeasure,
        scores['rougeL'].fmeasure
    ]
    return np.mean(rouge_scores)

def bleu_sim(text1: str, text2: str) -> float:
    """BLEU similarity"""
    reference = [tokenize(text1)]
    candidate = tokenize(text2)
    return sentence_bleu(reference, candidate)