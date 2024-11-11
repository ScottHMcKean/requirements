import numpy as np
from typing import List, Tuple
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import nltk

class RequirementSimilarity:
    def __init__(self):
        # Initialize models
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
        # Weights for final scoring
        self.weights = {
            'levenshtein': 0.1,
            'jaccard': 0.1,
            'rouge': 0.1,
            'bleu': 0.1,
            'semantic': 0.6
        }

    def tokenize(self, text: str) -> List[str]:
        """Basic tokenizeing"""
        tokens = nltk.word_tokenize(text.lower())
        return tokens

    def levenshtein_sim(self, text1: str, text2: str) -> float:
        """Normalized Levenshtein similarity"""
        dist = levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        return 1 - (dist / max_len)

    def jaccard_sim(self, text1: str, text2: str) -> float:
        """Jaccard similarity"""
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union)

    def rouge_sim(self, text1: str, text2: str) -> float:
        """ROUGE similarity (average of ROUGE-1, ROUGE-2, ROUGE-L)"""
        scores = self.rouge_scorer.score(text1, text2)
        rouge_scores = [
            scores['rouge1'].fmeasure,
            scores['rouge2'].fmeasure,
            scores['rougeL'].fmeasure
        ]
        return np.mean(rouge_scores)

    def bleu_sim(self, text1: str, text2: str) -> float:
        """BLEU similarity"""
        reference = [self.tokenize(text1)]
        candidate = self.tokenize(text2)
        return sentence_bleu(reference, candidate)

    def semantic_sim(self, text1: str, text2: str) -> float:
        """Semantic similarity using sentence transformers"""
        embeddings = self.sentence_transformer.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def get_combined_similarity(self, text1: str, text2: str) -> Tuple[float, dict]:
        """Calculate weighted combination of all similarity metrics"""
        scores = {
            'levenshtein': self.levenshtein_sim(text1, text2),
            'jaccard': self.jaccard_sim(text1, text2),
            'rouge': self.rouge_sim(text1, text2),
            'bleu': self.bleu_sim(text1, text2),
            'semantic': self.semantic_sim(text1, text2)
        }
        
        # Calculate weighted sum
        final_score = sum(
            scores[metric] * weight 
            for metric, weight in self.weights.items()
            )
        
        return final_score, scores