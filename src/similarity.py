import numpy as np
from typing import List, Tuple, Dict, Union
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import nltk

class RequirementSimilarity:
    def __init__(self, embedding_models: Union[List[str], Dict[str, str]] = {'default': 'all-MiniLM-L6-v2'}):
        # Initialize models
        if isinstance(embedding_models, list):
            # If list provided, use model names as keys
            self.sentence_transformers = {
                model: SentenceTransformer(model)
                for model in embedding_models
            }
        else:
            # If dict provided, use custom names as keys
            self.sentence_transformers = {
                custom_name: SentenceTransformer(model_name)
                for custom_name, model_name in embedding_models.items()
            }
            
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
        # Calculate semantic weight split between models
        semantic_weight = 0.6
        per_semantic_model_weight = semantic_weight / len(self.sentence_transformers)
        per_textual_model_weight = (1 - semantic_weight) / len(self.sentence_transformers)
        
        # Weights for final scoring
        self.weights = {
            'levenshtein': per_textual_model_weight,
            'jaccard': per_textual_model_weight,
            'rouge': per_textual_model_weight,
            'bleu': per_textual_model_weight,
        }
        
        # Add weights for each semantic model
        for model_name in self.sentence_transformers:
            self.weights[f'semantic_{model_name}'] = per_semantic_model_weight

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

    def semantic_sim(self, text1: str, text2: str, model_name: str) -> float:
        """Semantic similarity using sentence transformers"""
        embeddings = self.sentence_transformers[model_name].encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def get_combined_similarity(self, text1: str, text2: str) -> Dict:
        """Calculate weighted combination of all similarity metrics"""
        scores = {
            'levenshtein': self.levenshtein_sim(text1, text2),
            'jaccard': self.jaccard_sim(text1, text2),
            'rouge': self.rouge_sim(text1, text2),
            'bleu': self.bleu_sim(text1, text2),
        }
        
        # Add semantic scores for each model
        for model_name in self.sentence_transformers:
            scores[f'semantic_{model_name}'] = self.semantic_sim(text1, text2, model_name)
        
        # Calculate weighted sum
        scores['final_score'] = sum(
            scores[metric] * weight 
            for metric, weight in self.weights.items()
            )
        
        scores['query'] = text1
        scores['requirement'] = text2
        
        return scores