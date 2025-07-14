from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from sentence_transformers import SentenceTransformer
import json

logger = logging.getLogger(__name__)

class EvaluationEngine:
    """Comprehensive evaluation engine for BioMuse benchmarking tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_config = self.config.get('evaluation', {})
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
    
    def evaluate_retrieval(self, predicted_papers: List[int], ground_truth: List[int], 
                          top_k: int = 3) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            predicted_papers: List of predicted paper indices
            ground_truth: List of ground truth paper indices
            top_k: Number of top predictions to consider
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not predicted_papers or not ground_truth:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'ndcg': 0.0}
        
        # Take top-k predictions
        top_k_predictions = predicted_papers[:top_k]
        
        # Calculate metrics
        overlap = len(set(top_k_predictions) & set(ground_truth))
        precision = overlap / len(top_k_predictions) if top_k_predictions else 0.0
        recall = overlap / len(ground_truth) if ground_truth else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate NDCG (simplified version)
        ndcg = self._calculate_ndcg(top_k_predictions, ground_truth)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ndcg': ndcg,
            'top_k': top_k,
            'num_predictions': len(top_k_predictions),
            'num_ground_truth': len(ground_truth),
            'overlap': overlap
        }
    
    def evaluate_tag_prediction(self, predicted_tags: List[str], gold_tags: List[str]) -> Dict[str, float]:
        """
        Evaluate tag prediction performance.
        
        Args:
            predicted_tags: List of predicted tags
            gold_tags: List of gold standard tags
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not predicted_tags and not gold_tags:
            return {'macro_f1': 1.0, 'micro_f1': 1.0, 'jaccard_similarity': 1.0}
        
        if not predicted_tags or not gold_tags:
            return {'macro_f1': 0.0, 'micro_f1': 0.0, 'jaccard_similarity': 0.0}
        
        # Convert to sets for set-based metrics
        pred_set = set(tag.lower().strip() for tag in predicted_tags)
        gold_set = set(tag.lower().strip() for tag in gold_tags)
        
        # Calculate Jaccard similarity
        intersection = len(pred_set & gold_set)
        union = len(pred_set | gold_set)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Calculate F1 scores
        all_tags = list(pred_set | gold_set)
        if not all_tags:
            return {'macro_f1': 1.0, 'micro_f1': 1.0, 'jaccard_similarity': 1.0}
        
        # Create binary vectors for each tag
        y_true = []
        y_pred = []
        
        for tag in all_tags:
            y_true.append(1 if tag in gold_set else 0)
            y_pred.append(1 if tag in pred_set else 0)
        
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        return {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'jaccard_similarity': jaccard,
            'num_predicted': len(predicted_tags),
            'num_gold': len(gold_tags),
            'num_correct': intersection
        }
    
    def evaluate_summarization(self, predicted_summary: str, gold_summary: str) -> Dict[str, float]:
        """
        Evaluate summarization performance.
        
        Args:
            predicted_summary: Generated summary
            gold_summary: Gold standard summary
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not predicted_summary or not gold_summary:
            return {'rouge_l': 0.0, 'bertscore': 0.0, 'hallucination_rate': 0.0}
        
        # Calculate ROUGE-L (simplified)
        rouge_l = self._calculate_rouge_l(predicted_summary, gold_summary)
        
        # Calculate BERTScore
        bertscore = self._calculate_bertscore(predicted_summary, gold_summary)
        
        # Calculate hallucination rate
        hallucination_rate = self._detect_hallucination(predicted_summary, gold_summary)
        
        return {
            'rouge_l': rouge_l,
            'bertscore': bertscore,
            'hallucination_rate': hallucination_rate,
            'predicted_length': len(predicted_summary),
            'gold_length': len(gold_summary)
        }
    
    def evaluate_classification(self, predicted_domain: str, true_domain: str, 
                              possible_domains: List[str]) -> Dict[str, Any]:
        """
        Evaluate domain classification performance.
        
        Args:
            predicted_domain: Predicted domain
            true_domain: True domain
            possible_domains: List of possible domains
            
        Returns:
            Dictionary of evaluation metrics
        """
        accuracy = 1.0 if predicted_domain == true_domain else 0.0
        
        # For ARI, we'd need multiple predictions, but here we just return accuracy
        return {
            'accuracy': accuracy,
            'predicted_domain': predicted_domain,
            'true_domain': true_domain,
            'correct': predicted_domain == true_domain
        }
    
    def _calculate_ndcg(self, predictions: List[int], ground_truth: List[int], k: int = 3) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not predictions or not ground_truth:
            return 0.0
        
        # Simplified NDCG calculation
        dcg = 0.0
        idcg = 0.0
        
        for i, pred in enumerate(predictions[:k]):
            if pred in ground_truth:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Ideal DCG
        for i in range(min(k, len(ground_truth))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_rouge_l(self, predicted: str, gold: str) -> float:
        """Calculate ROUGE-L score (simplified version)."""
        if not predicted or not gold:
            return 0.0
        
        # Simple longest common subsequence
        pred_words = predicted.lower().split()
        gold_words = gold.lower().split()
        
        if not pred_words or not gold_words:
            return 0.0
        
        # Calculate LCS length
        lcs_length = self._longest_common_subsequence(pred_words, gold_words)
        
        precision = lcs_length / len(pred_words) if pred_words else 0.0
        recall = lcs_length / len(gold_words) if gold_words else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def _calculate_bertscore(self, predicted: str, gold: str) -> float:
        """Calculate BERTScore (simplified version using sentence transformers)."""
        if not self.sentence_model or not predicted or not gold:
            return 0.0
        
        try:
            # Encode both texts
            pred_embedding = self.sentence_model.encode([predicted])[0]
            gold_embedding = self.sentence_model.encode([gold])[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([pred_embedding], [gold_embedding])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating BERTScore: {e}")
            return 0.0
    
    def _detect_hallucination(self, predicted: str, gold: str) -> float:
        """
        Detect hallucination in generated text.
        
        Args:
            predicted: Generated text
            gold: Ground truth text
            
        Returns:
            Hallucination rate (0.0 = no hallucination, 1.0 = complete hallucination)
        """
        if not predicted or not gold:
            return 1.0 if predicted else 0.0
        
        # Simple hallucination detection based on semantic similarity
        if self.sentence_model:
            try:
                pred_embedding = self.sentence_model.encode([predicted])[0]
                gold_embedding = self.sentence_model.encode([gold])[0]
                similarity = cosine_similarity([pred_embedding], [gold_embedding])[0][0]
                
                # Convert similarity to hallucination rate (inverse relationship)
                hallucination_rate = 1.0 - similarity
                return max(0.0, min(1.0, hallucination_rate))
            except Exception as e:
                logger.warning(f"Error in hallucination detection: {e}")
        
        # Fallback: simple word overlap
        pred_words = set(predicted.lower().split())
        gold_words = set(gold.lower().split())
        
        if not pred_words:
            return 1.0
        
        overlap = len(pred_words & gold_words)
        hallucination_rate = 1.0 - (overlap / len(pred_words))
        return max(0.0, min(1.0, hallucination_rate))
    
    def evaluate_task_batch(self, tasks: List[Dict[str, Any]], 
                           predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch of tasks.
        
        Args:
            tasks: List of task dictionaries
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary with aggregated evaluation results
        """
        results = {
            'retrieval': [],
            'tag_prediction': [],
            'summarization': [],
            'classification': []
        }
        
        for task, prediction in zip(tasks, predictions):
            task_type = task.get('task_type')
            
            if task_type == 'retrieval':
                eval_result = self.evaluate_retrieval(
                    prediction.get('predicted_papers', []),
                    task.get('ground_truth', [])
                )
                results['retrieval'].append(eval_result)
            
            elif task_type == 'tag_prediction':
                eval_result = self.evaluate_tag_prediction(
                    prediction.get('predicted_tags', []),
                    task.get('ground_truth', [])
                )
                results['tag_prediction'].append(eval_result)
            
            elif task_type == 'summarization':
                eval_result = self.evaluate_summarization(
                    prediction.get('predicted_summary', ''),
                    task.get('ground_truth', '')
                )
                results['summarization'].append(eval_result)
            
            elif task_type == 'classification':
                eval_result = self.evaluate_classification(
                    prediction.get('predicted_domain', ''),
                    task.get('ground_truth', ''),
                    task.get('possible_domains', [])
                )
                results['classification'].append(eval_result)
        
        # Aggregate results
        aggregated = {}
        for task_type, task_results in results.items():
            if task_results:
                aggregated[task_type] = self._aggregate_metrics(task_results)
        
        return aggregated
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple tasks."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    aggregated[f'{key}_mean'] = np.mean(values)
                    aggregated[f'{key}_std'] = np.std(values)
                    aggregated[f'{key}_min'] = np.min(values)
                    aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated

# Legacy functions for backward compatibility
def evaluate_retrieval(predicted, ground_truth):
    engine = EvaluationEngine()
    return engine.evaluate_retrieval(predicted, ground_truth)

def evaluate_tag_prediction(predicted_tags, gold_tags):
    engine = EvaluationEngine()
    return engine.evaluate_tag_prediction(predicted_tags, gold_tags)