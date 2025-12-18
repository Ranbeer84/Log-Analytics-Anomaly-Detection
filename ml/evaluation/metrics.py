"""
Evaluation metrics for anomaly detection models
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetectionMetrics:
    """
    Comprehensive metrics for anomaly detection evaluation
    """
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Precision: TP / (TP + FP)
        
        Of all predicted anomalies, how many are actually anomalies?
        """
        return precision_score(y_true, y_pred, zero_division=0)
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Recall (Sensitivity): TP / (TP + FN)
        
        Of all actual anomalies, how many did we detect?
        """
        return recall_score(y_true, y_pred, zero_division=0)
    
    @staticmethod
    def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        F1 Score: 2 * (precision * recall) / (precision + recall)
        
        Harmonic mean of precision and recall
        """
        return f1_score(y_true, y_pred, zero_division=0)
    
    @staticmethod
    def f_beta(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        beta: float = 2.0
    ) -> float:
        """
        F-Beta Score: Weighted harmonic mean of precision and recall
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            beta: Weight of recall vs precision (>1 favors recall)
            
        Returns:
            F-beta score
        """
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        if precision + recall == 0:
            return 0.0
        
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    
    @staticmethod
    def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Specificity: TN / (TN + FP)
        
        Of all actual normal samples, how many did we correctly identify?
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        if (tn + fp) == 0:
            return 0.0
        return tn / (tn + fp)
    
    @staticmethod
    def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        False Positive Rate: FP / (FP + TN)
        
        Of all normal samples, how many did we incorrectly flag as anomalies?
        """
        return 1.0 - AnomalyDetectionMetrics.specificity(y_true, y_pred)
    
    @staticmethod
    def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        False Negative Rate: FN / (FN + TP)
        
        Of all anomalies, how many did we miss?
        """
        return 1.0 - AnomalyDetectionMetrics.recall(y_true, y_pred)
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Accuracy: (TP + TN) / (TP + TN + FP + FN)
        
        Overall correctness (can be misleading for imbalanced datasets)
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Balanced Accuracy: (Sensitivity + Specificity) / 2
        
        Better for imbalanced datasets
        """
        sensitivity = AnomalyDetectionMetrics.recall(y_true, y_pred)
        specificity = AnomalyDetectionMetrics.specificity(y_true, y_pred)
        return (sensitivity + specificity) / 2
    
    @staticmethod
    def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        ROC AUC Score: Area under ROC curve
        
        Measures ability to distinguish between classes
        """
        try:
            return roc_auc_score(y_true, y_scores)
        except ValueError:
            return 0.0
    
    @staticmethod
    def pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Precision-Recall AUC: Area under PR curve
        
        Better for imbalanced datasets than ROC AUC
        """
        try:
            return average_precision_score(y_true, y_scores)
        except ValueError:
            return 0.0
    
    @staticmethod
    def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Matthews Correlation Coefficient
        
        Balanced measure even for imbalanced datasets
        Range: [-1, 1], where 1 is perfect, 0 is random, -1 is inverse
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def confusion_matrix_dict(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, int]:
        """
        Get confusion matrix as dictionary
        
        Returns:
            Dict with tn, fp, fn, tp
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    
    @staticmethod
    def detection_rate_at_k(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        k: int
    ) -> float:
        """
        Detection rate at top-k predictions
        
        What percentage of anomalies are in the top-k scored samples?
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            k: Number of top predictions
            
        Returns:
            Detection rate
        """
        # Get indices of top-k scores
        top_k_indices = np.argsort(y_scores)[-k:]
        
        # Count how many true anomalies are in top-k
        n_anomalies_detected = np.sum(y_true[top_k_indices])
        total_anomalies = np.sum(y_true)
        
        if total_anomalies == 0:
            return 0.0
        
        return n_anomalies_detected / total_anomalies
    
    @staticmethod
    def precision_at_k(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        k: int
    ) -> float:
        """
        Precision at top-k predictions
        
        Of the top-k predictions, what percentage are actual anomalies?
        """
        top_k_indices = np.argsort(y_scores)[-k:]
        return np.mean(y_true[top_k_indices])
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate all available metrics
        
        Args:
            y_true: True labels (0=normal, 1=anomaly)
            y_pred: Predicted labels
            y_scores: Anomaly scores (optional, for AUC metrics)
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['precision'] = self.precision(y_true, y_pred)
        metrics['recall'] = self.recall(y_true, y_pred)
        metrics['f1_score'] = self.f1(y_true, y_pred)
        metrics['f2_score'] = self.f_beta(y_true, y_pred, beta=2.0)
        metrics['accuracy'] = self.accuracy(y_true, y_pred)
        metrics['balanced_accuracy'] = self.balanced_accuracy(y_true, y_pred)
        
        # Specificity and error rates
        metrics['specificity'] = self.specificity(y_true, y_pred)
        metrics['false_positive_rate'] = self.false_positive_rate(y_true, y_pred)
        metrics['false_negative_rate'] = self.false_negative_rate(y_true, y_pred)
        
        # Matthews correlation coefficient
        metrics['matthews_corrcoef'] = self.matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix
        metrics['confusion_matrix'] = self.confusion_matrix_dict(y_true, y_pred)
        
        # Score-based metrics (if scores provided)
        if y_scores is not None:
            metrics['roc_auc'] = self.roc_auc(y_true, y_scores)
            metrics['pr_auc'] = self.pr_auc(y_true, y_scores)
            
            # Detection at various k values
            n_samples = len(y_true)
            for k_percent in [1, 5, 10, 20]:
                k = max(1, int(n_samples * k_percent / 100))
                metrics[f'detection_rate_at_{k_percent}pct'] = self.detection_rate_at_k(
                    y_true, y_scores, k
                )
                metrics[f'precision_at_{k_percent}pct'] = self.precision_at_k(
                    y_true, y_scores, k
                )
        
        # Additional statistics
        metrics['total_samples'] = len(y_true)
        metrics['total_anomalies'] = int(np.sum(y_true))
        metrics['predicted_anomalies'] = int(np.sum(y_pred))
        metrics['true_anomaly_rate'] = float(np.mean(y_true))
        metrics['predicted_anomaly_rate'] = float(np.mean(y_pred))
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics in readable format"""
        print("\n" + "="*60)
        print("ANOMALY DETECTION METRICS")
        print("="*60)
        
        # Basic metrics
        print("\nClassification Metrics:")
        print(f"  Precision:          {metrics['precision']:.4f}")
        print(f"  Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"  F1 Score:           {metrics['f1_score']:.4f}")
        print(f"  F2 Score:           {metrics['f2_score']:.4f}")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        
        # Additional metrics
        print("\nAdditional Metrics:")
        print(f"  Specificity:        {metrics['specificity']:.4f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
        print(f"  Matthews Corr. Coef: {metrics['matthews_corrcoef']:.4f}")
        
        # Score-based metrics
        if 'roc_auc' in metrics:
            print("\nScore-based Metrics:")
            print(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
            print(f"  PR AUC:             {metrics['pr_auc']:.4f}")
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {cm['true_negatives']:,}")
        print(f"  False Positives: {cm['false_positives']:,}")
        print(f"  False Negatives: {cm['false_negatives']:,}")
        print(f"  True Positives:  {cm['true_positives']:,}")
        
        # Statistics
        print("\nDataset Statistics:")
        print(f"  Total Samples:         {metrics['total_samples']:,}")
        print(f"  True Anomalies:        {metrics['total_anomalies']:,}")
        print(f"  Predicted Anomalies:   {metrics['predicted_anomalies']:,}")
        print(f"  True Anomaly Rate:     {metrics['true_anomaly_rate']:.2%}")
        print(f"  Predicted Anomaly Rate: {metrics['predicted_anomaly_rate']:.2%}")
        
        print("\n" + "="*60 + "\n")


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    print_results: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Anomaly scores (optional)
        print_results: Whether to print metrics
        
    Returns:
        Dictionary of metrics
    """
    calculator = AnomalyDetectionMetrics()
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_scores)
    
    if print_results:
        calculator.print_metrics(metrics)
    
    return metrics


def compare_models(
    models_results: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    y_true: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models
    
    Args:
        models_results: Dict of {model_name: (y_pred, y_scores)}
        y_true: True labels
        
    Returns:
        DataFrame with comparison
    """
    import pandas as pd
    
    calculator = AnomalyDetectionMetrics()
    comparison = []
    
    for model_name, (y_pred, y_scores) in models_results.items():
        metrics = calculator.calculate_all_metrics(y_true, y_pred, y_scores)
        
        comparison.append({
            'Model': model_name,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1_score'],
            'Accuracy': metrics['accuracy'],
            'ROC_AUC': metrics.get('roc_auc', np.nan),
            'PR_AUC': metrics.get('pr_auc', np.nan),
            'MCC': metrics['matthews_corrcoef']
        })
    
    df = pd.DataFrame(comparison)
    return df.sort_values('F1', ascending=False)