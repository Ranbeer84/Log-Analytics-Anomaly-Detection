"""
Visualization utilities for anomaly detection results
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict, Any
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class AnomalyVisualization:
    """Visualization tools for anomaly detection"""
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Confusion Matrix',
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'],
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = 'ROC Curve',
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            title: Plot title
            save_path: Path to save figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = 'Precision-Recall Curve',
        save_path: Optional[str] = None
    ):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            title: Plot title
            save_path: Path to save figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax.plot([0, 1], [baseline, baseline], 'k--', linewidth=1, label=f'Baseline = {baseline:.3f}')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curve to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_score_distribution(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = 'Anomaly Score Distribution',
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of anomaly scores for normal vs anomaly
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            title: Plot title
            save_path: Path to save figure
        """
        normal_scores = y_scores[y_true == 0]
        anomaly_scores = y_scores[y_true == 1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue')
        ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red')
        
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved score distribution to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_training_history(
        train_losses: List[float],
        val_losses: List[float],
        title: str = 'Training History',
        save_path: Optional[str] = None
    ):
        """
        Plot training and validation losses
        
        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
        ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_anomaly_timeline(
        timestamps: pd.Series,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        title: str = 'Anomaly Timeline',
        save_path: Optional[str] = None
    ):
        """
        Plot anomalies over time
        
        Args:
            timestamps: Timestamps
            y_pred: Predicted labels
            y_scores: Anomaly scores
            title: Plot title
            save_path: Path to save figure
        """
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'is_anomaly': y_pred,
            'score': y_scores
        })
        
        df = df.sort_values('timestamp')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Plot 1: Anomaly scores over time
        ax1.plot(df['timestamp'], df['score'], 'b-', alpha=0.5, linewidth=1)
        
        # Highlight anomalies
        anomalies = df[df['is_anomaly'] == 1]
        ax1.scatter(
            anomalies['timestamp'],
            anomalies['score'],
            color='red',
            s=50,
            alpha=0.7,
            label='Anomalies'
        )
        
        ax1.set_ylabel('Anomaly Score')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly count per time window
        df.set_index('timestamp', inplace=True)
        anomaly_counts = df['is_anomaly'].resample('1H').sum()
        
        ax2.bar(
            anomaly_counts.index,
            anomaly_counts.values,
            width=0.04,
            color='red',
            alpha=0.6
        )
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Anomaly Count')
        ax2.set_title('Anomalies per Hour')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved anomaly timeline to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_feature_importance(
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
        title: str = 'Feature Importance',
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            importances: Feature importance scores
            top_n: Number of top features to show
            title: Plot title
            save_path: Path to save figure
        """
        # Sort by importance
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_threshold_analysis(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        title: str = 'Threshold Analysis',
        save_path: Optional[str] = None
    ):
        """
        Plot metrics vs threshold to help choose optimal threshold
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            thresholds: Threshold values to test
            title: Plot title
            save_path: Path to save figure
        """
        if thresholds is None:
            thresholds = np.linspace(
                np.min(y_scores),
                np.max(y_scores),
                100
            )
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_scores > threshold).astype(int)
            
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
        ax.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
        ax.plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1 Score')
        
        # Mark best F1
        best_idx = np.argmax(f1_scores)
        ax.axvline(
            thresholds[best_idx],
            color='k',
            linestyle='--',
            linewidth=1,
            label=f'Best F1 at {thresholds[best_idx]:.3f}'
        )
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved threshold analysis to {save_path}")
        
        plt.show()
    
    @staticmethod
    def create_evaluation_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        output_dir: str,
        timestamps: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        feature_importances: Optional[np.ndarray] = None
    ):
        """
        Create comprehensive evaluation report with all plots
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Anomaly scores
            output_dir: Directory to save plots
            timestamps: Timestamps (optional)
            feature_names: Feature names (optional)
            feature_importances: Feature importances (optional)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating evaluation report in {output_dir}")
        
        # Confusion matrix
        AnomalyVisualization.plot_confusion_matrix(
            y_true, y_pred,
            save_path=str(output_path / 'confusion_matrix.png')
        )
        plt.close()
        
        # ROC curve
        AnomalyVisualization.plot_roc_curve(
            y_true, y_scores,
            save_path=str(output_path / 'roc_curve.png')
        )
        plt.close()
        
        # PR curve
        AnomalyVisualization.plot_precision_recall_curve(
            y_true, y_scores,
            save_path=str(output_path / 'pr_curve.png')
        )
        plt.close()
        
        # Score distribution
        AnomalyVisualization.plot_score_distribution(
            y_true, y_scores,
            save_path=str(output_path / 'score_distribution.png')
        )
        plt.close()
        
        # Threshold analysis
        AnomalyVisualization.plot_threshold_analysis(
            y_true, y_scores,
            save_path=str(output_path / 'threshold_analysis.png')
        )
        plt.close()
        
        # Timeline (if timestamps provided)
        if timestamps is not None:
            AnomalyVisualization.plot_anomaly_timeline(
                timestamps, y_pred, y_scores,
                save_path=str(output_path / 'anomaly_timeline.png')
            )
            plt.close()
        
        # Feature importance (if provided)
        if feature_names and feature_importances is not None:
            AnomalyVisualization.plot_feature_importance(
                feature_names, feature_importances,
                save_path=str(output_path / 'feature_importance.png')
            )
            plt.close()
        
        logger.info(f"Evaluation report completed in {output_dir}")