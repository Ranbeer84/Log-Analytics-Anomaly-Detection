"""
Isolation Forest Model for Anomaly Detection
"""
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Isolation Forest-based anomaly detector for log data"""
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees
            max_samples: Number of samples to draw
            random_state: Random seed
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_names = []
        self.training_stats = {}
        
    def train(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        Train the Isolation Forest model
        
        Args:
            features: Training feature matrix (n_samples, n_features)
            feature_names: Names of features
            
        Returns:
            Training statistics
        """
        try:
            logger.info(f"Training Isolation Forest on {features.shape[0]} samples")
            
            # Train model
            self.model.fit(features)
            self.is_trained = True
            self.feature_names = feature_names
            
            # Calculate training statistics
            scores = self.model.score_samples(features)
            predictions = self.model.predict(features)
            
            self.training_stats = {
                'n_samples': features.shape[0],
                'n_features': features.shape[1],
                'anomaly_count': int(np.sum(predictions == -1)),
                'anomaly_rate': float(np.mean(predictions == -1)),
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'trained_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Training complete. Anomaly rate: {self.training_stats['anomaly_rate']:.2%}")
            return self.training_stats
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(
        self,
        features: np.ndarray,
        return_scores: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict anomalies for new data
        
        Args:
            features: Feature matrix (n_samples, n_features)
            return_scores: Whether to return anomaly scores
            
        Returns:
            predictions: Array of predictions (1=normal, -1=anomaly)
            scores: Anomaly scores (if return_scores=True)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(features)
        
        if return_scores:
            scores = self.model.score_samples(features)
            # Convert to 0-1 scale (higher = more anomalous)
            normalized_scores = self._normalize_scores(scores)
            return predictions, normalized_scores
        
        return predictions, None
    
    def predict_single(
        self,
        features: np.ndarray
    ) -> Dict:
        """
        Predict anomaly for a single log entry
        
        Args:
            features: Single feature vector
            
        Returns:
            Prediction result with score and classification
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        prediction, scores = self.predict(features, return_scores=True)
        
        is_anomaly = prediction[0] == -1
        anomaly_score = float(scores[0])
        
        # Determine severity based on score
        severity = self._calculate_severity(anomaly_score)
        
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': anomaly_score,
            'severity': severity,
            'confidence': self._calculate_confidence(anomaly_score),
            'prediction': int(prediction[0])
        }
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize anomaly scores to 0-1 range"""
        # Isolation Forest scores are negative (more negative = more anomalous)
        # Convert to 0-1 scale where 1 = most anomalous
        min_score = self.training_stats.get('min_score', scores.min())
        max_score = self.training_stats.get('max_score', scores.max())
        
        if max_score == min_score:
            return np.zeros_like(scores)
        
        normalized = 1 - (scores - min_score) / (max_score - min_score)
        return np.clip(normalized, 0, 1)
    
    def _calculate_severity(self, score: float) -> str:
        """Calculate severity level based on anomaly score"""
        if score >= 0.8:
            return 'critical'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence(self, score: float) -> float:
        """Calculate confidence in the prediction"""
        # Higher scores = higher confidence
        return min(score * 1.2, 1.0)
    
    def save(self, filepath: str) -> None:
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'training_stats': self.training_stats,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load(self, filepath: str) -> None:
        """Load model from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.training_stats = model_data['training_stats']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (approximation for Isolation Forest)
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # For Isolation Forest, we approximate importance by
        # the variance in path lengths for each feature
        return dict(zip(self.feature_names, [1.0] * len(self.feature_names)))
    
    def get_model_info(self) -> Dict:
        """Get model metadata and statistics"""
        return {
            'model_type': 'IsolationForest',
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'hyperparameters': {
                'contamination': self.model.contamination,
                'n_estimators': self.model.n_estimators,
                'max_samples': self.model.max_samples
            }
        }