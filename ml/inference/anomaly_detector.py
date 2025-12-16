"""
Real-time Anomaly Detection Service
"""
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

from ml.models.isolation_forest import IsolationForestDetector
from ml.feature_engineering.extractors import LogFeatureExtractor, SequenceFeatureExtractor

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Real-time anomaly detection for log entries"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        contamination: float = 0.1,
        window_size: int = 10
    ):
        """
        Initialize anomaly detector
        
        Args:
            model_path: Path to pre-trained model (if available)
            contamination: Expected anomaly rate for new model
            window_size: Size of sliding window for sequence features
        """
        self.model = IsolationForestDetector(contamination=contamination)
        self.feature_extractor = LogFeatureExtractor()
        self.sequence_extractor = SequenceFeatureExtractor(window_size=window_size)
        
        self.log_window = []
        self.window_size = window_size
        self.detection_stats = {
            'total_processed': 0,
            'anomalies_detected': 0,
            'last_detection': None
        }
        
        # Load pre-trained model if available
        if model_path and Path(model_path).exists():
            try:
                self.model.load(model_path)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {str(e)}")
    
    def detect_anomaly(
        self,
        log_entry: Dict[str, Any],
        use_sequence: bool = False
    ) -> Dict:
        """
        Detect if a log entry is anomalous
        
        Args:
            log_entry: Log entry dictionary
            use_sequence: Whether to use sequence features
            
        Returns:
            Detection result with anomaly score and metadata
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features(log_entry)
            
            # Add sequence features if enabled
            if use_sequence and len(self.log_window) > 0:
                seq_features = self.sequence_extractor.extract_window_features(
                    self.log_window
                )
                features = np.concatenate([features, seq_features])
            
            # Detect anomaly
            if self.model.is_trained:
                result = self.model.predict_single(features)
            else:
                # Default response if model not trained
                result = {
                    'is_anomaly': False,
                    'anomaly_score': 0.0,
                    'severity': 'low',
                    'confidence': 0.0,
                    'prediction': 1
                }
                result['model_trained'] = False
            
            # Update statistics
            self.detection_stats['total_processed'] += 1
            if result['is_anomaly']:
                self.detection_stats['anomalies_detected'] += 1
                self.detection_stats['last_detection'] = datetime.utcnow().isoformat()
            
            # Update sliding window
            self._update_window(log_entry)
            
            # Add metadata to result
            result.update({
                'log_id': log_entry.get('_id'),
                'timestamp': log_entry.get('timestamp'),
                'service': log_entry.get('service'),
                'detection_method': 'isolation_forest',
                'feature_vector_size': len(features)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return {
                'is_anomaly': False,
                'error': str(e),
                'anomaly_score': 0.0
            }
    
    def detect_batch(
        self,
        log_entries: List[Dict]
    ) -> List[Dict]:
        """
        Detect anomalies for multiple log entries
        
        Args:
            log_entries: List of log entry dictionaries
            
        Returns:
            List of detection results
        """
        results = []
        
        for log_entry in log_entries:
            result = self.detect_anomaly(log_entry)
            results.append(result)
        
        return results
    
    def train_model(
        self,
        training_logs: List[Dict],
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train the anomaly detection model
        
        Args:
            training_logs: List of log entries for training
            save_path: Path to save trained model
            
        Returns:
            Training statistics
        """
        try:
            logger.info(f"Training model on {len(training_logs)} log entries")
            
            # Extract features from training data
            features = self.feature_extractor.extract_batch_features(training_logs)
            feature_names = self.feature_extractor.get_feature_names()
            
            # Train model
            stats = self.model.train(features, feature_names)
            
            # Save model if path provided
            if save_path:
                self.model.save(save_path)
                logger.info(f"Model saved to {save_path}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def update_model(
        self,
        new_logs: List[Dict],
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Retrain model with new data (simple retraining strategy)
        
        Args:
            new_logs: New log entries for retraining
            save_path: Path to save updated model
            
        Returns:
            Updated training statistics
        """
        return self.train_model(new_logs, save_path)
    
    def _update_window(self, log_entry: Dict) -> None:
        """Update the sliding window of recent logs"""
        self.log_window.append(log_entry)
        
        if len(self.log_window) > self.window_size:
            self.log_window.pop(0)
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        anomaly_rate = 0.0
        if self.detection_stats['total_processed'] > 0:
            anomaly_rate = (
                self.detection_stats['anomalies_detected'] / 
                self.detection_stats['total_processed']
            )
        
        return {
            **self.detection_stats,
            'anomaly_rate': anomaly_rate,
            'window_size': len(self.log_window),
            'model_info': self.model.get_model_info() if self.model.is_trained else None
        }
    
    def reset_statistics(self) -> None:
        """Reset detection statistics"""
        self.detection_stats = {
            'total_processed': 0,
            'anomalies_detected': 0,
            'last_detection': None
        }
        logger.info("Detection statistics reset")


class EnsembleAnomalyDetector:
    """Ensemble of multiple anomaly detectors for improved accuracy"""
    
    def __init__(self, detectors: List[AnomalyDetector]):
        """
        Initialize ensemble detector
        
        Args:
            detectors: List of AnomalyDetector instances
        """
        self.detectors = detectors
        
    def detect_anomaly(self, log_entry: Dict) -> Dict:
        """
        Detect anomaly using ensemble voting
        
        Args:
            log_entry: Log entry dictionary
            
        Returns:
            Ensemble detection result
        """
        results = []
        
        for detector in self.detectors:
            result = detector.detect_anomaly(log_entry)
            results.append(result)
        
        # Voting: majority decides
        anomaly_votes = sum(1 for r in results if r.get('is_anomaly', False))
        is_anomaly = anomaly_votes > len(self.detectors) / 2
        
        # Average scores
        avg_score = np.mean([r.get('anomaly_score', 0) for r in results])
        avg_confidence = np.mean([r.get('confidence', 0) for r in results])
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': float(avg_score),
            'confidence': float(avg_confidence),
            'ensemble_votes': anomaly_votes,
            'total_detectors': len(self.detectors),
            'individual_results': results,
            'detection_method': 'ensemble'
        }