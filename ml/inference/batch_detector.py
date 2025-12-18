"""
Batch anomaly detection for processing large volumes of logs
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import torch
import pickle
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.isolation_forest import IsolationForestDetector
from models.autoencoder import Autoencoder, AutoencoderTrainer
from models.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderTrainer
from training.data_preprocessor import LogDataPreprocessor, create_sequences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchAnomalyDetector:
    """
    Unified interface for batch anomaly detection
    
    Supports multiple models:
    - Isolation Forest
    - Autoencoder
    - LSTM Autoencoder
    """
    
    def __init__(self, model_dir: str, model_type: str = 'isolation_forest'):
        """
        Initialize batch detector
        
        Args:
            model_dir: Directory containing model artifacts
            model_type: Type of model ('isolation_forest', 'autoencoder', 'lstm')
        """
        self.model_dir = Path(model_dir)
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.metadata = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, preprocessor, and metadata"""
        logger.info(f"Loading {self.model_type} artifacts from {self.model_dir}")
        
        if self.model_type == 'isolation_forest':
            self._load_isolation_forest()
        elif self.model_type == 'autoencoder':
            self._load_autoencoder()
        elif self.model_type == 'lstm':
            self._load_lstm()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_isolation_forest(self):
        """Load Isolation Forest model"""
        model_path = self.model_dir / 'isolation_forest_model.pkl'
        preprocessor_path = self.model_dir / 'preprocessor.pkl'
        
        self.model = IsolationForestDetector()
        self.model.load(str(model_path))
        
        self.preprocessor = LogDataPreprocessor()
        self.preprocessor.load(str(preprocessor_path))
        
        logger.info("Isolation Forest model loaded")
    
    def _load_autoencoder(self):
        """Load Autoencoder model"""
        model_path = self.model_dir / 'autoencoder_model.pt'
        preprocessor_path = self.model_dir / 'autoencoder_preprocessor.pkl'
        
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['model_config']
        
        model = Autoencoder(
            input_dim=config['input_dim'],
            encoding_dims=config['encoding_dims'],
            dropout_rate=config['dropout_rate']
        )
        
        self.model = AutoencoderTrainer(model, device='cpu')
        self.model.load_model(str(model_path))
        
        self.preprocessor = LogDataPreprocessor()
        self.preprocessor.load(str(preprocessor_path))
        
        logger.info("Autoencoder model loaded")
    
    def _load_lstm(self):
        """Load LSTM Autoencoder model"""
        model_path = self.model_dir / 'lstm_autoencoder_model.pt'
        preprocessor_path = self.model_dir / 'lstm_autoencoder_preprocessor.pkl'
        
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['model_config']
        
        model = LSTMAutoencoder(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional']
        )
        
        self.model = LSTMAutoencoderTrainer(model, device='cpu')
        self.model.load_model(str(model_path))
        
        self.preprocessor = LogDataPreprocessor()
        self.preprocessor.load(str(preprocessor_path))
        
        logger.info("LSTM Autoencoder model loaded")
    
    def detect(
        self,
        data: pd.DataFrame,
        threshold: Optional[float] = None,
        batch_size: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Detect anomalies in batch
        
        Args:
            data: Input DataFrame
            threshold: Anomaly threshold
            batch_size: Processing batch size
            
        Returns:
            (anomaly_scores, is_anomaly, anomaly_details)
        """
        logger.info(f"Processing {len(data)} log entries...")
        
        # Preprocess data
        X = self.preprocessor.transform(data)
        
        # Process in batches
        all_scores = []
        all_predictions = []
        
        n_batches = (len(X) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            X_batch = X[start_idx:end_idx]
            
            if self.model_type == 'isolation_forest':
                batch_predictions = self.model.predict(X_batch)
                batch_scores = self.model.score_samples(X_batch)
            else:
                # Neural network models
                X_tensor = torch.FloatTensor(X_batch)
                batch_scores, batch_predictions = self.model.predict_anomalies(
                    X_tensor,
                    threshold=threshold
                )
            
            all_scores.extend(batch_scores)
            all_predictions.extend(batch_predictions)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {end_idx}/{len(X)} samples")
        
        scores = np.array(all_scores)
        predictions = np.array(all_predictions)
        
        # Create anomaly details
        anomaly_details = self._create_anomaly_details(
            data, predictions, scores
        )
        
        logger.info(
            f"Detection complete. Found {np.sum(predictions)} anomalies "
            f"({np.mean(predictions)*100:.2f}%)"
        )
        
        return scores, predictions, anomaly_details
    
    def _create_anomaly_details(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        scores: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Create detailed anomaly information"""
        anomaly_indices = np.where(predictions > 0)[0]
        
        details = []
        
        for idx in anomaly_indices:
            detail = {
                'index': int(idx),
                'anomaly_score': float(scores[idx]),
                'timestamp': data.iloc[idx].get('timestamp'),
                'message': data.iloc[idx].get('message', ''),
                'level': data.iloc[idx].get('level', 'UNKNOWN'),
                'service': data.iloc[idx].get('service', 'unknown'),
                'response_time': data.iloc[idx].get('response_time'),
            }
            details.append(detail)
        
        # Sort by score (most anomalous first)
        details.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        return details
    
    def detect_and_save(
        self,
        data: pd.DataFrame,
        output_path: str,
        threshold: Optional[float] = None,
        save_all: bool = False
    ):
        """
        Detect anomalies and save results
        
        Args:
            data: Input DataFrame
            output_path: Output file path
            threshold: Anomaly threshold
            save_all: If True, save all logs with scores; if False, only anomalies
        """
        scores, predictions, details = self.detect(data, threshold)
        
        # Add predictions to DataFrame
        result_df = data.copy()
        result_df['anomaly_score'] = scores
        result_df['is_anomaly'] = predictions
        
        # Filter if needed
        if not save_all:
            result_df = result_df[result_df['is_anomaly'] == 1]
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            result_df.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            result_df.to_json(output_path, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")
        
        logger.info(f"Results saved to {output_path}")
        
        # Save summary
        summary = {
            'total_logs': len(data),
            'anomalies_detected': int(np.sum(predictions)),
            'anomaly_rate': float(np.mean(predictions)),
            'detection_date': datetime.now().isoformat(),
            'model_type': self.model_type,
            'score_statistics': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            }
        }
        
        summary_path = output_path.parent / f"{output_path.stem}_summary.json"
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_path}")
        
        return result_df, summary


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detectors
    
    Combines predictions from multiple models for better accuracy
    """
    
    def __init__(self, detectors: List[BatchAnomalyDetector]):
        """
        Initialize ensemble
        
        Args:
            detectors: List of BatchAnomalyDetector instances
        """
        self.detectors = detectors
        logger.info(f"Initialized ensemble with {len(detectors)} detectors")
    
    def detect(
        self,
        data: pd.DataFrame,
        voting: str = 'soft',
        weights: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using ensemble
        
        Args:
            data: Input DataFrame
            voting: 'soft' (average scores) or 'hard' (majority vote)
            weights: Optional weights for each detector
            
        Returns:
            (ensemble_scores, ensemble_predictions)
        """
        if weights is None:
            weights = [1.0] * len(self.detectors)
        
        if len(weights) != len(self.detectors):
            raise ValueError("Number of weights must match number of detectors")
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Get predictions from all detectors
        all_scores = []
        all_predictions = []
        
        for detector in self.detectors:
            scores, predictions, _ = detector.detect(data)
            all_scores.append(scores)
            all_predictions.append(predictions)
        
        all_scores = np.array(all_scores)
        all_predictions = np.array(all_predictions)
        
        if voting == 'soft':
            # Weighted average of scores
            ensemble_scores = np.average(all_scores, axis=0, weights=weights)
            
            # Threshold at median
            threshold = np.median(ensemble_scores)
            ensemble_predictions = (ensemble_scores > threshold).astype(int)
            
        elif voting == 'hard':
            # Majority vote
            ensemble_predictions = (
                np.sum(all_predictions * weights.reshape(-1, 1), axis=0) > 0.5
            ).astype(int)
            
            # Average scores for information
            ensemble_scores = np.mean(all_scores, axis=0)
        else:
            raise ValueError(f"Unknown voting method: {voting}")
        
        logger.info(
            f"Ensemble detection: {np.sum(ensemble_predictions)} anomalies "
            f"({np.mean(ensemble_predictions)*100:.2f}%)"
        )
        
        return ensemble_scores, ensemble_predictions


def load_detector(model_dir: str, model_type: str = 'isolation_forest') -> BatchAnomalyDetector:
    """
    Convenience function to load a detector
    
    Args:
        model_dir: Model directory
        model_type: Model type
        
    Returns:
        Loaded detector
    """
    return BatchAnomalyDetector(model_dir=model_dir, model_type=model_type)


def create_ensemble(model_dirs: List[str], model_types: List[str]) -> EnsembleAnomalyDetector:
    """
    Create ensemble detector
    
    Args:
        model_dirs: List of model directories
        model_types: List of model types
        
    Returns:
        Ensemble detector
    """
    if len(model_dirs) != len(model_types):
        raise ValueError("Number of directories must match number of types")
    
    detectors = [
        BatchAnomalyDetector(model_dir=d, model_type=t)
        for d, t in zip(model_dirs, model_types)
    ]
    
    return EnsembleAnomalyDetector(detectors)