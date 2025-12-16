"""
Training Script for Isolation Forest Model
"""
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pymongo import MongoClient
from ml.inference.anomaly_detector import AnomalyDetector
from ml.feature_engineering.extractors import LogFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train anomaly detection models"""
    
    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "log_analytics"
    ):
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.logs_collection = self.db['logs']
        self.models_collection = self.db['ml_models']
        
    def load_training_data(
        self,
        limit: int = 10000,
        filter_query: dict = None
    ) -> list:
        """
        Load training data from MongoDB
        
        Args:
            limit: Maximum number of logs to load
            filter_query: MongoDB filter query
            
        Returns:
            List of log entries
        """
        logger.info(f"Loading training data (limit: {limit})")
        
        query = filter_query or {}
        logs = list(self.logs_collection.find(query).limit(limit))
        
        # Convert ObjectId to string
        for log in logs:
            log['_id'] = str(log['_id'])
        
        logger.info(f"Loaded {len(logs)} log entries")
        return logs
    
    def train_isolation_forest(
        self,
        training_logs: list,
        contamination: float = 0.1,
        n_estimators: int = 100
    ) -> tuple:
        """
        Train Isolation Forest model
        
        Args:
            training_logs: List of log entries
            contamination: Expected anomaly rate
            n_estimators: Number of trees
            
        Returns:
            Tuple of (detector, training_stats)
        """
        logger.info("Training Isolation Forest model")
        
        # Initialize detector
        detector = AnomalyDetector(contamination=contamination)
        detector.model.n_estimators = n_estimators
        
        # Train model
        stats = detector.train_model(training_logs)
        
        logger.info(f"Training completed: {stats}")
        return detector, stats
    
    def save_model(
        self,
        detector: AnomalyDetector,
        training_stats: dict,
        version: str = None
    ) -> str:
        """
        Save trained model and metadata
        
        Args:
            detector: Trained AnomalyDetector
            training_stats: Training statistics
            version: Model version
            
        Returns:
            Path to saved model
        """
        # Create models directory
        models_dir = Path(__file__).parent.parent / 'saved_models'
        models_dir.mkdir(exist_ok=True)
        
        # Generate version if not provided
        if not version:
            version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Save model file
        model_path = models_dir / f'isolation_forest_{version}.pkl'
        detector.model.save(str(model_path))
        
        # Save metadata
        metadata = {
            'model_type': 'isolation_forest',
            'version': version,
            'trained_at': datetime.utcnow().isoformat(),
            'model_path': str(model_path),
            'training_stats': training_stats,
            'feature_names': detector.feature_extractor.get_feature_names(),
            'hyperparameters': {
                'contamination': detector.model.model.contamination,
                'n_estimators': detector.model.model.n_estimators
            }
        }
        
        metadata_path = models_dir / f'model_metadata_{version}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save to MongoDB
        self.models_collection.insert_one({
            **metadata,
            'status': 'active',
            'created_at': datetime.utcnow()
        })
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Metadata saved: {metadata_path}")
        
        return str(model_path)
    
    def evaluate_model(
        self,
        detector: AnomalyDetector,
        test_logs: list
    ) -> dict:
        """
        Evaluate model on test data
        
        Args:
            detector: Trained AnomalyDetector
            test_logs: List of test log entries
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating model on {len(test_logs)} test logs")
        
        results = detector.detect_batch(test_logs)
        
        # Calculate metrics
        anomaly_count = sum(1 for r in results if r.get('is_anomaly', False))
        anomaly_rate = anomaly_count / len(results) if results else 0
        
        scores = [r.get('anomaly_score', 0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        metrics = {
            'total_samples': len(test_logs),
            'anomalies_detected': anomaly_count,
            'anomaly_rate': anomaly_rate,
            'average_score': avg_score,
            'score_distribution': {
                'min': min(scores) if scores else 0,
                'max': max(scores) if scores else 0,
                'mean': avg_score
            }
        }
        
        logger.info(f"Evaluation complete: {metrics}")
        return metrics
    
    def close(self):
        """Close database connections"""
        self.mongo_client.close()


def main():
    """Main training pipeline"""
    logger.info("Starting model training pipeline")
    
    # Initialize trainer
    trainer = ModelTrainer(
        mongo_uri=os.getenv('MONGODB_URI', 'mongodb://localhost:27017'),
        db_name=os.getenv('DB_NAME', 'log_analytics')
    )
    
    try:
        # 1. Load training data
        training_logs = trainer.load_training_data(limit=5000)
        
        if len(training_logs) < 100:
            logger.error("Insufficient training data. Need at least 100 logs.")
            return
        
        # 2. Split data (80/20)
        split_idx = int(len(training_logs) * 0.8)
        train_data = training_logs[:split_idx]
        test_data = training_logs[split_idx:]
        
        logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")
        
        # 3. Train model
        detector, training_stats = trainer.train_isolation_forest(
            train_data,
            contamination=0.1,
            n_estimators=100
        )
        
        # 4. Evaluate model
        evaluation_metrics = trainer.evaluate_model(detector, test_data)
        
        # 5. Save model
        model_path = trainer.save_model(
            detector,
            {**training_stats, 'evaluation': evaluation_metrics}
        )
        
        logger.info("=" * 50)
        logger.info("Training Pipeline Complete!")
        logger.info(f"Model saved at: {model_path}")
        logger.info(f"Training samples: {training_stats['n_samples']}")
        logger.info(f"Anomaly rate: {training_stats['anomaly_rate']:.2%}")
        logger.info(f"Test anomalies: {evaluation_metrics['anomalies_detected']}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    main()