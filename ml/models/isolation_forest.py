"""
Training pipeline for Isolation Forest anomaly detection
"""
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.isolation_forest import IsolationForestDetector
from training.data_preprocessor import LogDataPreprocessor
from evaluation.metrics import AnomalyDetectionMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_from_mongodb(
    mongodb_uri: str,
    database: str = 'log_analytics',
    collection: str = 'logs',
    limit: int = 10000
) -> pd.DataFrame:
    """
    Load log data from MongoDB
    
    Args:
        mongodb_uri: MongoDB connection string
        database: Database name
        collection: Collection name
        limit: Maximum number of records
        
    Returns:
        DataFrame with log data
    """
    from pymongo import MongoClient
    
    logger.info(f"Loading data from MongoDB: {database}.{collection}")
    
    client = MongoClient(mongodb_uri)
    db = client[database]
    
    # Get logs
    logs = list(db[collection].find().limit(limit))
    
    # Convert to DataFrame
    df = pd.DataFrame(logs)
    
    logger.info(f"Loaded {len(df)} log entries")
    
    client.close()
    
    return df


def load_data_from_csv(csv_path: str) -> pd.DataFrame:
    """Load log data from CSV file"""
    logger.info(f"Loading data from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} log entries")
    return df


def train_isolation_forest(
    train_data: pd.DataFrame,
    preprocessor: LogDataPreprocessor,
    contamination: float = 0.1,
    n_estimators: int = 100,
    random_state: int = 42
) -> IsolationForestDetector:
    """
    Train Isolation Forest model
    
    Args:
        train_data: Training data
        preprocessor: Fitted preprocessor
        contamination: Expected proportion of anomalies
        n_estimators: Number of trees
        random_state: Random seed
        
    Returns:
        Trained model
    """
    logger.info("Training Isolation Forest model...")
    
    # Preprocess data
    X_train = preprocessor.transform(train_data)
    
    # Create and train model
    model = IsolationForestDetector(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state
    )
    
    model.fit(X_train)
    
    logger.info("Training completed")
    
    return model


def evaluate_model(
    model: IsolationForestDetector,
    test_data: pd.DataFrame,
    preprocessor: LogDataPreprocessor,
    true_labels: np.ndarray = None
) -> dict:
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        test_data: Test data
        preprocessor: Fitted preprocessor
        true_labels: True anomaly labels (if available)
        
    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Preprocess
    X_test = preprocessor.transform(test_data)
    
    # Predict
    predictions = model.predict(X_test)
    scores = model.score_samples(X_test)
    
    # Calculate metrics
    metrics = {}
    
    if true_labels is not None:
        metrics_calculator = AnomalyDetectionMetrics()
        metrics = metrics_calculator.calculate_all_metrics(
            y_true=true_labels,
            y_pred=predictions,
            y_scores=scores
        )
    else:
        # Basic statistics without labels
        metrics = {
            'n_samples': len(predictions),
            'n_anomalies_detected': np.sum(predictions),
            'anomaly_rate': np.mean(predictions),
            'score_mean': np.mean(scores),
            'score_std': np.std(scores)
        }
    
    return metrics


def save_model_artifacts(
    model: IsolationForestDetector,
    preprocessor: LogDataPreprocessor,
    metrics: dict,
    output_dir: str
):
    """
    Save model, preprocessor, and metadata
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path / 'isolation_forest_model.pkl'
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save preprocessor
    preprocessor_path = output_path / 'preprocessor.pkl'
    preprocessor.save(str(preprocessor_path))
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'isolation_forest',
        'training_date': datetime.now().isoformat(),
        'n_features': len(preprocessor.feature_names),
        'feature_names': preprocessor.feature_names,
        'metrics': metrics,
        'model_params': {
            'contamination': model.contamination,
            'n_estimators': model.n_estimators,
            'max_samples': model.max_samples,
            'random_state': model.random_state
        }
    }
    
    metadata_path = output_path / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_path}")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train Isolation Forest anomaly detector')
    
    parser.add_argument(
        '--data-source',
        type=str,
        choices=['mongodb', 'csv'],
        default='mongodb',
        help='Data source'
    )
    parser.add_argument(
        '--mongodb-uri',
        type=str,
        default='mongodb://localhost:27017',
        help='MongoDB connection string'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        help='Path to CSV file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10000,
        help='Maximum number of records to load'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.1,
        help='Expected proportion of anomalies'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees in forest'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='saved_models',
        help='Output directory for model artifacts'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        if args.data_source == 'mongodb':
            df = load_data_from_mongodb(
                mongodb_uri=args.mongodb_uri,
                limit=args.limit
            )
        else:
            if not args.csv_path:
                raise ValueError("--csv-path required when data-source is 'csv'")
            df = load_data_from_csv(args.csv_path)
        
        # Check if we have minimum data
        if len(df) < 100:
            logger.warning(f"Only {len(df)} samples available. This may not be enough for training.")
        
        # Split data
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Create and fit preprocessor
        preprocessor = LogDataPreprocessor(
            max_tfidf_features=50,
            scale_numerical=True,
            scaler_type='standard'
        )
        
        X_train = preprocessor.fit_transform(train_df)
        logger.info(f"Feature matrix shape: {X_train.shape}")
        
        # Train model
        model = train_isolation_forest(
            train_data=train_df,
            preprocessor=preprocessor,
            contamination=args.contamination,
            n_estimators=args.n_estimators,
            random_state=args.random_state
        )
        
        # Evaluate
        metrics = evaluate_model(
            model=model,
            test_data=test_df,
            preprocessor=preprocessor
        )
        
        logger.info("Evaluation metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Save artifacts
        save_model_artifacts(
            model=model,
            preprocessor=preprocessor,
            metrics=metrics,
            output_dir=args.output_dir
        )
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()