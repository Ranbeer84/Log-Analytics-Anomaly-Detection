"""
Training pipeline for Autoencoder and LSTM Autoencoder
"""
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder, AutoencoderTrainer
from models.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderTrainer
from training.data_preprocessor import LogDataPreprocessor, create_sequences
from evaluation.metrics import AnomalyDetectionMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(source: str, **kwargs) -> pd.DataFrame:
    """Load data from source"""
    if source == 'mongodb':
        from pymongo import MongoClient
        
        uri = kwargs.get('mongodb_uri', 'mongodb://localhost:27017')
        database = kwargs.get('database', 'log_analytics')
        collection = kwargs.get('collection', 'logs')
        limit = kwargs.get('limit', 10000)
        
        client = MongoClient(uri)
        db = client[database]
        logs = list(db[collection].find().limit(limit))
        df = pd.DataFrame(logs)
        client.close()
        
        logger.info(f"Loaded {len(df)} records from MongoDB")
        
    elif source == 'csv':
        csv_path = kwargs.get('csv_path')
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from CSV")
        
    else:
        raise ValueError(f"Unknown data source: {source}")
    
    return df


def prepare_data_for_autoencoder(
    df: pd.DataFrame,
    preprocessor: LogDataPreprocessor,
    test_size: float = 0.2,
    batch_size: int = 32,
    random_state: int = 42
) -> tuple:
    """
    Prepare data for standard autoencoder
    
    Returns:
        (train_loader, val_loader, X_test, test_df)
    """
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Fit preprocessor on training data
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    
    # Further split training into train/val
    X_train_split, X_val_split = train_test_split(
        X_train, test_size=0.2, random_state=random_state
    )
    
    # Create PyTorch tensors
    train_tensor = torch.FloatTensor(X_train_split)
    val_tensor = torch.FloatTensor(X_val_split)
    
    # Create data loaders
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    logger.info(f"Train: {len(train_tensor)}, Val: {len(val_tensor)}, Test: {len(X_test)}")
    
    return train_loader, val_loader, X_test, test_df


def prepare_data_for_lstm(
    df: pd.DataFrame,
    preprocessor: LogDataPreprocessor,
    sequence_length: int = 10,
    test_size: float = 0.2,
    batch_size: int = 32,
    random_state: int = 42
) -> tuple:
    """
    Prepare sequential data for LSTM autoencoder
    
    Returns:
        (train_loader, val_loader, X_test_seq, test_df)
    """
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Preprocess
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    
    # Create sequences
    X_train_seq = create_sequences(X_train, sequence_length=sequence_length)
    X_test_seq = create_sequences(X_test, sequence_length=sequence_length)
    
    # Split train into train/val
    split_idx = int(len(X_train_seq) * 0.8)
    X_train_split = X_train_seq[:split_idx]
    X_val_split = X_train_seq[split_idx:]
    
    # Create tensors
    train_tensor = torch.FloatTensor(X_train_split)
    val_tensor = torch.FloatTensor(X_val_split)
    
    # Create loaders
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(
        f"Sequences - Train: {len(train_tensor)}, "
        f"Val: {len(val_tensor)}, Test: {len(X_test_seq)}"
    )
    
    return train_loader, val_loader, X_test_seq, test_df


def train_standard_autoencoder(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    architecture: str = 'medium',
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> AutoencoderTrainer:
    """Train standard autoencoder"""
    logger.info(f"Training standard autoencoder (architecture: {architecture})...")
    
    # Create model
    model = Autoencoder(
        input_dim=input_dim,
        encoding_dims={'small': [64, 32], 'medium': [128, 64, 32], 'large': [256, 128, 64, 32]}[architecture],
        dropout_rate=0.2
    )
    
    # Create trainer
    trainer = AutoencoderTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device
    )
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=10,
        verbose=True
    )
    
    return trainer


def train_lstm_autoencoder(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    architecture: str = 'medium',
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> LSTMAutoencoderTrainer:
    """Train LSTM autoencoder"""
    logger.info(f"Training LSTM autoencoder (architecture: {architecture})...")
    
    # Create model
    architectures = {
        'small': {'hidden_dim': 32, 'num_layers': 1},
        'medium': {'hidden_dim': 64, 'num_layers': 2},
        'large': {'hidden_dim': 128, 'num_layers': 3}
    }
    config = architectures[architecture]
    
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=0.2,
        bidirectional=False
    )
    
    # Create trainer
    trainer = LSTMAutoencoderTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device
    )
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=10,
        verbose=True
    )
    
    return trainer


def evaluate_and_save(
    trainer,
    X_test: np.ndarray,
    test_df: pd.DataFrame,
    output_dir: str,
    model_name: str,
    preprocessor: LogDataPreprocessor
):
    """Evaluate model and save artifacts"""
    logger.info("Evaluating model...")
    
    # Convert to tensor
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Predict anomalies
    scores, predictions = trainer.predict_anomalies(
        X_test_tensor,
        percentile=95
    )
    
    # Calculate metrics
    metrics = {
        'n_samples': len(predictions),
        'n_anomalies': int(np.sum(predictions)),
        'anomaly_rate': float(np.mean(predictions)),
        'score_mean': float(np.mean(scores)),
        'score_std': float(np.std(scores)),
        'score_min': float(np.min(scores)),
        'score_max': float(np.max(scores))
    }
    
    logger.info("Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Save artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path / f'{model_name}_model.pt'
    trainer.save_model(str(model_path))
    
    # Save preprocessor
    preprocessor_path = output_path / f'{model_name}_preprocessor.pkl'
    preprocessor.save(str(preprocessor_path))
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'training_date': datetime.now().isoformat(),
        'input_dim': trainer.model.input_dim,
        'metrics': metrics,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses
    }
    
    metadata_path = output_path / f'{model_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {output_path}")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train autoencoder models')
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['autoencoder', 'lstm'],
        required=True,
        help='Model type to train'
    )
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
        help='MongoDB URI'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        help='CSV file path'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10000,
        help='Data limit'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Model architecture'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=10,
        help='Sequence length for LSTM'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cpu or cuda)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='saved_models',
        help='Output directory'
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
        df = load_data(
            source=args.data_source,
            mongodb_uri=args.mongodb_uri,
            csv_path=args.csv_path,
            limit=args.limit
        )
        
        # Create preprocessor
        preprocessor = LogDataPreprocessor(
            max_tfidf_features=50,
            scale_numerical=True,
            scaler_type='minmax'  # MinMax for neural networks
        )
        
        # Train based on model type
        if args.model_type == 'autoencoder':
            train_loader, val_loader, X_test, test_df = prepare_data_for_autoencoder(
                df=df,
                preprocessor=preprocessor,
                batch_size=args.batch_size,
                random_state=args.random_state
            )
            
            input_dim = X_test.shape[1]
            
            trainer = train_standard_autoencoder(
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                architecture=args.architecture,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=args.device
            )
            
            evaluate_and_save(
                trainer=trainer,
                X_test=X_test,
                test_df=test_df,
                output_dir=args.output_dir,
                model_name='autoencoder',
                preprocessor=preprocessor
            )
            
        elif args.model_type == 'lstm':
            train_loader, val_loader, X_test_seq, test_df = prepare_data_for_lstm(
                df=df,
                preprocessor=preprocessor,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                random_state=args.random_state
            )
            
            input_dim = X_test_seq.shape[2]
            
            trainer = train_lstm_autoencoder(
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                architecture=args.architecture,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=args.device
            )
            
            # Reshape for evaluation
            X_test_reshaped = X_test_seq.reshape(-1, input_dim)
            
            evaluate_and_save(
                trainer=trainer,
                X_test=X_test_reshaped,
                test_df=test_df,
                output_dir=args.output_dir,
                model_name='lstm_autoencoder',
                preprocessor=preprocessor
            )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()