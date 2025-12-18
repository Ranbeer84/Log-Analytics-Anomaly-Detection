"""
Autoencoder model for anomaly detection in log data
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """
    Autoencoder neural network for log anomaly detection
    
    Architecture:
    - Encoder: Compresses input to latent representation
    - Decoder: Reconstructs input from latent representation
    - Anomaly detected when reconstruction error is high
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dims: list = [128, 64, 32],
        dropout_rate: float = 0.2
    ):
        """
        Initialize autoencoder
        
        Args:
            input_dim: Input feature dimension
            encoding_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.dropout_rate = dropout_rate
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        decoding_dims = list(reversed(encoding_dims[:-1])) + [input_dim]
        prev_dim = encoding_dims[-1]
        
        for i, dim in enumerate(decoding_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            
            # Don't add activation/norm/dropout on last layer
            if i < len(decoding_dims) - 1:
                decoder_layers.extend([
                    nn.ReLU(),
                    nn.BatchNorm1d(dim),
                    nn.Dropout(dropout_rate)
                ])
            else:
                # Sigmoid on output layer for normalized inputs
                decoder_layers.append(nn.Sigmoid())
            
            prev_dim = dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        logger.info(f"Initialized Autoencoder: {input_dim} -> {encoding_dims[-1]} -> {input_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Reconstructed tensor (batch_size, input_dim)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded tensor (latent representation)
        """
        return self.encoder(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation
        
        Args:
            x: Encoded tensor
            
        Returns:
            Reconstructed tensor
        """
        return self.decoder(x)
    
    def reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Calculate reconstruction error (MSE)
        
        Args:
            x: Input tensor
            reduction: 'none', 'mean', or 'sum'
            
        Returns:
            Reconstruction error per sample or aggregated
        """
        reconstructed = self.forward(x)
        mse = nn.functional.mse_loss(reconstructed, x, reduction='none')
        
        if reduction == 'none':
            # Return error per sample (averaged across features)
            return mse.mean(dim=1)
        elif reduction == 'mean':
            return mse.mean()
        elif reduction == 'sum':
            return mse.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


class AutoencoderTrainer:
    """Trainer for autoencoder model"""
    
    def __init__(
        self,
        model: Autoencoder,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        """
        Initialize trainer
        
        Args:
            model: Autoencoder model
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move to device
            if isinstance(batch, (tuple, list)):
                data = batch[0].to(self.device)
            else:
                data = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.model(data)
            loss = self.criterion(reconstructed, data)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                if isinstance(batch, (tuple, list)):
                    data = batch[0].to(self.device)
                else:
                    data = batch.to(self.device)
                
                # Forward pass
                reconstructed = self.model(data)
                loss = self.criterion(reconstructed, data)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 50,
        early_stopping_patience: int = 5,
        verbose: bool = True
    ) -> dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Print progress
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}"
                    )
                
                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.6f}"
                    )
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def predict_anomalies(
        self,
        data: torch.Tensor,
        threshold: Optional[float] = None,
        percentile: float = 95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies based on reconstruction error
        
        Args:
            data: Input data
            threshold: Anomaly threshold (if None, computed from percentile)
            percentile: Percentile for threshold calculation
            
        Returns:
            (anomaly_scores, is_anomaly)
        """
        self.model.eval()
        
        with torch.no_grad():
            data = data.to(self.device)
            errors = self.model.reconstruction_error(data).cpu().numpy()
        
        # Compute threshold if not provided
        if threshold is None:
            threshold = np.percentile(errors, percentile)
        
        is_anomaly = errors > threshold
        
        return errors, is_anomaly
    
    def save_model(self, path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'input_dim': self.model.input_dim,
                'encoding_dims': self.model.encoding_dims,
                'dropout_rate': self.model.dropout_rate
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {path}")


def create_autoencoder(
    input_dim: int,
    architecture: str = 'small'
) -> Autoencoder:
    """
    Factory function to create autoencoder with predefined architectures
    
    Args:
        input_dim: Input feature dimension
        architecture: 'small', 'medium', or 'large'
        
    Returns:
        Autoencoder model
    """
    architectures = {
        'small': [64, 32],
        'medium': [128, 64, 32],
        'large': [256, 128, 64, 32]
    }
    
    encoding_dims = architectures.get(architecture, architectures['medium'])
    
    return Autoencoder(
        input_dim=input_dim,
        encoding_dims=encoding_dims,
        dropout_rate=0.2
    )