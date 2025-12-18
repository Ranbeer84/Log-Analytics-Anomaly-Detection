"""
LSTM Autoencoder for sequence-based anomaly detection
Detects anomalies in temporal patterns of log data
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for sequence anomaly detection
    
    Useful for detecting:
    - Unusual temporal patterns
    - Sequence anomalies
    - Time-series outliers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM autoencoder
        
        Args:
            input_dim: Feature dimension per timestep
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim * self.num_directions,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(
            f"Initialized LSTM Autoencoder: "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, bidirectional={bidirectional}"
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Encode input sequence
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            (encoded_sequence, (hidden_state, cell_state))
        """
        encoded, (hidden, cell) = self.encoder_lstm(x)
        
        # For bidirectional, combine forward and backward hidden states
        if self.bidirectional:
            # hidden shape: (num_layers * 2, batch, hidden_dim)
            # Reshape to: (num_layers, batch, hidden_dim * 2)
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
            
            cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        
        return encoded, (hidden, cell)
    
    def decode(
        self,
        encoded: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """
        Decode from latent representation
        
        Args:
            encoded: Encoded tensor from encoder
            seq_len: Target sequence length
            
        Returns:
            Decoded sequence
        """
        batch_size = encoded.size(0)
        
        # Use last encoded state as initial input
        decoder_input = encoded[:, -1:, :]  # (batch, 1, hidden_dim * directions)
        
        # Initialize hidden state from encoder output
        hidden = None
        cell = None
        
        outputs = []
        
        for _ in range(seq_len):
            # Decode one step
            if hidden is None:
                output, (hidden, cell) = self.decoder_lstm(decoder_input)
            else:
                output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            
            # Project to input dimension
            output = self.output_layer(output)
            outputs.append(output)
            
            # Use output as next input (teacher forcing alternative)
            decoder_input = self.dropout_layer(hidden[-1:].transpose(0, 1))
        
        # Concatenate all outputs
        decoded = torch.cat(outputs, dim=1)
        
        return decoded
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Reconstructed sequence
        """
        seq_len = x.size(1)
        
        # Encode
        encoded, _ = self.encode(x)
        
        # Decode
        decoded = self.decode(encoded, seq_len)
        
        return decoded
    
    def reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Calculate reconstruction error
        
        Args:
            x: Input tensor
            reduction: 'none', 'mean', or 'sum'
            
        Returns:
            Reconstruction error
        """
        reconstructed = self.forward(x)
        mse = nn.functional.mse_loss(reconstructed, x, reduction='none')
        
        if reduction == 'none':
            # Return error per sample (averaged across time and features)
            return mse.mean(dim=(1, 2))
        elif reduction == 'mean':
            return mse.mean()
        elif reduction == 'sum':
            return mse.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


class LSTMAutoencoderTrainer:
    """Trainer for LSTM autoencoder"""
    
    def __init__(
        self,
        model: LSTMAutoencoder,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        """
        Initialize trainer
        
        Args:
            model: LSTM Autoencoder model
            learning_rate: Learning rate
            device: Training device
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
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move to device
            if isinstance(batch, (tuple, list)):
                data = batch[0].to(self.device)
            else:
                data = batch.to(self.device)
            
            # Ensure 3D input: (batch, seq_len, features)
            if data.dim() == 2:
                data = data.unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.model(data)
            loss = self.criterion(reconstructed, data)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    data = batch[0].to(self.device)
                else:
                    data = batch.to(self.device)
                
                if data.dim() == 2:
                    data = data.unsqueeze(1)
                
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
        """Train the model"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                
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
            data: Input sequences
            threshold: Anomaly threshold
            percentile: Percentile for threshold
            
        Returns:
            (anomaly_scores, is_anomaly)
        """
        self.model.eval()
        
        with torch.no_grad():
            data = data.to(self.device)
            if data.dim() == 2:
                data = data.unsqueeze(1)
            
            errors = self.model.reconstruction_error(data).cpu().numpy()
        
        if threshold is None:
            threshold = np.percentile(errors, percentile)
        
        is_anomaly = errors > threshold
        
        return errors, is_anomaly
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
                'bidirectional': self.model.bidirectional
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {path}")


def create_lstm_autoencoder(
    input_dim: int,
    architecture: str = 'medium'
) -> LSTMAutoencoder:
    """
    Factory function for LSTM autoencoder
    
    Args:
        input_dim: Input feature dimension
        architecture: 'small', 'medium', or 'large'
        
    Returns:
        LSTM Autoencoder model
    """
    architectures = {
        'small': {'hidden_dim': 32, 'num_layers': 1},
        'medium': {'hidden_dim': 64, 'num_layers': 2},
        'large': {'hidden_dim': 128, 'num_layers': 3}
    }
    
    config = architectures.get(architecture, architectures['medium'])
    
    return LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=0.2,
        bidirectional=False
    )