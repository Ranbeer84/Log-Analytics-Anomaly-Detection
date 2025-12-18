"""
Data preprocessing and feature engineering for log anomaly detection
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogDataPreprocessor:
    """
    Comprehensive preprocessor for log data
    
    Features extracted:
    - Temporal features (hour, day, minute)
    - Text features (TF-IDF of messages)
    - Categorical features (service, level)
    - Numerical features (response time, etc.)
    - Statistical features
    """
    
    def __init__(
        self,
        max_tfidf_features: int = 100,
        scale_numerical: bool = True,
        scaler_type: str = 'standard'
    ):
        """
        Initialize preprocessor
        
        Args:
            max_tfidf_features: Maximum TF-IDF features
            scale_numerical: Whether to scale numerical features
            scaler_type: 'standard' or 'minmax'
        """
        self.max_tfidf_features = max_tfidf_features
        self.scale_numerical = scale_numerical
        self.scaler_type = scaler_type
        
        # Transformers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        if scaler_type == 'standard':
            self.numerical_scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.numerical_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.categorical_encoders = {}
        
        # Feature names tracking
        self.feature_names = []
        self.is_fitted = False
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamp
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            DataFrame with temporal features
        """
        temporal_df = pd.DataFrame()
        
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            
            temporal_df['hour'] = timestamps.dt.hour
            temporal_df['day_of_week'] = timestamps.dt.dayofweek
            temporal_df['day_of_month'] = timestamps.dt.day
            temporal_df['minute'] = timestamps.dt.minute
            temporal_df['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(int)
            temporal_df['is_business_hours'] = (
                (timestamps.dt.hour >= 9) & (timestamps.dt.hour <= 17)
            ).astype(int)
            
            # Cyclical encoding for hour
            temporal_df['hour_sin'] = np.sin(2 * np.pi * temporal_df['hour'] / 24)
            temporal_df['hour_cos'] = np.cos(2 * np.pi * temporal_df['hour'] / 24)
            
            # Cyclical encoding for day of week
            temporal_df['dow_sin'] = np.sin(2 * np.pi * temporal_df['day_of_week'] / 7)
            temporal_df['dow_cos'] = np.cos(2 * np.pi * temporal_df['day_of_week'] / 7)
        
        return temporal_df
    
    def extract_text_features(
        self,
        messages: List[str],
        fit: bool = False
    ) -> np.ndarray:
        """
        Extract TF-IDF features from log messages
        
        Args:
            messages: List of log messages
            fit: Whether to fit the vectorizer
            
        Returns:
            TF-IDF feature matrix
        """
        if fit:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(messages)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(messages)
        
        return tfidf_matrix.toarray()
    
    def extract_categorical_features(
        self,
        df: pd.DataFrame,
        categorical_columns: List[str],
        fit: bool = False
    ) -> np.ndarray:
        """
        Encode categorical features
        
        Args:
            df: DataFrame with categorical columns
            categorical_columns: List of categorical column names
            fit: Whether to fit encoders
            
        Returns:
            Encoded categorical features
        """
        encoded_features = []
        
        for col in categorical_columns:
            if col not in df.columns:
                continue
            
            if fit:
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(df[col].fillna('unknown'))
                self.categorical_encoders[col] = encoder
            else:
                encoder = self.categorical_encoders.get(col)
                if encoder is None:
                    continue
                
                # Handle unknown categories
                encoded = []
                for val in df[col].fillna('unknown'):
                    try:
                        encoded.append(encoder.transform([val])[0])
                    except ValueError:
                        # Unknown category - assign to 0
                        encoded.append(0)
                encoded = np.array(encoded)
            
            encoded_features.append(encoded.reshape(-1, 1))
        
        if encoded_features:
            return np.hstack(encoded_features)
        else:
            return np.array([]).reshape(len(df), 0)
    
    def extract_numerical_features(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str]
    ) -> pd.DataFrame:
        """
        Extract and process numerical features
        
        Args:
            df: DataFrame with numerical columns
            numerical_columns: List of numerical column names
            
        Returns:
            DataFrame with processed numerical features
        """
        numerical_df = pd.DataFrame()
        
        for col in numerical_columns:
            if col in df.columns:
                # Fill NaN with median
                numerical_df[col] = df[col].fillna(df[col].median())
                
                # Add derived features for response_time
                if col == 'response_time':
                    numerical_df['response_time_log'] = np.log1p(numerical_df[col])
                    numerical_df['response_time_squared'] = numerical_df[col] ** 2
        
        return numerical_df
    
    def extract_statistical_features(
        self,
        df: pd.DataFrame,
        window_size: int = 10
    ) -> pd.DataFrame:
        """
        Extract statistical features using rolling windows
        
        Args:
            df: DataFrame with features
            window_size: Window size for rolling statistics
            
        Returns:
            DataFrame with statistical features
        """
        stat_df = pd.DataFrame()
        
        # Group by service for service-level statistics
        if 'service' in df.columns and 'response_time' in df.columns:
            # Service-level mean response time
            service_means = df.groupby('service')['response_time'].transform('mean')
            stat_df['service_response_mean'] = service_means
            
            # Deviation from service mean
            stat_df['response_deviation'] = df['response_time'] - service_means
        
        # Rolling statistics for response time
        if 'response_time' in df.columns and len(df) >= window_size:
            stat_df['response_time_rolling_mean'] = (
                df['response_time'].rolling(window=window_size, min_periods=1).mean()
            )
            stat_df['response_time_rolling_std'] = (
                df['response_time'].rolling(window=window_size, min_periods=1).std().fillna(0)
            )
        
        return stat_df
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        text_column: str = 'message',
        categorical_columns: List[str] = ['service', 'level'],
        numerical_columns: List[str] = ['response_time']
    ) -> np.ndarray:
        """
        Fit preprocessor and transform data
        
        Args:
            df: Input DataFrame
            text_column: Column containing text data
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            
        Returns:
            Processed feature matrix
        """
        logger.info("Fitting preprocessor and transforming data...")
        
        all_features = []
        feature_names = []
        
        # Temporal features
        temporal_features = self.extract_temporal_features(df)
        if not temporal_features.empty:
            all_features.append(temporal_features.values)
            feature_names.extend(temporal_features.columns.tolist())
            logger.info(f"Extracted {len(temporal_features.columns)} temporal features")
        
        # Text features (TF-IDF)
        if text_column in df.columns:
            text_features = self.extract_text_features(
                df[text_column].fillna('').astype(str).tolist(),
                fit=True
            )
            all_features.append(text_features)
            feature_names.extend([f'tfidf_{i}' for i in range(text_features.shape[1])])
            logger.info(f"Extracted {text_features.shape[1]} text features")
        
        # Categorical features
        categorical_features = self.extract_categorical_features(
            df, categorical_columns, fit=True
        )
        if categorical_features.shape[1] > 0:
            all_features.append(categorical_features)
            feature_names.extend([f'{col}_encoded' for col in categorical_columns])
            logger.info(f"Extracted {categorical_features.shape[1]} categorical features")
        
        # Numerical features
        numerical_features_df = self.extract_numerical_features(df, numerical_columns)
        if not numerical_features_df.empty:
            if self.scale_numerical:
                numerical_features = self.numerical_scaler.fit_transform(
                    numerical_features_df.values
                )
            else:
                numerical_features = numerical_features_df.values
            
            all_features.append(numerical_features)
            feature_names.extend(numerical_features_df.columns.tolist())
            logger.info(f"Extracted {numerical_features_df.shape[1]} numerical features")
        
        # Statistical features
        statistical_features_df = self.extract_statistical_features(df)
        if not statistical_features_df.empty:
            all_features.append(statistical_features_df.values)
            feature_names.extend(statistical_features_df.columns.tolist())
            logger.info(f"Extracted {statistical_features_df.shape[1]} statistical features")
        
        # Combine all features
        if all_features:
            X = np.hstack(all_features)
        else:
            X = np.array([]).reshape(len(df), 0)
        
        self.feature_names = feature_names
        self.is_fitted = True
        
        logger.info(f"Total features extracted: {X.shape[1]}")
        
        return X
    
    def transform(
        self,
        df: pd.DataFrame,
        text_column: str = 'message',
        categorical_columns: List[str] = ['service', 'level'],
        numerical_columns: List[str] = ['response_time']
    ) -> np.ndarray:
        """
        Transform data using fitted preprocessor
        
        Args:
            df: Input DataFrame
            text_column: Column containing text data
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            
        Returns:
            Processed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        all_features = []
        
        # Temporal features
        temporal_features = self.extract_temporal_features(df)
        if not temporal_features.empty:
            all_features.append(temporal_features.values)
        
        # Text features
        if text_column in df.columns:
            text_features = self.extract_text_features(
                df[text_column].fillna('').astype(str).tolist(),
                fit=False
            )
            all_features.append(text_features)
        
        # Categorical features
        categorical_features = self.extract_categorical_features(
            df, categorical_columns, fit=False
        )
        if categorical_features.shape[1] > 0:
            all_features.append(categorical_features)
        
        # Numerical features
        numerical_features_df = self.extract_numerical_features(df, numerical_columns)
        if not numerical_features_df.empty:
            if self.scale_numerical:
                numerical_features = self.numerical_scaler.transform(
                    numerical_features_df.values
                )
            else:
                numerical_features = numerical_features_df.values
            
            all_features.append(numerical_features)
        
        # Statistical features
        statistical_features_df = self.extract_statistical_features(df)
        if not statistical_features_df.empty:
            all_features.append(statistical_features_df.values)
        
        # Combine
        if all_features:
            X = np.hstack(all_features)
        else:
            X = np.array([]).reshape(len(df), 0)
        
        return X
    
    def save(self, path: str):
        """Save preprocessor to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'numerical_scaler': self.numerical_scaler,
                'categorical_encoders': self.categorical_encoders,
                'feature_names': self.feature_names,
                'max_tfidf_features': self.max_tfidf_features,
                'scale_numerical': self.scale_numerical,
                'scaler_type': self.scaler_type,
                'is_fitted': self.is_fitted
            }, f)
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: str):
        """Load preprocessor from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.numerical_scaler = data['numerical_scaler']
        self.categorical_encoders = data['categorical_encoders']
        self.feature_names = data['feature_names']
        self.max_tfidf_features = data['max_tfidf_features']
        self.scale_numerical = data['scale_numerical']
        self.scaler_type = data['scaler_type']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {path}")


def create_sequences(
    data: np.ndarray,
    sequence_length: int = 10,
    stride: int = 1
) -> np.ndarray:
    """
    Create sequences for LSTM training
    
    Args:
        data: Input data (n_samples, n_features)
        sequence_length: Length of sequences
        stride: Step size between sequences
        
    Returns:
        Sequences (n_sequences, sequence_length, n_features)
    """
    sequences = []
    
    for i in range(0, len(data) - sequence_length + 1, stride):
        sequences.append(data[i:i + sequence_length])
    
    return np.array(sequences)