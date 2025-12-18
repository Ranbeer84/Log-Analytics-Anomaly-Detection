"""
Advanced feature transformations for log data
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogLevelEncoder(BaseEstimator, TransformerMixin):
    """
    Encode log levels with severity ordering
    
    Maps: DEBUG=0, INFO=1, WARNING=2, ERROR=3, CRITICAL=4
    """
    
    def __init__(self):
        self.level_map = {
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'WARN': 2,
            'ERROR': 3,
            'CRITICAL': 4,
            'FATAL': 4
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.map(lambda x: self.level_map.get(x.upper(), 1)).values.reshape(-1, 1)
        elif isinstance(X, np.ndarray):
            return np.array([self.level_map.get(str(x).upper(), 1) for x in X]).reshape(-1, 1)
        return X


class ServiceFrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encode services by their frequency in training data
    
    Rare services get higher values (potentially more anomalous)
    """
    
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.frequency_map = {}
        self.default_value = 0.5
    
    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            service_counts = X.value_counts()
        elif isinstance(X, np.ndarray):
            unique, counts = np.unique(X, return_counts=True)
            service_counts = pd.Series(counts, index=unique)
        else:
            return self
        
        total = service_counts.sum()
        
        # Inverse frequency (rare = high value)
        self.frequency_map = {}
        for service, count in service_counts.items():
            freq = (count + self.smoothing) / (total + self.smoothing * len(service_counts))
            self.frequency_map[service] = 1.0 / freq
        
        # Normalize to [0, 1]
        max_val = max(self.frequency_map.values())
        self.frequency_map = {k: v/max_val for k, v in self.frequency_map.items()}
        self.default_value = np.mean(list(self.frequency_map.values()))
        
        return self
    
    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.map(lambda x: self.frequency_map.get(x, self.default_value)).values.reshape(-1, 1)
        elif isinstance(X, np.ndarray):
            return np.array([
                self.frequency_map.get(str(x), self.default_value) for x in X
            ]).reshape(-1, 1)
        return X


class ResponseTimeTransformer(BaseEstimator, TransformerMixin):
    """
    Advanced response time transformations
    
    Creates multiple derived features from response time
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.median_ = None
        self.p95_ = None
    
    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            values = X.values
        elif isinstance(X, np.ndarray):
            values = X.flatten()
        else:
            return self
        
        # Remove NaN
        values = values[~np.isnan(values)]
        
        if len(values) > 0:
            self.mean_ = np.mean(values)
            self.std_ = np.std(values)
            self.median_ = np.median(values)
            self.p95_ = np.percentile(values, 95)
        else:
            self.mean_ = 0
            self.std_ = 1
            self.median_ = 0
            self.p95_ = 0
        
        return self
    
    def transform(self, X):
        if isinstance(X, pd.Series):
            values = X.values
        elif isinstance(X, np.ndarray):
            values = X.flatten()
        else:
            return X
        
        # Fill NaN with median
        values = np.where(np.isnan(values), self.median_, values)
        
        features = []
        
        # Original value (log-transformed)
        features.append(np.log1p(values))
        
        # Z-score (standardized)
        if self.std_ > 0:
            features.append((values - self.mean_) / self.std_)
        else:
            features.append(np.zeros_like(values))
        
        # Deviation from median
        features.append(values - self.median_)
        
        # Is outlier (above p95)
        features.append((values > self.p95_).astype(float))
        
        # Square root transformation
        features.append(np.sqrt(values))
        
        return np.column_stack(features)


class TemporalPatternExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal patterns indicating anomalous behavior
    
    Features:
    - Time-based patterns (night vs day, weekend vs weekday)
    - Cyclical encodings
    - Interaction features
    """
    
    def __init__(self):
        self.day_hour_counts = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        X should have 'timestamp' column
        """
        if 'timestamp' not in X.columns:
            return self
        
        timestamps = pd.to_datetime(X['timestamp'])
        
        # Learn typical patterns
        day_hour = list(zip(timestamps.dt.dayofweek, timestamps.dt.hour))
        unique, counts = np.unique(day_hour, axis=0, return_counts=True)
        
        for (day, hour), count in zip(unique, counts):
            self.day_hour_counts[(day, hour)] = count
        
        return self
    
    def transform(self, X: pd.DataFrame):
        if 'timestamp' not in X.columns:
            return np.zeros((len(X), 1))
        
        timestamps = pd.to_datetime(X['timestamp'])
        
        features = []
        
        # Hour of day
        hour = timestamps.dt.hour
        features.append(hour.values.reshape(-1, 1))
        
        # Day of week
        dow = timestamps.dt.dayofweek
        features.append(dow.values.reshape(-1, 1))
        
        # Is night time (10 PM - 6 AM)
        is_night = ((hour < 6) | (hour >= 22)).astype(int).values.reshape(-1, 1)
        features.append(is_night)
        
        # Is weekend
        is_weekend = (dow >= 5).astype(int).values.reshape(-1, 1)
        features.append(is_weekend)
        
        # Cyclical hour encoding
        hour_sin = np.sin(2 * np.pi * hour / 24).values.reshape(-1, 1)
        hour_cos = np.cos(2 * np.pi * hour / 24).values.reshape(-1, 1)
        features.extend([hour_sin, hour_cos])
        
        # Cyclical day encoding
        dow_sin = np.sin(2 * np.pi * dow / 7).values.reshape(-1, 1)
        dow_cos = np.cos(2 * np.pi * dow / 7).values.reshape(-1, 1)
        features.extend([dow_sin, dow_cos])
        
        # Pattern rarity (how unusual is this day-hour combination)
        if self.day_hour_counts:
            max_count = max(self.day_hour_counts.values())
            rarity = []
            for d, h in zip(dow, hour):
                count = self.day_hour_counts.get((d, h), 0)
                # Inverse frequency (rare = high value)
                rarity.append(1.0 - (count / max_count))
            features.append(np.array(rarity).reshape(-1, 1))
        
        return np.hstack(features)


class MessageLengthTransformer(BaseEstimator, TransformerMixin):
    """
    Extract features from message length
    
    Anomalous logs often have unusual lengths
    """
    
    def __init__(self):
        self.mean_length_ = None
        self.std_length_ = None
    
    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            lengths = X.str.len().values
        elif isinstance(X, (list, np.ndarray)):
            lengths = np.array([len(str(x)) for x in X])
        else:
            return self
        
        self.mean_length_ = np.mean(lengths)
        self.std_length_ = np.std(lengths)
        
        return self
    
    def transform(self, X):
        if isinstance(X, pd.Series):
            lengths = X.str.len().values
        elif isinstance(X, (list, np.ndarray)):
            lengths = np.array([len(str(x)) for x in X])
        else:
            return X
        
        features = []
        
        # Raw length (log-transformed)
        features.append(np.log1p(lengths))
        
        # Standardized length
        if self.std_length_ > 0:
            standardized = (lengths - self.mean_length_) / self.std_length_
            features.append(standardized)
        else:
            features.append(np.zeros_like(lengths, dtype=float))
        
        # Is unusually long
        threshold = self.mean_length_ + 2 * self.std_length_
        features.append((lengths > threshold).astype(float))
        
        # Is unusually short
        threshold = max(0, self.mean_length_ - 2 * self.std_length_)
        features.append((lengths < threshold).astype(float))
        
        return np.column_stack(features)


class PCATransformer(BaseEstimator, TransformerMixin):
    """
    Apply PCA for dimensionality reduction
    
    Useful for high-dimensional features like TF-IDF
    """
    
    def __init__(self, n_components: int = 20, whiten: bool = True):
        self.n_components = n_components
        self.whiten = whiten
        self.pca = None
    
    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        
        # Adjust components if necessary
        n_components = min(self.n_components, n_samples, n_features)
        
        self.pca = PCA(n_components=n_components, whiten=self.whiten)
        self.pca.fit(X)
        
        logger.info(
            f"PCA: {n_features} features -> {n_components} components "
            f"(variance explained: {self.pca.explained_variance_ratio_.sum():.3f})"
        )
        
        return self
    
    def transform(self, X):
        if self.pca is None:
            raise ValueError("Transformer must be fitted before transform")
        
        return self.pca.transform(X)


class AggregationFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract aggregation features from grouped data
    
    Example: For each service, compute statistics
    """
    
    def __init__(self, group_col: str, agg_cols: List[str]):
        self.group_col = group_col
        self.agg_cols = agg_cols
        self.group_stats = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            return self
        
        if self.group_col not in X.columns:
            return self
        
        # Compute group statistics
        for col in self.agg_cols:
            if col not in X.columns:
                continue
            
            stats = X.groupby(self.group_col)[col].agg(['mean', 'std', 'median'])
            self.group_stats[col] = stats.to_dict('index')
        
        return self
    
    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            return np.zeros((len(X), 1))
        
        if self.group_col not in X.columns:
            return np.zeros((len(X), 1))
        
        features = []
        
        for col in self.agg_cols:
            if col not in X.columns or col not in self.group_stats:
                continue
            
            stats = self.group_stats[col]
            
            # Deviation from group mean
            group_means = X[self.group_col].map(
                lambda g: stats.get(g, {}).get('mean', X[col].mean())
            )
            deviation = X[col] - group_means
            features.append(deviation.values.reshape(-1, 1))
            
            # Z-score within group
            group_stds = X[self.group_col].map(
                lambda g: stats.get(g, {}).get('std', X[col].std())
            )
            z_score = np.where(
                group_stds > 0,
                deviation / group_stds,
                0
            )
            features.append(z_score.reshape(-1, 1))
        
        if features:
            return np.hstack(features)
        else:
            return np.zeros((len(X), 1))


def create_advanced_features(
    df: pd.DataFrame,
    include_pca: bool = False,
    pca_components: int = 20
) -> np.ndarray:
    """
    Create all advanced features
    
    Args:
        df: Input DataFrame
        include_pca: Whether to apply PCA
        pca_components: Number of PCA components
        
    Returns:
        Feature matrix
    """
    features = []
    
    # Log level encoding
    if 'level' in df.columns:
        encoder = LogLevelEncoder()
        level_features = encoder.fit_transform(df['level'])
        features.append(level_features)
    
    # Service encoding
    if 'service' in df.columns:
        encoder = ServiceFrequencyEncoder()
        service_features = encoder.fit_transform(df['service'])
        features.append(service_features)
    
    # Response time features
    if 'response_time' in df.columns:
        transformer = ResponseTimeTransformer()
        rt_features = transformer.fit_transform(df['response_time'])
        features.append(rt_features)
    
    # Temporal patterns
    if 'timestamp' in df.columns:
        transformer = TemporalPatternExtractor()
        temporal_features = transformer.fit_transform(df)
        features.append(temporal_features)
    
    # Message length
    if 'message' in df.columns:
        transformer = MessageLengthTransformer()
        msg_features = transformer.fit_transform(df['message'])
        features.append(msg_features)
    
    # Combine
    if features:
        X = np.hstack(features)
        
        # Apply PCA if requested
        if include_pca and X.shape[1] > pca_components:
            pca = PCATransformer(n_components=pca_components)
            X = pca.fit_transform(X)
        
        return X
    else:
        return np.zeros((len(df), 1))