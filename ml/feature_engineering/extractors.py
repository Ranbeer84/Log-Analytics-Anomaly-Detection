"""
Feature Extraction for Log Anomaly Detection
"""
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class LogFeatureExtractor:
    """Extract features from log entries for ML models"""
    
    def __init__(self):
        self.feature_names = []
        self.severity_map = {
            'debug': 0,
            'info': 1,
            'warning': 2,
            'error': 3,
            'critical': 4
        }
        
    def extract_features(self, log_entry: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a single log entry
        
        Args:
            log_entry: Dictionary containing log data
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # 1. Severity level (numerical)
        severity = log_entry.get('severity', 'info').lower()
        features.append(self.severity_map.get(severity, 1))
        
        # 2. Response time (if available)
        response_time = log_entry.get('response_time', 0)
        features.append(float(response_time))
        
        # 3. HTTP status code features
        status_code = log_entry.get('status_code', 200)
        features.append(float(status_code))
        features.append(1.0 if status_code >= 400 else 0.0)  # Is error
        features.append(1.0 if status_code >= 500 else 0.0)  # Is server error
        
        # 4. Time-based features
        timestamp = log_entry.get('timestamp')
        if timestamp:
            dt = self._parse_timestamp(timestamp)
            features.extend(self._extract_time_features(dt))
        else:
            features.extend([0, 0, 0, 0])
        
        # 5. Message features
        message = log_entry.get('message', '')
        features.extend(self._extract_message_features(message))
        
        # 6. Service/source features
        service = log_entry.get('service', '')
        features.extend(self._extract_service_features(service))
        
        # 7. Additional metadata features
        metadata = log_entry.get('metadata', {})
        features.extend(self._extract_metadata_features(metadata))
        
        return np.array(features, dtype=np.float32)
    
    def extract_batch_features(self, log_entries: List[Dict]) -> np.ndarray:
        """
        Extract features from multiple log entries
        
        Args:
            log_entries: List of log dictionaries
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features_list = []
        
        for log_entry in log_entries:
            try:
                features = self.extract_features(log_entry)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features: {str(e)}")
                continue
        
        if not features_list:
            raise ValueError("No features could be extracted")
        
        return np.vstack(features_list)
    
    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """Parse timestamp to datetime object"""
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return datetime.utcnow()
        else:
            return datetime.utcnow()
    
    def _extract_time_features(self, dt: datetime) -> List[float]:
        """Extract time-based features"""
        return [
            float(dt.hour),  # Hour of day (0-23)
            float(dt.weekday()),  # Day of week (0-6)
            1.0 if dt.weekday() >= 5 else 0.0,  # Is weekend
            1.0 if 9 <= dt.hour < 17 else 0.0  # Is business hours
        ]
    
    def _extract_message_features(self, message: str) -> List[float]:
        """Extract features from log message"""
        if not message:
            return [0, 0, 0, 0, 0, 0]
        
        message_lower = message.lower()
        
        return [
            float(len(message)),  # Message length
            float(len(message.split())),  # Word count
            1.0 if any(kw in message_lower for kw in ['error', 'exception', 'failed']) else 0.0,
            1.0 if any(kw in message_lower for kw in ['timeout', 'slow']) else 0.0,
            1.0 if any(kw in message_lower for kw in ['null', 'none', 'undefined']) else 0.0,
            float(len(re.findall(r'\d+', message)))  # Number count
        ]
    
    def _extract_service_features(self, service: str) -> List[float]:
        """Extract features from service name"""
        if not service:
            return [0, 0]
        
        # Hash service name to a numerical value
        service_hash = hash(service) % 1000 / 1000.0
        
        return [
            service_hash,
            float(len(service))
        ]
    
    def _extract_metadata_features(self, metadata: Dict) -> List[float]:
        """Extract features from metadata"""
        features = []
        
        # User agent features
        user_agent = metadata.get('user_agent', '')
        features.append(1.0 if 'bot' in user_agent.lower() else 0.0)
        
        # IP-based features
        ip = metadata.get('ip', '')
        features.append(1.0 if ip.startswith('10.') or ip.startswith('192.168.') else 0.0)
        
        # Request features
        method = metadata.get('method', 'GET')
        features.append(1.0 if method in ['POST', 'PUT', 'DELETE'] else 0.0)
        
        # Path complexity
        path = metadata.get('path', '')
        features.append(float(len(path.split('/'))))
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features"""
        return [
            'severity_level',
            'response_time',
            'status_code',
            'is_error',
            'is_server_error',
            'hour_of_day',
            'day_of_week',
            'is_weekend',
            'is_business_hours',
            'message_length',
            'word_count',
            'has_error_keyword',
            'has_timeout_keyword',
            'has_null_keyword',
            'number_count',
            'service_hash',
            'service_length',
            'is_bot',
            'is_internal_ip',
            'is_mutating_method',
            'path_complexity'
        ]


class SequenceFeatureExtractor:
    """Extract sequence-based features for temporal patterns"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        
    def extract_window_features(
        self,
        log_window: List[Dict]
    ) -> np.ndarray:
        """
        Extract features from a window of sequential logs
        
        Args:
            log_window: List of recent log entries
            
        Returns:
            Window feature vector
        """
        if not log_window:
            return np.zeros(10)
        
        features = []
        
        # 1. Error rate in window
        error_count = sum(1 for log in log_window 
                         if log.get('status_code', 200) >= 400)
        features.append(error_count / len(log_window))
        
        # 2. Average response time
        response_times = [log.get('response_time', 0) for log in log_window]
        features.append(np.mean(response_times))
        features.append(np.std(response_times))
        
        # 3. Severity distribution
        severities = [log.get('severity', 'info') for log in log_window]
        severity_counts = Counter(severities)
        features.append(severity_counts.get('error', 0) / len(log_window))
        features.append(severity_counts.get('critical', 0) / len(log_window))
        
        # 4. Service diversity
        services = [log.get('service', '') for log in log_window]
        unique_services = len(set(services))
        features.append(unique_services / max(len(log_window), 1))
        
        # 5. Time gaps
        timestamps = [log.get('timestamp') for log in log_window if log.get('timestamp')]
        if len(timestamps) > 1:
            time_gaps = []
            for i in range(1, len(timestamps)):
                t1 = self._parse_timestamp(timestamps[i-1])
                t2 = self._parse_timestamp(timestamps[i])
                gap = (t2 - t1).total_seconds()
                time_gaps.append(gap)
            
            features.append(np.mean(time_gaps))
            features.append(np.std(time_gaps))
            features.append(np.max(time_gaps))
        else:
            features.extend([0, 0, 0])
        
        # 6. Log volume spike indicator
        features.append(min(len(log_window) / self.window_size, 2.0))
        
        return np.array(features, dtype=np.float32)
    
    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """Parse timestamp to datetime object"""
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return datetime.utcnow()
        else:
            return datetime.utcnow()