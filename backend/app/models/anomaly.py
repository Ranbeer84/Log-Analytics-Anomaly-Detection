"""
Anomaly data models
Models for storing and managing detected anomalies
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


class Anomaly(BaseModel):
    """Anomaly detection result model"""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    log_id: PyObjectId  # Reference to the log entry
    timestamp: datetime  # Timestamp of the anomalous event
    
    # Anomaly Detection Results
    anomaly_score: float = Field(..., ge=0.0, le=1.0)  # Score between 0 and 1
    is_anomaly: bool  # True if score exceeds threshold
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)  # Confidence level
    
    # Anomaly Classification
    anomaly_type: Optional[str] = None  # e.g., "error_spike", "slow_response", "unusual_pattern"
    severity: str = Field(default="medium")  # low, medium, high, critical
    
    # Model Information
    model_name: str  # e.g., "isolation_forest", "lstm_autoencoder"
    model_version: str  # Model version used for detection
    
    # Feature Information
    features: Dict[str, Any] = Field(default_factory=dict)  # Features used for detection
    feature_importance: Optional[Dict[str, float]] = None  # Feature importance scores
    
    # Context
    context: Optional[Dict[str, Any]] = None  # Additional context
    related_anomalies: List[str] = Field(default_factory=list)  # Related anomaly IDs
    
    # Status
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    is_confirmed: bool = False  # Manual confirmation
    is_false_positive: bool = False  # Marked as false positive
    
    # Notes
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "log_id": "65c9f7e8a1b2c3d4e5f6g7h8",
                "timestamp": "2025-02-12T14:22:44Z",
                "anomaly_score": 0.85,
                "is_anomaly": True,
                "confidence": 0.92,
                "anomaly_type": "error_spike",
                "severity": "high",
                "model_name": "isolation_forest",
                "model_version": "v1.0.0",
                "features": {
                    "error_rate": 0.45,
                    "response_time": 1250.5,
                    "request_count": 523
                },
                "detected_at": "2025-02-12T14:23:00Z"
            }
        }


class AnomalyPattern(BaseModel):
    """Pattern of related anomalies"""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    pattern_type: str  # e.g., "recurring", "cascading", "clustered"
    anomaly_ids: List[str]  # List of anomaly IDs in this pattern
    
    # Pattern Characteristics
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    frequency: Optional[float] = None  # Anomalies per hour
    
    # Affected Resources
    affected_services: List[str] = Field(default_factory=list)
    affected_endpoints: List[str] = Field(default_factory=list)
    
    # Pattern Analysis
    severity: str = Field(default="medium")
    impact_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Description
    description: Optional[str] = None
    root_cause: Optional[str] = None
    
    # Status
    is_ongoing: bool = True
    resolved_at: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class AnomalyStats(BaseModel):
    """Anomaly statistics"""
    
    total_anomalies: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    by_service: Dict[str, int]
    
    # Time-based stats
    anomalies_last_hour: int
    anomalies_last_day: int
    anomalies_last_week: int
    
    # Accuracy metrics
    confirmed_count: int
    false_positive_count: int
    accuracy_rate: float
    
    # Patterns
    active_patterns: int
    resolved_patterns: int
    
    time_range: Dict[str, datetime]
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_anomalies": 234,
                "by_type": {
                    "error_spike": 120,
                    "slow_response": 80,
                    "unusual_pattern": 34
                },
                "by_severity": {
                    "low": 100,
                    "medium": 90,
                    "high": 40,
                    "critical": 4
                },
                "by_service": {
                    "api-gateway": 100,
                    "auth-service": 80,
                    "payment-service": 54
                },
                "anomalies_last_hour": 12,
                "anomalies_last_day": 45,
                "anomalies_last_week": 234,
                "confirmed_count": 180,
                "false_positive_count": 20,
                "accuracy_rate": 0.90,
                "active_patterns": 3,
                "resolved_patterns": 15,
                "time_range": {
                    "start": "2025-02-05T00:00:00Z",
                    "end": "2025-02-12T23:59:59Z"
                }
            }
        }


class MLModelMetadata(BaseModel):
    """ML model metadata"""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    model_name: str
    version: str
    model_type: str  # e.g., "isolation_forest", "lstm", "autoencoder"
    
    # Hyperparameters
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance Metrics
    metrics: Dict[str, float] = Field(default_factory=dict)  # precision, recall, f1, etc.
    
    # Training Info
    training_data_size: Optional[int] = None
    training_duration_seconds: Optional[float] = None
    trained_at: Optional[datetime] = None
    
    # Deployment Info
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = False
    activated_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None
    
    # File Info
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    
    # Additional Info
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "model_name": "isolation_forest",
                "version": "v1.0.0",
                "model_type": "isolation_forest",
                "hyperparameters": {
                    "n_estimators": 100,
                    "contamination": 0.1,
                    "max_samples": "auto"
                },
                "metrics": {
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1_score": 0.90
                },
                "training_data_size": 10000,
                "is_active": True,
                "file_path": "/app/saved_models/isolation_forest_v1.pkl"
            }
        }