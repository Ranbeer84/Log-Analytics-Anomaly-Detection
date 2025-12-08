"""
Alert data models
Models for alert management and notifications
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from enum import Enum


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


class AlertType(str, Enum):
    """Alert type enumeration"""
    ANOMALY = "anomaly"
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    SYSTEM = "system"


class AlertSeverity(str, Enum):
    """Alert severity enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status enumeration"""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class NotificationChannel(str, Enum):
    """Notification channel enumeration"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    TELEGRAM = "telegram"


class Alert(BaseModel):
    """Alert model"""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    
    # Alert Information
    alert_type: AlertType
    severity: AlertSeverity
    title: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=2000)
    
    # Details
    details: Optional[Dict[str, Any]] = None
    
    # Related Entities
    related_log_ids: List[str] = Field(default_factory=list)
    related_anomaly_ids: List[str] = Field(default_factory=list)
    
    # Affected Resources
    affected_services: List[str] = Field(default_factory=list)
    affected_endpoints: List[str] = Field(default_factory=list)
    
    # Metrics
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None
    occurrence_count: int = 1
    
    # Status
    status: AlertStatus = AlertStatus.PENDING
    
    # Notification
    notification_channels: List[NotificationChannel] = Field(default_factory=list)
    notified_at: Optional[datetime] = None
    notification_attempts: int = 0
    notification_errors: List[str] = Field(default_factory=list)
    
    # Resolution
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    # Auto-resolution
    auto_resolve_after_seconds: Optional[int] = None
    
    # Priority
    priority: int = Field(default=0, ge=0, le=10)  # 0 = lowest, 10 = highest
    
    # Tags
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "alert_type": "anomaly",
                "severity": "high",
                "title": "High Error Rate Detected",
                "message": "Error rate has exceeded 10% in the last 5 minutes",
                "details": {
                    "current_error_rate": 0.15,
                    "threshold": 0.10,
                    "affected_users": 523
                },
                "related_log_ids": ["65c9f7e8a1b2c3d4e5f6g7h8"],
                "affected_services": ["api-gateway", "auth-service"],
                "status": "pending",
                "notification_channels": ["email", "slack"],
                "priority": 8
            }
        }


class AlertRule(BaseModel):
    """Alert rule configuration"""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    
    # Rule Info
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    is_enabled: bool = True
    
    # Trigger Conditions
    condition_type: str  # e.g., "threshold", "rate", "pattern"
    metric: str  # e.g., "error_rate", "response_time", "log_count"
    operator: str  # e.g., "gt", "lt", "eq", "gte", "lte"
    threshold: float
    
    # Time Window
    window_seconds: int = 300  # 5 minutes default
    
    # Filters
    service_filter: Optional[List[str]] = None
    level_filter: Optional[List[str]] = None
    
    # Alert Configuration
    alert_type: AlertType = AlertType.THRESHOLD
    severity: AlertSeverity = AlertSeverity.MEDIUM
    title_template: str
    message_template: str
    
    # Notification
    notification_channels: List[NotificationChannel] = Field(default_factory=list)
    recipients: List[str] = Field(default_factory=list)  # Email addresses or user IDs
    
    # Rate Limiting
    cooldown_seconds: int = 300  # Don't send same alert within 5 minutes
    last_triggered_at: Optional[datetime] = None
    
    # Auto-resolution
    auto_resolve: bool = True
    auto_resolve_after_seconds: int = 3600  # 1 hour
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    
    # Statistics
    trigger_count: int = 0
    last_evaluation_at: Optional[datetime] = None
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "name": "High Error Rate",
                "description": "Alert when error rate exceeds 10%",
                "is_enabled": True,
                "condition_type": "threshold",
                "metric": "error_rate",
                "operator": "gt",
                "threshold": 0.10,
                "window_seconds": 300,
                "service_filter": ["api-gateway"],
                "level_filter": ["ERROR"],
                "alert_type": "threshold",
                "severity": "high",
                "title_template": "High Error Rate in {service}",
                "message_template": "Error rate {value} exceeds threshold {threshold}",
                "notification_channels": ["email", "slack"],
                "cooldown_seconds": 300
            }
        }


class NotificationLog(BaseModel):
    """Log of sent notifications"""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    
    alert_id: PyObjectId
    channel: NotificationChannel
    
    # Recipients
    recipient: str  # Email, user ID, webhook URL, etc.
    
    # Status
    sent_at: datetime = Field(default_factory=datetime.utcnow)
    status: str  # "success", "failed", "pending"
    
    # Error info
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Response
    response_code: Optional[int] = None
    response_data: Optional[Dict[str, Any]] = None
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class AlertStats(BaseModel):
    """Alert statistics"""
    
    total_alerts: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    by_status: Dict[str, int]
    
    # Time-based
    alerts_last_hour: int
    alerts_last_day: int
    alerts_last_week: int
    
    # Resolution
    average_resolution_time_seconds: float
    acknowledged_rate: float
    resolved_rate: float
    
    # Top triggers
    top_services: List[Dict[str, Any]]
    top_alert_types: List[Dict[str, Any]]
    
    time_range: Dict[str, datetime]
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_alerts": 156,
                "by_type": {
                    "anomaly": 89,
                    "threshold": 45,
                    "pattern": 22
                },
                "by_severity": {
                    "low": 40,
                    "medium": 70,
                    "high": 40,
                    "critical": 6
                },
                "by_status": {
                    "pending": 12,
                    "sent": 34,
                    "acknowledged": 56,
                    "resolved": 54
                },
                "alerts_last_hour": 8,
                "alerts_last_day": 45,
                "alerts_last_week": 156,
                "average_resolution_time_seconds": 1850.5,
                "acknowledged_rate": 0.85,
                "resolved_rate": 0.75,
                "top_services": [
                    {"service": "api-gateway", "count": 67},
                    {"service": "auth-service", "count": 45}
                ],
                "time_range": {
                    "start": "2025-02-05T00:00:00Z",
                    "end": "2025-02-12T23:59:59Z"
                }
            }
        }