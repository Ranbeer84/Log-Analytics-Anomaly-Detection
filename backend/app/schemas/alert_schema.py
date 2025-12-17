"""
Pydantic schemas for alert data
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    """Alert status"""
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class AlertRuleBase(BaseModel):
    """Base alert rule schema"""
    name: str = Field(..., min_length=1, max_length=200)
    description: str
    condition: Dict[str, Any]
    severity: AlertSeverity
    enabled: bool = True
    threshold: Optional[Dict[str, Any]] = None
    notification_channels: List[str] = Field(default_factory=list)


class AlertRuleCreate(AlertRuleBase):
    """Create alert rule schema"""
    pass


class AlertRuleUpdate(BaseModel):
    """Update alert rule schema"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None
    severity: Optional[AlertSeverity] = None
    enabled: Optional[bool] = None
    threshold: Optional[Dict[str, Any]] = None
    notification_channels: Optional[List[str]] = None


class AlertRule(AlertRuleBase):
    """Alert rule response schema"""
    id: str
    created_at: datetime
    updated_at: datetime
    last_triggered: Optional[datetime] = None
    trigger_count: int = Field(0, ge=0)
    
    class Config:
        from_attributes = True


class AlertRuleList(BaseModel):
    """List of alert rules"""
    success: bool = True
    data: List[AlertRule]
    count: int = Field(..., ge=0)


class AlertBase(BaseModel):
    """Base alert schema"""
    rule_id: str
    title: str = Field(..., min_length=1, max_length=200)
    description: str
    severity: AlertSeverity
    source: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class AlertCreate(AlertBase):
    """Create alert schema"""
    pass


class Alert(AlertBase):
    """Alert response schema"""
    id: str
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_note: Optional[str] = None
    notification_sent: bool = False
    
    class Config:
        from_attributes = True


class AlertList(BaseModel):
    """List of alerts"""
    success: bool = True
    data: List[Alert]
    count: int = Field(..., ge=0)


class AlertDetail(BaseModel):
    """Detailed alert information"""
    success: bool = True
    data: Alert


class AlertAcknowledge(BaseModel):
    """Acknowledge alert request"""
    acknowledged_by: str = Field(..., min_length=1, max_length=100)


class AlertResolve(BaseModel):
    """Resolve alert request"""
    resolved_by: str = Field(..., min_length=1, max_length=100)
    resolution_note: Optional[str] = Field(None, max_length=1000)


class AlertStatistics(BaseModel):
    """Alert statistics"""
    total: int = Field(..., ge=0)
    by_severity: Dict[str, int]
    by_status: Dict[str, int]
    period_start: datetime
    period_end: datetime


class AlertStatisticsResponse(BaseModel):
    """Alert statistics response"""
    success: bool = True
    data: AlertStatistics


class AlertCheckResult(BaseModel):
    """Alert check result"""
    metric: str
    value: float
    threshold: float
    window_minutes: Optional[int] = None
    percentile: Optional[int] = None
    total_logs: Optional[int] = None
    error_count: Optional[int] = None
    sample_count: Optional[int] = None


class AlertCheckResponse(BaseModel):
    """Alert check response"""
    success: bool = True
    alert_triggered: bool
    data: Optional[AlertCheckResult] = None
    message: Optional[str] = None


class AlertCondition(BaseModel):
    """Alert condition definition"""
    metric: str = Field(..., pattern=r"^(error_rate|response_time|log_volume|anomaly_score)$")
    operator: str = Field(..., pattern=r"^(gt|gte|lt|lte|eq|ne)$")
    threshold: float
    time_window_minutes: int = Field(5, ge=1, le=1440)
    aggregation: Optional[str] = Field("avg", pattern=r"^(avg|sum|min|max|count|p50|p95|p99)$")
    
    class Config:
        schema_extra = {
            "example": {
                "metric": "error_rate",
                "operator": "gt",
                "threshold": 10.0,
                "time_window_minutes": 5,
                "aggregation": "avg"
            }
        }


class NotificationChannel(BaseModel):
    """Notification channel configuration"""
    type: str = Field(..., pattern=r"^(email|slack|webhook|pagerduty)$")
    config: Dict[str, Any]
    enabled: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "type": "slack",
                "config": {
                    "webhook_url": "https://hooks.slack.com/services/...",
                    "channel": "#alerts"
                },
                "enabled": True
            }
        }


class AlertRuleConditionConfig(BaseModel):
    """Complete alert rule condition configuration"""
    conditions: List[AlertCondition]
    logical_operator: str = Field("AND", pattern=r"^(AND|OR)$")
    consecutive_breaches: int = Field(1, ge=1, le=10)
    
    class Config:
        schema_extra = {
            "example": {
                "conditions": [
                    {
                        "metric": "error_rate",
                        "operator": "gt",
                        "threshold": 10.0,
                        "time_window_minutes": 5
                    }
                ],
                "logical_operator": "AND",
                "consecutive_breaches": 2
            }
        }


class AlertResponse(BaseModel):
    """Generic alert response"""
    success: bool
    alert_id: Optional[str] = None
    message: str


class AlertRuleResponse(BaseModel):
    """Generic alert rule response"""
    success: bool
    rule_id: Optional[str] = None
    message: str