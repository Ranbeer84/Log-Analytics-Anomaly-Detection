"""
Pydantic schemas for analytics data
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class LogVolumeData(BaseModel):
    """Log volume data point"""
    timestamp: datetime
    total: int = Field(..., ge=0)
    errors: int = Field(..., ge=0)
    warnings: int = Field(..., ge=0)


class LogVolumeResponse(BaseModel):
    """Log volume response"""
    success: bool = True
    data: List[LogVolumeData]
    period: Dict[str, Any]


class ErrorRateData(BaseModel):
    """Error rate statistics"""
    total_logs: int = Field(..., ge=0)
    error_count: int = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=100)
    period_start: datetime
    period_end: datetime


class ErrorRateResponse(BaseModel):
    """Error rate response"""
    success: bool = True
    data: ErrorRateData


class TopErrorData(BaseModel):
    """Top error data"""
    message: str
    service: str
    count: int = Field(..., ge=1)
    first_seen: datetime
    last_seen: datetime


class TopErrorsResponse(BaseModel):
    """Top errors response"""
    success: bool = True
    data: List[TopErrorData]
    count: int = Field(..., ge=0)


class ServiceMetric(BaseModel):
    """Service metrics data"""
    service: str
    total_logs: int = Field(..., ge=0)
    errors: int = Field(..., ge=0)
    warnings: int = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=100)
    avg_response_time: Optional[float] = None


class ServiceMetricsResponse(BaseModel):
    """Service metrics response"""
    success: bool = True
    data: List[ServiceMetric]
    count: int = Field(..., ge=0)


class ResponseTimeStats(BaseModel):
    """Response time statistics"""
    avg: float = Field(..., ge=0)
    min: float = Field(..., ge=0)
    max: float = Field(..., ge=0)
    p50: Optional[float] = Field(None, ge=0)
    p95: Optional[float] = Field(None, ge=0)
    p99: Optional[float] = Field(None, ge=0)
    count: int = Field(..., ge=0)


class ResponseTimeResponse(BaseModel):
    """Response time response"""
    success: bool = True
    data: ResponseTimeStats


class AnomalyTypeData(BaseModel):
    """Anomaly data by type"""
    type: str
    count: int = Field(..., ge=0)
    avg_score: float = Field(..., ge=0, le=1)


class AnomalySummary(BaseModel):
    """Anomaly summary data"""
    total_anomalies: int = Field(..., ge=0)
    by_type: List[AnomalyTypeData]
    period_start: datetime
    period_end: datetime


class AnomalySummaryResponse(BaseModel):
    """Anomaly summary response"""
    success: bool = True
    data: AnomalySummary


class LogLevelData(BaseModel):
    """Log level distribution data"""
    level: str
    count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0, le=100)


class LogLevelResponse(BaseModel):
    """Log level distribution response"""
    success: bool = True
    data: List[LogLevelData]


class DashboardData(BaseModel):
    """Comprehensive dashboard data"""
    error_rate: ErrorRateData
    services: List[ServiceMetric]
    log_levels: List[LogLevelData]
    anomalies: AnomalySummary
    top_errors: List[TopErrorData]


class DashboardResponse(BaseModel):
    """Dashboard response"""
    success: bool = True
    data: DashboardData
    period: Dict[str, datetime]


class TimeSeriesQuery(BaseModel):
    """Time series query parameters"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    interval: str = Field("1h", pattern=r"^[0-9]+[smhd]$")
    service: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-02T00:00:00Z",
                "interval": "1h",
                "service": "api-service"
            }
        }


class AggregationQuery(BaseModel):
    """Aggregation query parameters"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    group_by: Optional[str] = Field(None, pattern=r"^(service|level|host)$")
    metrics: List[str] = Field(default_factory=lambda: ["count"])
    
    class Config:
        schema_extra = {
            "example": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-02T00:00:00Z",
                "group_by": "service",
                "metrics": ["count", "error_rate", "avg_response_time"]
            }
        }


class MetricThreshold(BaseModel):
    """Metric threshold configuration"""
    metric_name: str
    operator: str = Field(..., pattern=r"^(gt|gte|lt|lte|eq)$")
    value: float
    time_window_minutes: int = Field(5, ge=1, le=1440)
    
    class Config:
        schema_extra = {
            "example": {
                "metric_name": "error_rate",
                "operator": "gt",
                "value": 10.0,
                "time_window_minutes": 5
            }
        }