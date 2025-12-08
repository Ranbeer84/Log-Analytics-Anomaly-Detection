"""
Pydantic schemas for API request/response validation
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator


class LogCreate(BaseModel):
    """Schema for creating a log entry"""
    
    timestamp: datetime
    level: str = Field(..., pattern="^(DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL)$")
    service: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=5000)
    metadata: Optional[Dict[str, Any]] = None
    
    # Optional fields
    user_id: Optional[str] = Field(None, max_length=100)
    ip_address: Optional[str] = Field(None, max_length=45)
    endpoint: Optional[str] = Field(None, max_length=500)
    response_time: Optional[float] = Field(None, ge=0)
    status_code: Optional[int] = Field(None, ge=100, le=599)
    
    @field_validator('level')
    @classmethod
    def normalize_level(cls, v: str) -> str:
        """Normalize log level to uppercase"""
        return v.upper()
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-02-12T14:22:44Z",
                "level": "ERROR",
                "service": "api-gateway",
                "message": "Failed login attempt",
                "metadata": {
                    "attempt_count": 3
                },
                "user_id": "user_123",
                "ip_address": "192.168.1.1"
            }
        }


class LogResponse(BaseModel):
    """Schema for log response"""
    
    id: str = Field(..., alias="_id")
    timestamp: datetime
    level: str
    service: str
    message: str
    metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    endpoint: Optional[str] = None
    response_time: Optional[float] = None
    status_code: Optional[int] = None
    created_at: datetime
    
    class Config:
        populate_by_name = True


class LogBatchCreate(BaseModel):
    """Schema for batch log creation"""
    
    logs: List[LogCreate] = Field(..., max_length=1000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "logs": [
                    {
                        "timestamp": "2025-02-12T14:22:44Z",
                        "level": "ERROR",
                        "service": "api-gateway",
                        "message": "Failed login"
                    },
                    {
                        "timestamp": "2025-02-12T14:22:45Z",
                        "level": "INFO",
                        "service": "api-gateway",
                        "message": "User logged in"
                    }
                ]
            }
        }


class LogQuery(BaseModel):
    """Schema for querying logs"""
    
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    level: Optional[str] = None
    service: Optional[str] = None
    search: Optional[str] = None
    limit: int = Field(50, ge=1, le=1000)
    skip: int = Field(0, ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2025-02-01T00:00:00Z",
                "end_date": "2025-02-12T23:59:59Z",
                "level": "ERROR",
                "service": "api-gateway",
                "limit": 100,
                "skip": 0
            }
        }


class LogIngestResponse(BaseModel):
    """Response after log ingestion"""
    
    status: str
    message: str
    log_id: Optional[str] = None
    stream_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Log ingested successfully",
                "log_id": "65c9f7e8a1b2c3d4e5f6g7h8",
                "stream_id": "1707748964123-0"
            }
        }


class LogBatchIngestResponse(BaseModel):
    """Response after batch log ingestion"""
    
    status: str
    message: str
    ingested_count: int
    failed_count: int
    errors: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Batch ingestion completed",
                "ingested_count": 98,
                "failed_count": 2,
                "errors": ["Invalid timestamp at index 5", "Missing service at index 12"]
            }
        }