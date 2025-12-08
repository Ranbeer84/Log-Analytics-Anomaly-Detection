"""
Log entry data models
"""
from datetime import datetime
from typing import Optional, Dict, Any
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


class LogEntry(BaseModel):
    """Log entry model"""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    timestamp: datetime
    level: str  # INFO, WARN, ERROR, DEBUG, CRITICAL
    service: str
    message: str
    metadata: Optional[Dict[str, Any]] = None
    
    # Additional fields
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    endpoint: Optional[str] = None
    response_time: Optional[float] = None
    status_code: Optional[int] = None
    
    # System fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False
    anomaly_checked: bool = False
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "timestamp": "2025-02-12T14:22:44Z",
                "level": "ERROR",
                "service": "api-gateway",
                "message": "Failed login attempt for user",
                "metadata": {
                    "attempt_count": 3,
                    "source": "web"
                },
                "user_id": "user_123",
                "ip_address": "192.168.1.1",
                "endpoint": "/api/login"
            }
        }


class LogStats(BaseModel):
    """Log statistics model"""
    
    total_logs: int
    error_count: int
    warning_count: int
    info_count: int
    unique_services: int
    time_range: Dict[str, datetime]
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_logs": 15420,
                "error_count": 234,
                "warning_count": 567,
                "info_count": 14619,
                "unique_services": 8,
                "time_range": {
                    "start": "2025-02-01T00:00:00Z",
                    "end": "2025-02-12T23:59:59Z"
                }
            }
        }