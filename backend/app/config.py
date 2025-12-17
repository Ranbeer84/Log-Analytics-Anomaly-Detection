"""
Configuration management using Pydantic Settings
"""
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "AI Log Analytics"
    ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # API
    API_V1_PREFIX: str = "/api"
    SECRET_KEY: str = "your-secret-key-change-this-in-production-min-32-chars"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    MONGODB_URL: str = "mongodb://admin:adminpass123@mongodb:27017/"
    MONGODB_DB_NAME: str = "log_analytics"
    
    # Redis
    REDIS_URL: str = "redis://redis:6379"
    REDIS_STREAM_NAME: str = "log_stream"
    REDIS_CONSUMER_GROUP: str = "log_processors"
    
    # ML Service
    ML_SERVICE_URL: str = "http://ml_service:8000"
    ANOMALY_THRESHOLD: float = 0.8
    
    # Email Alerts
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    ALERT_FROM_EMAIL: str = "alerts@loganalytics.com"
    ALERT_TO_EMAIL: str = "admin@company.com"
    
    # Webhook
    SLACK_WEBHOOK_URL: str = ""
    WEBHOOK_ENABLED: bool = False
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:80"]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # Log Retention
    LOG_RETENTION_DAYS: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()