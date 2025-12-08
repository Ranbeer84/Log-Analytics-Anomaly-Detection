"""
Shared API dependencies
Reusable dependency injection functions for FastAPI routes
"""
from typing import Optional
from fastapi import Header, HTTPException, status, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
import redis.asyncio as redis
from datetime import datetime
import logging

from app.core.database import get_database
from app.core.redis_client import get_redis
from app.config import settings

logger = logging.getLogger(__name__)


# Database Dependencies
async def get_db() -> AsyncIOMotorDatabase:
    """
    Get database instance
    Use this in route parameters: db: AsyncIOMotorDatabase = Depends(get_db)
    """
    return get_database()


async def get_redis_client() -> redis.Redis:
    """
    Get Redis client instance
    Use this in route parameters: redis_client: redis.Redis = Depends(get_redis_client)
    """
    return get_redis()


# Rate Limiting Dependency
class RateLimiter:
    """Simple rate limiter using Redis"""
    
    def __init__(self, requests_per_minute: int = None):
        self.requests_per_minute = requests_per_minute or settings.RATE_LIMIT_PER_MINUTE
    
    async def __call__(
        self,
        client_ip: str = Header(None, alias="X-Real-IP"),
        redis_client: redis.Redis = Depends(get_redis_client)
    ):
        """Check rate limit for client IP"""
        if not client_ip:
            client_ip = "unknown"
        
        # Create rate limit key
        key = f"rate_limit:{client_ip}:{datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}"
        
        try:
            # Increment counter
            current = await redis_client.incr(key)
            
            # Set expiry on first request
            if current == 1:
                await redis_client.expire(key, 60)
            
            # Check limit
            if current > self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute."
                )
        
        except HTTPException:
            raise
        except Exception as e:
            # If Redis fails, allow the request (fail open)
            logger.error(f"Rate limiter error: {e}")
            pass


# API Key Authentication (Optional)
async def verify_api_key(
    x_api_key: Optional[str] = Header(None)
) -> bool:
    """
    Verify API key from header
    Use this for protected endpoints
    """
    # For now, just check if key exists
    # In production, validate against database
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required"
        )
    
    # TODO: Validate against stored API keys
    # For development, accept any key
    return True


# Pagination Dependency
class PaginationParams:
    """Pagination parameters"""
    
    def __init__(
        self,
        skip: int = 0,
        limit: int = 50,
        max_limit: int = 1000
    ):
        if skip < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Skip must be non-negative"
            )
        
        if limit < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be positive"
            )
        
        if limit > max_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Limit cannot exceed {max_limit}"
            )
        
        self.skip = skip
        self.limit = limit


# Date Range Dependency
class DateRangeParams:
    """Date range filter parameters"""
    
    def __init__(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        self.start_date = start_date
        self.end_date = end_date
        
        # Validate date range
        if start_date and end_date and start_date > end_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="start_date must be before end_date"
            )


# Service Health Check
async def check_service_health(
    db: AsyncIOMotorDatabase = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis_client)
) -> dict:
    """
    Check health of dependent services
    Returns status dictionary
    """
    health = {
        "mongodb": "unknown",
        "redis": "unknown"
    }
    
    # Check MongoDB
    try:
        await db.command("ping")
        health["mongodb"] = "healthy"
    except Exception as e:
        health["mongodb"] = f"unhealthy: {str(e)}"
        logger.error(f"MongoDB health check failed: {e}")
    
    # Check Redis
    try:
        await redis_client.ping()
        health["redis"] = "healthy"
    except Exception as e:
        health["redis"] = f"unhealthy: {str(e)}"
        logger.error(f"Redis health check failed: {e}")
    
    return health


# Request Logging Dependency
async def log_request_info(
    client_ip: str = Header(None, alias="X-Real-IP"),
    user_agent: str = Header(None, alias="User-Agent")
):
    """
    Log request information for analytics
    """
    logger.info(f"Request from IP: {client_ip}, User-Agent: {user_agent}")
    return {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "timestamp": datetime.utcnow()
    }