"""
Health check endpoints
"""
from fastapi import APIRouter, Depends
from datetime import datetime
from app.core.database import get_database
from app.core.redis_client import get_redis
from motor.motor_asyncio import AsyncIOMotorDatabase
import redis.asyncio as redis

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AI Log Analytics API"
    }


@router.get("/health/detailed")
async def detailed_health_check(
    db: AsyncIOMotorDatabase = Depends(get_database),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Detailed health check with dependency status"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check MongoDB
    try:
        await db.command("ping")
        health_status["services"]["mongodb"] = "connected"
    except Exception as e:
        health_status["services"]["mongodb"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        await redis_client.ping()
        health_status["services"]["redis"] = "connected"
    except Exception as e:
        health_status["services"]["redis"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


@router.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"message": "pong"}