"""
Main FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.core.database import connect_to_mongo, close_mongo_connection
from app.core.redis_client import connect_to_redis, close_redis_connection
from app.core.logging_config import setup_logging
from app.api.routes import health, logs

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    """
    # Startup
    logger.info("🚀 Starting AI Log Analytics API...")
    logger.info(f"Environment: {settings.ENV}")
    
    # Connect to databases
    await connect_to_mongo()
    await connect_to_redis()
    
    logger.info("✓ All services initialized")
    logger.info(f"API Documentation: http://localhost:8000/docs")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AI Log Analytics API...")
    await close_mongo_connection()
    await close_redis_connection()
    logger.info("✓ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-Powered Log Analytics & Anomaly Detection System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(logs.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Log Analytics API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )