"""
Log ingestion and retrieval endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.schemas.log_schema import (
    LogCreate,
    LogBatchCreate,
    LogResponse,
    LogIngestResponse,
    LogBatchIngestResponse,
    LogQuery
)
from app.services.log_service import LogService
from app.core.database import get_database

router = APIRouter(prefix="/logs", tags=["Logs"])


def get_log_service(db: AsyncIOMotorDatabase = Depends(get_database)) -> LogService:
    """Dependency to get log service"""
    return LogService(db)


@router.post("", response_model=LogIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_log(
    log: LogCreate,
    service: LogService = Depends(get_log_service)
):
    """
    Ingest a single log entry
    
    - **timestamp**: ISO 8601 timestamp
    - **level**: Log level (DEBUG, INFO, WARN, ERROR, CRITICAL)
    - **service**: Service name
    - **message**: Log message
    - **metadata**: Optional additional data
    """
    try:
        result = await service.create_log(log)
        
        return LogIngestResponse(
            status="success",
            message="Log ingested successfully",
            log_id=result["log_id"],
            stream_id=result["stream_id"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest log: {str(e)}"
        )


@router.post("/batch", response_model=LogBatchIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_logs_batch(
    batch: LogBatchCreate,
    service: LogService = Depends(get_log_service)
):
    """
    Ingest multiple log entries in batch
    
    Maximum 1000 logs per request
    """
    try:
        result = await service.create_logs_batch(batch.logs)
        
        response_status = "success" if result["failed_count"] == 0 else "partial_success"
        
        return LogBatchIngestResponse(
            status=response_status,
            message=f"Batch ingestion completed: {result['ingested_count']} succeeded, {result['failed_count']} failed",
            ingested_count=result["ingested_count"],
            failed_count=result["failed_count"],
            errors=result.get("errors")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest batch: {str(e)}"
        )


@router.get("", response_model=List[LogResponse])
async def get_logs(
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    level: Optional[str] = Query(None, description="Log level filter"),
    service: Optional[str] = Query(None, description="Service name filter"),
    search: Optional[str] = Query(None, description="Search in message"),
    limit: int = Query(50, ge=1, le=1000, description="Number of logs to return"),
    skip: int = Query(0, ge=0, description="Number of logs to skip"),
    log_service: LogService = Depends(get_log_service)
):
    """
    Query logs with filters
    
    - Supports date range filtering
    - Filter by log level
    - Filter by service name
    - Full-text search in messages
    - Pagination with limit and skip
    """
    try:
        query = LogQuery(
            start_date=start_date,
            end_date=end_date,
            level=level,
            service=service,
            search=search,
            limit=limit,
            skip=skip
        )
        
        logs = await log_service.get_logs(query)
        return logs
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query logs: {str(e)}"
        )


@router.get("/{log_id}", response_model=LogResponse)
async def get_log_by_id(
    log_id: str,
    service: LogService = Depends(get_log_service)
):
    """
    Get a specific log by ID
    """
    try:
        log = await service.get_log_by_id(log_id)
        
        if not log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Log with ID {log_id} not found"
            )
        
        return log
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get log: {str(e)}"
        )


@router.get("/stats/summary")
async def get_log_statistics(
    start_date: Optional[datetime] = Query(None, description="Start date for stats"),
    end_date: Optional[datetime] = Query(None, description="End date for stats"),
    service: LogService = Depends(get_log_service)
):
    """
    Get log statistics
    
    Returns aggregated statistics including:
    - Total log count
    - Error count
    - Warning count
    - Info count
    - Unique services
    - Time range
    """
    try:
        stats = await service.get_log_stats(start_date, end_date)
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )