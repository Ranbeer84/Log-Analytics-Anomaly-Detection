"""
Analytics API endpoints
"""
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field

from app.services.analytics_service import AnalyticsService, get_analytics_service
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


class TimeRangeQuery(BaseModel):
    """Time range query parameters"""
    start_time: Optional[datetime] = Field(
        None,
        description="Start time (defaults to 24 hours ago)"
    )
    end_time: Optional[datetime] = Field(
        None,
        description="End time (defaults to now)"
    )
    
    def get_times(self):
        """Get start and end times with defaults"""
        end = self.end_time or datetime.utcnow()
        start = self.start_time or (end - timedelta(hours=24))
        return start, end


@router.get("/volume")
async def get_log_volume(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    interval: str = Query("1h", regex="^[0-9]+[smhd]$"),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get log volume over time
    
    Parameters:
    - **start_time**: Start of time range (ISO format)
    - **end_time**: End of time range (ISO format)
    - **interval**: Time interval (1m, 5m, 1h, 1d)
    """
    try:
        time_range = TimeRangeQuery(start_time=start_time, end_time=end_time)
        start, end = time_range.get_times()
        
        data = await service.get_log_volume(start, end, interval)
        
        return {
            "success": True,
            "data": data,
            "period": {
                "start": start,
                "end": end,
                "interval": interval
            }
        }
    except Exception as e:
        logger.error(f"Failed to get log volume: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/error-rate")
async def get_error_rate(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get error rate statistics
    
    Returns total logs, error count, and error rate percentage
    """
    try:
        time_range = TimeRangeQuery(start_time=start_time, end_time=end_time)
        start, end = time_range.get_times()
        
        data = await service.get_error_rate(start, end)
        
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        logger.error(f"Failed to get error rate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top-errors")
async def get_top_errors(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get most frequent error messages
    
    Parameters:
    - **limit**: Maximum number of errors to return (1-100)
    """
    try:
        time_range = TimeRangeQuery(start_time=start_time, end_time=end_time)
        start, end = time_range.get_times()
        
        data = await service.get_top_errors(start, end, limit)
        
        return {
            "success": True,
            "data": data,
            "count": len(data)
        }
    except Exception as e:
        logger.error(f"Failed to get top errors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services")
async def get_service_metrics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get metrics grouped by service
    
    Returns log counts, error rates, and response times per service
    """
    try:
        time_range = TimeRangeQuery(start_time=start_time, end_time=end_time)
        start, end = time_range.get_times()
        
        data = await service.get_service_metrics(start, end)
        
        return {
            "success": True,
            "data": data,
            "count": len(data)
        }
    except Exception as e:
        logger.error(f"Failed to get service metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/response-time")
async def get_response_time_stats(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    service_name: Optional[str] = Query(None),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get response time statistics
    
    Returns average, min, max, and percentiles (p50, p95, p99)
    
    Parameters:
    - **service_name**: Filter by specific service (optional)
    """
    try:
        time_range = TimeRangeQuery(start_time=start_time, end_time=end_time)
        start, end = time_range.get_times()
        
        data = await service.get_response_time_stats(start, end, service_name)
        
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        logger.error(f"Failed to get response time stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies")
async def get_anomaly_summary(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get anomaly detection summary
    
    Returns total anomalies and breakdown by type
    """
    try:
        time_range = TimeRangeQuery(start_time=start_time, end_time=end_time)
        start, end = time_range.get_times()
        
        data = await service.get_anomaly_summary(start, end)
        
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        logger.error(f"Failed to get anomaly summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/log-levels")
async def get_log_level_distribution(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get distribution of log levels
    
    Returns count and percentage for each log level (INFO, WARNING, ERROR, etc.)
    """
    try:
        time_range = TimeRangeQuery(start_time=start_time, end_time=end_time)
        start, end = time_range.get_times()
        
        data = await service.get_log_level_distribution(start, end)
        
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        logger.error(f"Failed to get log level distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_dashboard_data(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get comprehensive dashboard data
    
    Returns all key metrics in a single response
    """
    try:
        time_range = TimeRangeQuery(start_time=start_time, end_time=end_time)
        start, end = time_range.get_times()
        
        # Fetch all metrics concurrently
        error_rate = await service.get_error_rate(start, end)
        service_metrics = await service.get_service_metrics(start, end)
        log_levels = await service.get_log_level_distribution(start, end)
        anomaly_summary = await service.get_anomaly_summary(start, end)
        top_errors = await service.get_top_errors(start, end, limit=5)
        
        return {
            "success": True,
            "data": {
                "error_rate": error_rate,
                "services": service_metrics,
                "log_levels": log_levels,
                "anomalies": anomaly_summary,
                "top_errors": top_errors
            },
            "period": {
                "start": start,
                "end": end
            }
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))