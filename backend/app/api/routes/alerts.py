"""
Alerts API endpoints
"""
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, Query, HTTPException, Body
from pydantic import BaseModel, Field

from app.services.alert_service import AlertService, get_alert_service
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])


# Request/Response Models
class AlertRuleCreate(BaseModel):
    """Create alert rule request"""
    name: str = Field(..., min_length=1, max_length=200)
    description: str
    condition: Dict[str, Any]
    severity: str = Field(..., pattern="^(INFO|WARNING|ERROR|CRITICAL)$")
    enabled: bool = True
    threshold: Optional[Dict[str, Any]] = None
    notification_channels: Optional[List[str]] = []


class AlertRuleUpdate(BaseModel):
    """Update alert rule request"""
    name: Optional[str] = None
    description: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None
    severity: Optional[str] = None
    enabled: Optional[bool] = None
    threshold: Optional[Dict[str, Any]] = None
    notification_channels: Optional[List[str]] = None


class AlertCreate(BaseModel):
    """Create alert request"""
    rule_id: str
    title: str = Field(..., min_length=1, max_length=200)
    description: str
    severity: str = Field(..., pattern="^(INFO|WARNING|ERROR|CRITICAL)$")
    source: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class AlertAcknowledge(BaseModel):
    """Acknowledge alert request"""
    acknowledged_by: str = Field(..., min_length=1)


class AlertResolve(BaseModel):
    """Resolve alert request"""
    resolved_by: str = Field(..., min_length=1)
    resolution_note: Optional[str] = None


# Alert Rules Endpoints
@router.post("/rules")
async def create_alert_rule(
    rule: AlertRuleCreate,
    service: AlertService = Depends(get_alert_service)
):
    """
    Create a new alert rule
    
    Alert rules define conditions that trigger alerts when met.
    """
    try:
        rule_id = await service.create_alert_rule(
            name=rule.name,
            description=rule.description,
            condition=rule.condition,
            severity=rule.severity,
            enabled=rule.enabled,
            threshold=rule.threshold,
            notification_channels=rule.notification_channels
        )
        
        if not rule_id:
            raise HTTPException(status_code=500, detail="Failed to create alert rule")
        
        return {
            "success": True,
            "rule_id": rule_id,
            "message": "Alert rule created successfully"
        }
    except Exception as e:
        logger.error(f"Failed to create alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules")
async def get_alert_rules(
    enabled_only: bool = Query(False),
    service: AlertService = Depends(get_alert_service)
):
    """
    Get all alert rules
    
    Parameters:
    - **enabled_only**: Return only enabled rules
    """
    try:
        rules = await service.get_alert_rules(enabled_only=enabled_only)
        
        return {
            "success": True,
            "data": rules,
            "count": len(rules)
        }
    except Exception as e:
        logger.error(f"Failed to get alert rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/rules/{rule_id}")
async def update_alert_rule(
    rule_id: str,
    updates: AlertRuleUpdate,
    service: AlertService = Depends(get_alert_service)
):
    """
    Update an alert rule
    
    Only provided fields will be updated.
    """
    try:
        # Filter out None values
        update_data = {k: v for k, v in updates.dict().items() if v is not None}
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No updates provided")
        
        success = await service.update_alert_rule(rule_id, update_data)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert rule not found")
        
        return {
            "success": True,
            "message": "Alert rule updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rules/{rule_id}")
async def delete_alert_rule(
    rule_id: str,
    service: AlertService = Depends(get_alert_service)
):
    """Delete an alert rule"""
    try:
        success = await service.delete_alert_rule(rule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert rule not found")
        
        return {
            "success": True,
            "message": "Alert rule deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Alerts Endpoints
@router.post("")
async def create_alert(
    alert: AlertCreate,
    service: AlertService = Depends(get_alert_service)
):
    """
    Create a new alert
    
    Usually called automatically when alert conditions are met.
    """
    try:
        alert_id = await service.create_alert(
            rule_id=alert.rule_id,
            title=alert.title,
            description=alert.description,
            severity=alert.severity,
            source=alert.source,
            metadata=alert.metadata
        )
        
        if not alert_id:
            raise HTTPException(status_code=500, detail="Failed to create alert")
        
        return {
            "success": True,
            "alert_id": alert_id,
            "message": "Alert created successfully"
        }
    except Exception as e:
        logger.error(f"Failed to create alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def get_alerts(
    status: Optional[str] = Query(None, pattern="^(OPEN|ACKNOWLEDGED|RESOLVED)$"),
    severity: Optional[str] = Query(None, pattern="^(INFO|WARNING|ERROR|CRITICAL)$"),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    service: AlertService = Depends(get_alert_service)
):
    """
    Get alerts with optional filters
    
    Parameters:
    - **status**: Filter by status (OPEN, ACKNOWLEDGED, RESOLVED)
    - **severity**: Filter by severity (INFO, WARNING, ERROR, CRITICAL)
    - **start_time**: Filter by creation time (start)
    - **end_time**: Filter by creation time (end)
    - **limit**: Maximum number of alerts to return (1-1000)
    """
    try:
        alerts = await service.get_alerts(
            status=status,
            severity=severity,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return {
            "success": True,
            "data": alerts,
            "count": len(alerts)
        }
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{alert_id}")
async def get_alert(
    alert_id: str,
    service: AlertService = Depends(get_alert_service)
):
    """Get a specific alert by ID"""
    try:
        alert = await service.get_alert_by_id(alert_id)
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "success": True,
            "data": alert
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    data: AlertAcknowledge,
    service: AlertService = Depends(get_alert_service)
):
    """
    Acknowledge an alert
    
    Marks the alert as acknowledged by a user.
    """
    try:
        success = await service.acknowledge_alert(
            alert_id=alert_id,
            acknowledged_by=data.acknowledged_by
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "success": True,
            "message": "Alert acknowledged successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    data: AlertResolve,
    service: AlertService = Depends(get_alert_service)
):
    """
    Resolve an alert
    
    Marks the alert as resolved with optional resolution notes.
    """
    try:
        success = await service.resolve_alert(
            alert_id=alert_id,
            resolved_by=data.resolved_by,
            resolution_note=data.resolution_note
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "success": True,
            "message": "Alert resolved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/summary")
async def get_alert_statistics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    service: AlertService = Depends(get_alert_service)
):
    """
    Get alert statistics
    
    Returns counts by severity and status for the specified time period.
    """
    try:
        from datetime import timedelta
        
        end = end_time or datetime.utcnow()
        start = start_time or (end - timedelta(hours=24))
        
        stats = await service.get_alert_statistics(start, end)
        
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Failed to get alert statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check/error-rate")
async def check_error_rate(
    time_window_minutes: int = Query(5, ge=1, le=60),
    threshold_percentage: float = Query(10.0, ge=0, le=100),
    service: AlertService = Depends(get_alert_service)
):
    """
    Check if error rate exceeds threshold
    
    Returns alert data if threshold is exceeded.
    
    Parameters:
    - **time_window_minutes**: Time window to check (1-60 minutes)
    - **threshold_percentage**: Error rate threshold (0-100%)
    """
    try:
        result = await service.check_error_rate_alert(
            time_window_minutes=time_window_minutes,
            threshold_percentage=threshold_percentage
        )
        
        if result:
            return {
                "success": True,
                "alert_triggered": True,
                "data": result
            }
        else:
            return {
                "success": True,
                "alert_triggered": False,
                "message": "Error rate within acceptable limits"
            }
    except Exception as e:
        logger.error(f"Failed to check error rate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check/response-time")
async def check_response_time(
    time_window_minutes: int = Query(5, ge=1, le=60),
    threshold_ms: float = Query(1000.0, ge=0),
    percentile: int = Query(95, ge=50, le=99),
    service: AlertService = Depends(get_alert_service)
):
    """
    Check if response time exceeds threshold
    
    Returns alert data if threshold is exceeded.
    
    Parameters:
    - **time_window_minutes**: Time window to check (1-60 minutes)
    - **threshold_ms**: Response time threshold in milliseconds
    - **percentile**: Percentile to check (50-99)
    """
    try:
        result = await service.check_response_time_alert(
            time_window_minutes=time_window_minutes,
            threshold_ms=threshold_ms,
            percentile=percentile
        )
        
        if result:
            return {
                "success": True,
                "alert_triggered": True,
                "data": result
            }
        else:
            return {
                "success": True,
                "alert_triggered": False,
                "message": "Response time within acceptable limits"
            }
    except Exception as e:
        logger.error(f"Failed to check response time: {e}")
        raise HTTPException(status_code=500, detail=str(e))