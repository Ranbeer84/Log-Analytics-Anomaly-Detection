"""
Alert service for managing and triggering alerts
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum

from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

from app.core.database import get_database
from app.core.logging_config import get_logger
from app.models.alert import AlertStatus, AlertSeverity

logger = get_logger(__name__)


class AlertService:
    """Service for alert management and triggering"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.alerts_collection = db.alerts
        self.alert_rules_collection = db.alert_rules
        self.logs_collection = db.logs
    
    async def create_alert_rule(
        self,
        name: str,
        description: str,
        condition: Dict[str, Any],
        severity: str,
        enabled: bool = True,
        threshold: Optional[Dict[str, Any]] = None,
        notification_channels: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Create a new alert rule
        
        Args:
            name: Rule name
            description: Rule description
            condition: Alert condition (e.g., error rate > threshold)
            severity: Alert severity
            enabled: Whether rule is active
            threshold: Threshold configuration
            notification_channels: List of notification channels
            
        Returns:
            Created rule ID
        """
        try:
            rule_doc = {
                "name": name,
                "description": description,
                "condition": condition,
                "severity": severity,
                "enabled": enabled,
                "threshold": threshold or {},
                "notification_channels": notification_channels or [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "last_triggered": None,
                "trigger_count": 0
            }
            
            result = await self.alert_rules_collection.insert_one(rule_doc)
            logger.info(f"Created alert rule: {name}")
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            return None
    
    async def get_alert_rules(
        self,
        enabled_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get all alert rules"""
        try:
            query = {"enabled": True} if enabled_only else {}
            
            cursor = self.alert_rules_collection.find(query)
            rules = await cursor.to_list(None)
            
            # Convert ObjectId to string
            for rule in rules:
                rule["id"] = str(rule.pop("_id"))
            
            return rules
            
        except Exception as e:
            logger.error(f"Failed to get alert rules: {e}")
            return []
    
    async def update_alert_rule(
        self,
        rule_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update an alert rule"""
        try:
            updates["updated_at"] = datetime.utcnow()
            
            result = await self.alert_rules_collection.update_one(
                {"_id": ObjectId(rule_id)},
                {"$set": updates}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update alert rule: {e}")
            return False
    
    async def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete an alert rule"""
        try:
            result = await self.alert_rules_collection.delete_one(
                {"_id": ObjectId(rule_id)}
            )
            
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete alert rule: {e}")
            return False
    
    async def create_alert(
        self,
        rule_id: str,
        title: str,
        description: str,
        severity: str,
        source: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a new alert
        
        Args:
            rule_id: ID of the triggered rule
            title: Alert title
            description: Alert description
            severity: Alert severity
            source: Source information (log entry, metric, etc.)
            metadata: Additional metadata
            
        Returns:
            Created alert ID
        """
        try:
            alert_doc = {
                "rule_id": rule_id,
                "title": title,
                "description": description,
                "severity": severity,
                "status": AlertStatus.OPEN.value,
                "source": source,
                "metadata": metadata or {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "acknowledged_at": None,
                "acknowledged_by": None,
                "resolved_at": None,
                "resolved_by": None,
                "notification_sent": False
            }
            
            result = await self.alerts_collection.insert_one(alert_doc)
            alert_id = str(result.inserted_id)
            
            # Update rule trigger count and last triggered time
            await self.alert_rules_collection.update_one(
                {"_id": ObjectId(rule_id)},
                {
                    "$set": {"last_triggered": datetime.utcnow()},
                    "$inc": {"trigger_count": 1}
                }
            )
            
            logger.info(f"Created alert: {title} (ID: {alert_id})")
            
            return alert_id
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return None
    
    async def get_alerts(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get alerts with optional filters"""
        try:
            query = {}
            
            if status:
                query["status"] = status
            
            if severity:
                query["severity"] = severity
            
            if start_time or end_time:
                query["created_at"] = {}
                if start_time:
                    query["created_at"]["$gte"] = start_time
                if end_time:
                    query["created_at"]["$lte"] = end_time
            
            cursor = self.alerts_collection.find(query).sort(
                "created_at", -1
            ).limit(limit)
            
            alerts = await cursor.to_list(None)
            
            # Convert ObjectId to string
            for alert in alerts:
                alert["id"] = str(alert.pop("_id"))
                if alert.get("rule_id"):
                    alert["rule_id"] = str(alert["rule_id"])
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    async def get_alert_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific alert by ID"""
        try:
            alert = await self.alerts_collection.find_one(
                {"_id": ObjectId(alert_id)}
            )
            
            if alert:
                alert["id"] = str(alert.pop("_id"))
                if alert.get("rule_id"):
                    alert["rule_id"] = str(alert["rule_id"])
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to get alert: {e}")
            return None
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> bool:
        """Acknowledge an alert"""
        try:
            result = await self.alerts_collection.update_one(
                {"_id": ObjectId(alert_id)},
                {
                    "$set": {
                        "status": AlertStatus.ACKNOWLEDGED.value,
                        "acknowledged_at": datetime.utcnow(),
                        "acknowledged_by": acknowledged_by,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_note: Optional[str] = None
    ) -> bool:
        """Resolve an alert"""
        try:
            update_data = {
                "status": AlertStatus.RESOLVED.value,
                "resolved_at": datetime.utcnow(),
                "resolved_by": resolved_by,
                "updated_at": datetime.utcnow()
            }
            
            if resolution_note:
                update_data["resolution_note"] = resolution_note
            
            result = await self.alerts_collection.update_one(
                {"_id": ObjectId(alert_id)},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False
    
    async def check_error_rate_alert(
        self,
        time_window_minutes: int = 5,
        threshold_percentage: float = 10.0
    ) -> Optional[Dict[str, Any]]:
        """
        Check if error rate exceeds threshold
        
        Returns alert data if threshold exceeded, None otherwise
        """
        try:
            start_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            
            pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": start_time}
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "total": {"$sum": 1},
                        "errors": {
                            "$sum": {
                                "$cond": [
                                    {"$in": ["$level", ["ERROR", "CRITICAL", "FATAL"]]},
                                    1,
                                    0
                                ]
                            }
                        }
                    }
                }
            ]
            
            result = await self.logs_collection.aggregate(pipeline).to_list(1)
            
            if result and result[0]["total"] > 0:
                total = result[0]["total"]
                errors = result[0]["errors"]
                error_rate = (errors / total) * 100
                
                if error_rate > threshold_percentage:
                    return {
                        "metric": "error_rate",
                        "value": round(error_rate, 2),
                        "threshold": threshold_percentage,
                        "window_minutes": time_window_minutes,
                        "total_logs": total,
                        "error_count": errors
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check error rate: {e}")
            return None
    
    async def check_response_time_alert(
        self,
        time_window_minutes: int = 5,
        threshold_ms: float = 1000.0,
        percentile: int = 95
    ) -> Optional[Dict[str, Any]]:
        """Check if response time exceeds threshold"""
        try:
            start_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            
            pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": start_time},
                        "response_time": {"$exists": True}
                    }
                },
                {
                    "$sort": {"response_time": 1}
                },
                {
                    "$group": {
                        "_id": None,
                        "values": {"$push": "$response_time"},
                        "count": {"$sum": 1}
                    }
                }
            ]
            
            result = await self.logs_collection.aggregate(pipeline).to_list(1)
            
            if result and result[0]["count"] > 0:
                values = result[0]["values"]
                p_value = self._calculate_percentile(values, percentile)
                
                if p_value > threshold_ms:
                    return {
                        "metric": "response_time",
                        "value": round(p_value, 2),
                        "threshold": threshold_ms,
                        "percentile": percentile,
                        "window_minutes": time_window_minutes,
                        "sample_count": result[0]["count"]
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check response time: {e}")
            return None
    
    async def get_alert_statistics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get alert statistics for a time period"""
        try:
            pipeline = [
                {
                    "$match": {
                        "created_at": {
                            "$gte": start_time,
                            "$lte": end_time
                        }
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "severity": "$severity",
                            "status": "$status"
                        },
                        "count": {"$sum": 1}
                    }
                }
            ]
            
            results = await self.alerts_collection.aggregate(pipeline).to_list(None)
            
            stats = {
                "total": 0,
                "by_severity": {},
                "by_status": {},
                "period_start": start_time,
                "period_end": end_time
            }
            
            for r in results:
                count = r["count"]
                severity = r["_id"]["severity"]
                status = r["_id"]["status"]
                
                stats["total"] += count
                stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + count
                stats["by_status"][status] = stats["by_status"].get(status, 0) + count
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get alert statistics: {e}")
            return {}
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile from list of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = f + 1
        
        if c >= len(sorted_values):
            return sorted_values[-1]
        
        d0 = sorted_values[f] * (c - k)
        d1 = sorted_values[c] * (k - f)
        
        return d0 + d1


# Dependency injection
async def get_alert_service() -> AlertService:
    """Get AlertService instance"""
    db = await get_database()
    return AlertService(db)