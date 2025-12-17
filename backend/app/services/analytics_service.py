"""
Analytics service for log data analysis and aggregations
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import DESCENDING, ASCENDING

from app.core.database import get_database
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class AnalyticsService:
    """Service for log analytics and aggregations"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.logs_collection = db.logs
        self.anomalies_collection = db.anomalies
    
    async def get_log_volume(
        self,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h"
    ) -> List[Dict[str, Any]]:
        """
        Get log volume over time
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            interval: Time interval (1m, 5m, 1h, 1d)
        """
        try:
            # Convert interval to seconds
            interval_seconds = self._parse_interval(interval)
            
            pipeline = [
                {
                    "$match": {
                        "timestamp": {
                            "$gte": start_time,
                            "$lte": end_time
                        }
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "$toDate": {
                                "$subtract": [
                                    {"$toLong": "$timestamp"},
                                    {"$mod": [
                                        {"$toLong": "$timestamp"},
                                        interval_seconds * 1000
                                    ]}
                                ]
                            }
                        },
                        "count": {"$sum": 1},
                        "error_count": {
                            "$sum": {
                                "$cond": [
                                    {"$in": ["$level", ["ERROR", "CRITICAL", "FATAL"]]},
                                    1,
                                    0
                                ]
                            }
                        },
                        "warning_count": {
                            "$sum": {
                                "$cond": [{"$eq": ["$level", "WARNING"]}, 1, 0]
                            }
                        }
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            
            results = await self.logs_collection.aggregate(pipeline).to_list(None)
            
            return [
                {
                    "timestamp": r["_id"],
                    "total": r["count"],
                    "errors": r["error_count"],
                    "warnings": r["warning_count"]
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get log volume: {e}")
            return []
    
    async def get_error_rate(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Calculate error rate statistics"""
        try:
            pipeline = [
                {
                    "$match": {
                        "timestamp": {
                            "$gte": start_time,
                            "$lte": end_time
                        }
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
            
            if result:
                total = result[0]["total"]
                errors = result[0]["errors"]
                error_rate = (errors / total * 100) if total > 0 else 0
                
                return {
                    "total_logs": total,
                    "error_count": errors,
                    "error_rate": round(error_rate, 2),
                    "period_start": start_time,
                    "period_end": end_time
                }
            
            return {
                "total_logs": 0,
                "error_count": 0,
                "error_rate": 0.0,
                "period_start": start_time,
                "period_end": end_time
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate error rate: {e}")
            return {}
    
    async def get_top_errors(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get most frequent error messages"""
        try:
            pipeline = [
                {
                    "$match": {
                        "timestamp": {
                            "$gte": start_time,
                            "$lte": end_time
                        },
                        "level": {"$in": ["ERROR", "CRITICAL", "FATAL"]}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "message": "$message",
                            "service": "$service"
                        },
                        "count": {"$sum": 1},
                        "last_seen": {"$max": "$timestamp"},
                        "first_seen": {"$min": "$timestamp"}
                    }
                },
                {"$sort": {"count": -1}},
                {"$limit": limit}
            ]
            
            results = await self.logs_collection.aggregate(pipeline).to_list(None)
            
            return [
                {
                    "message": r["_id"]["message"],
                    "service": r["_id"]["service"],
                    "count": r["count"],
                    "first_seen": r["first_seen"],
                    "last_seen": r["last_seen"]
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get top errors: {e}")
            return []
    
    async def get_service_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get metrics grouped by service"""
        try:
            pipeline = [
                {
                    "$match": {
                        "timestamp": {
                            "$gte": start_time,
                            "$lte": end_time
                        }
                    }
                },
                {
                    "$group": {
                        "_id": "$service",
                        "total_logs": {"$sum": 1},
                        "errors": {
                            "$sum": {
                                "$cond": [
                                    {"$in": ["$level", ["ERROR", "CRITICAL", "FATAL"]]},
                                    1,
                                    0
                                ]
                            }
                        },
                        "warnings": {
                            "$sum": {
                                "$cond": [{"$eq": ["$level", "WARNING"]}, 1, 0]
                            }
                        },
                        "avg_response_time": {"$avg": "$response_time"}
                    }
                },
                {"$sort": {"total_logs": -1}}
            ]
            
            results = await self.logs_collection.aggregate(pipeline).to_list(None)
            
            return [
                {
                    "service": r["_id"],
                    "total_logs": r["total_logs"],
                    "errors": r["errors"],
                    "warnings": r["warnings"],
                    "error_rate": round(r["errors"] / r["total_logs"] * 100, 2) if r["total_logs"] > 0 else 0,
                    "avg_response_time": round(r["avg_response_time"], 2) if r["avg_response_time"] else None
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get service metrics: {e}")
            return []
    
    async def get_response_time_stats(
        self,
        start_time: datetime,
        end_time: datetime,
        service: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get response time statistics"""
        try:
            match_query = {
                "timestamp": {
                    "$gte": start_time,
                    "$lte": end_time
                },
                "response_time": {"$exists": True, "$ne": None}
            }
            
            if service:
                match_query["service"] = service
            
            pipeline = [
                {"$match": match_query},
                {
                    "$group": {
                        "_id": None,
                        "avg": {"$avg": "$response_time"},
                        "min": {"$min": "$response_time"},
                        "max": {"$max": "$response_time"},
                        "count": {"$sum": 1}
                    }
                }
            ]
            
            # Get percentiles separately
            percentile_pipeline = [
                {"$match": match_query},
                {"$sort": {"response_time": 1}},
                {
                    "$group": {
                        "_id": None,
                        "values": {"$push": "$response_time"}
                    }
                }
            ]
            
            result = await self.logs_collection.aggregate(pipeline).to_list(1)
            percentile_result = await self.logs_collection.aggregate(percentile_pipeline).to_list(1)
            
            if result and result[0]:
                stats = {
                    "avg": round(result[0]["avg"], 2),
                    "min": round(result[0]["min"], 2),
                    "max": round(result[0]["max"], 2),
                    "count": result[0]["count"]
                }
                
                # Calculate percentiles
                if percentile_result and percentile_result[0].get("values"):
                    values = sorted(percentile_result[0]["values"])
                    stats["p50"] = self._percentile(values, 50)
                    stats["p95"] = self._percentile(values, 95)
                    stats["p99"] = self._percentile(values, 99)
                
                return stats
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get response time stats: {e}")
            return {}
    
    async def get_anomaly_summary(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get anomaly detection summary"""
        try:
            pipeline = [
                {
                    "$match": {
                        "detected_at": {
                            "$gte": start_time,
                            "$lte": end_time
                        }
                    }
                },
                {
                    "$group": {
                        "_id": "$anomaly_type",
                        "count": {"$sum": 1},
                        "avg_score": {"$avg": "$anomaly_score"}
                    }
                }
            ]
            
            results = await self.anomalies_collection.aggregate(pipeline).to_list(None)
            
            total = sum(r["count"] for r in results)
            
            return {
                "total_anomalies": total,
                "by_type": [
                    {
                        "type": r["_id"],
                        "count": r["count"],
                        "avg_score": round(r["avg_score"], 3)
                    }
                    for r in results
                ],
                "period_start": start_time,
                "period_end": end_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get anomaly summary: {e}")
            return {}
    
    async def get_log_level_distribution(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get distribution of log levels"""
        try:
            pipeline = [
                {
                    "$match": {
                        "timestamp": {
                            "$gte": start_time,
                            "$lte": end_time
                        }
                    }
                },
                {
                    "$group": {
                        "_id": "$level",
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"count": -1}}
            ]
            
            results = await self.logs_collection.aggregate(pipeline).to_list(None)
            total = sum(r["count"] for r in results)
            
            return [
                {
                    "level": r["_id"],
                    "count": r["count"],
                    "percentage": round(r["count"] / total * 100, 2) if total > 0 else 0
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get log level distribution: {e}")
            return []
    
    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to seconds"""
        unit = interval[-1]
        value = int(interval[:-1])
        
        units = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400
        }
        
        return value * units.get(unit, 60)
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile from sorted values"""
        if not values:
            return 0.0
        
        k = (len(values) - 1) * percentile / 100
        f = int(k)
        c = f + 1
        
        if c >= len(values):
            return round(values[-1], 2)
        
        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        
        return round(d0 + d1, 2)


# Dependency injection
async def get_analytics_service() -> AnalyticsService:
    """Get AnalyticsService instance"""
    db = await get_database()
    return AnalyticsService(db)