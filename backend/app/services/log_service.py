"""
Log service - Business logic for log operations
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.schemas.log_schema import LogCreate, LogQuery
from app.models.log_entry import LogEntry
from app.core.redis_client import publish_log_to_stream
# import encoder(fix error)
from fastapi.encoders import jsonable_encoder
import logging

logger = logging.getLogger(__name__)


class LogService:
    """Service for log operations"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.logs
    
    async def create_log(self, log_data: LogCreate) -> Dict[str, Any]:
        """
        Create a new log entry
        
        Args:
            log_data: Log creation schema
            
        Returns:
            Created log with ID
        """
        # try:
        #     # Prepare log document

        #     # While ingesting logs cause error - Object of type datetime is not JSON serializable
        #     # MongoDB is fine — Redis is failing.
        #     # convert the Pydantic model into a JSON-safe dict before inserting into MongoDB and publishing to Redis. 

        #     # log_dict = log_data.model_dump()
        #     # log_dict['created_at'] = datetime.utcnow()

        #     # Correct 
        #     log_dict = jsonable_encoder(log_data)
        #     log_dict['created_at'] = datetime.utcnow().isoformat()

        #     log_dict['processed'] = False
        #     log_dict['anomaly_checked'] = False
            
        #     # Insert into MongoDB
        #     result = await self.collection.insert_one(log_dict)
        #     log_id = str(result.inserted_id)
            
        #     # Publish to Redis Stream for async processing
        #     log_dict['_id'] = log_id
        #     stream_id = await publish_log_to_stream(log_dict)
            
        #     logger.info(f"Log created: {log_id}, Stream ID: {stream_id}")
            
        #     return {
        #         "log_id": log_id,
        #         "stream_id": stream_id,
        #         "timestamp": log_dict['timestamp']
        #     }
            
        # except Exception as e:
        #     logger.error(f"Error creating log: {e}")
        #     raise
        try:
            # 1️⃣ MongoDB document (KEEP datetime objects)
            mongo_doc = log_data.model_dump()     # timestamp is datetime
            mongo_doc["created_at"] = datetime.utcnow()
            mongo_doc["processed"] = False
            mongo_doc["anomaly_checked"] = False

            # Insert into MongoDB (schema expects BSON dates)
            result = await self.collection.insert_one(mongo_doc)
            log_id = str(result.inserted_id)

            # 2️⃣ Redis payload (JSON-safe)
            redis_doc = jsonable_encoder(mongo_doc)
            redis_doc["_id"] = log_id

            stream_id = await publish_log_to_stream(redis_doc)

            logger.info(f"Log created: {log_id}, Stream ID: {stream_id}")

            return {
                "log_id": log_id,
                "stream_id": stream_id
            }

        except Exception as e:
            logger.error(f"Error creating log: {e}")
            raise
        # 
    
    async def create_logs_batch(self, logs: List[LogCreate]) -> Dict[str, Any]:
        """
        Create multiple log entries in batch
        
        Args:
            logs: List of log creation schemas
            
        Returns:
            Batch creation result
        """
        try:
            ingested = 0
            failed = 0
            errors = []
            
            for idx, log_data in enumerate(logs):
                try:
                    await self.create_log(log_data)
                    ingested += 1
                except Exception as e:
                    failed += 1
                    errors.append(f"Error at index {idx}: {str(e)}")
            
            return {
                "ingested_count": ingested,
                "failed_count": failed,
                "errors": errors if errors else None
            }
            
        except Exception as e:
            logger.error(f"Error in batch creation: {e}")
            raise
    
    async def get_logs(self, query: LogQuery) -> List[Dict[str, Any]]:
        """
        Query logs with filters
        
        Args:
            query: Log query parameters
            
        Returns:
            List of matching logs
        """
        try:
            # Build MongoDB query
            filter_query = {}
            
            # Date range filter
            if query.start_date or query.end_date:
                filter_query['timestamp'] = {}
                if query.start_date:
                    filter_query['timestamp']['$gte'] = query.start_date
                if query.end_date:
                    filter_query['timestamp']['$lte'] = query.end_date
            
            # Level filter
            if query.level:
                filter_query['level'] = query.level.upper()
            
            # Service filter
            if query.service:
                filter_query['service'] = query.service
            
            # Text search
            if query.search:
                filter_query['message'] = {'$regex': query.search, '$options': 'i'}
            
            # Execute query
            cursor = self.collection.find(filter_query).sort('timestamp', -1).skip(query.skip).limit(query.limit)
            
            logs = []
            async for log in cursor:
                log['_id'] = str(log['_id'])
                logs.append(log)
            
            return logs
            
        except Exception as e:
            logger.error(f"Error querying logs: {e}")
            raise
    
    async def get_log_by_id(self, log_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single log by ID
        
        Args:
            log_id: Log ID
            
        Returns:
            Log document or None
        """
        try:
            from bson import ObjectId
            
            log = await self.collection.find_one({"_id": ObjectId(log_id)})
            if log:
                log['_id'] = str(log['_id'])
            return log
            
        except Exception as e:
            logger.error(f"Error getting log: {e}")
            return None
    
    async def get_log_stats(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get log statistics
        
        Args:
            start_date: Start date for stats
            end_date: End date for stats
            
        Returns:
            Statistics dictionary
        """
        try:
            # Build match query
            match_query = {}
            if start_date or end_date:
                match_query['timestamp'] = {}
                if start_date:
                    match_query['timestamp']['$gte'] = start_date
                if end_date:
                    match_query['timestamp']['$lte'] = end_date
            
            # Aggregation pipeline
            pipeline = [
                {"$match": match_query} if match_query else {"$match": {}},
                {
                    "$facet": {
                        "total": [{"$count": "count"}],
                        "by_level": [
                            {"$group": {"_id": "$level", "count": {"$sum": 1}}}
                        ],
                        "unique_services": [
                            {"$group": {"_id": "$service"}},
                            {"$count": "count"}
                        ],
                        "time_range": [
                            {
                                "$group": {
                                    "_id": None,
                                    "min_time": {"$min": "$timestamp"},
                                    "max_time": {"$max": "$timestamp"}
                                }
                            }
                        ]
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(1)
            
            if not result:
                return {
                    "total_logs": 0,
                    "error_count": 0,
                    "warning_count": 0,
                    "info_count": 0,
                    "unique_services": 0
                }
            
            data = result[0]
            
            # Extract counts by level
            level_counts = {item['_id']: item['count'] for item in data.get('by_level', [])}
            
            return {
                "total_logs": data['total'][0]['count'] if data.get('total') else 0,
                "error_count": level_counts.get('ERROR', 0),
                "warning_count": level_counts.get('WARN', 0) + level_counts.get('WARNING', 0),
                "info_count": level_counts.get('INFO', 0),
                "debug_count": level_counts.get('DEBUG', 0),
                "unique_services": data['unique_services'][0]['count'] if data.get('unique_services') else 0,
                "time_range": data['time_range'][0] if data.get('time_range') else None
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise
    
    async def delete_old_logs(self, days: int) -> int:
        """
        Delete logs older than specified days
        
        Args:
            days: Number of days to retain
            
        Returns:
            Number of deleted logs
        """
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            result = await self.collection.delete_many({"timestamp": {"$lt": cutoff_date}})
            
            logger.info(f"Deleted {result.deleted_count} old logs")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting old logs: {e}")
            raise