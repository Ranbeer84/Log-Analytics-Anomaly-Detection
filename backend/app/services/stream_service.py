"""
Redis Streams service for log message handling
"""
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.redis_client import get_redis
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class StreamService:
    """Service for handling Redis Streams operations"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.stream_name = "logs:stream"
        self.consumer_group = "log-processors"
        self.max_retries = 3
        
    async def initialize_stream(self):
        """Initialize the stream and consumer group"""
        try:
            # Create consumer group if it doesn't exist
            try:
                await self.redis.xgroup_create(
                    name=self.stream_name,
                    groupname=self.consumer_group,
                    id='0',
                    mkstream=True
                )
                logger.info(f"Created consumer group: {self.consumer_group}")
            except RedisError as e:
                if "BUSYGROUP" in str(e):
                    logger.info(f"Consumer group already exists: {self.consumer_group}")
                else:
                    raise
        except Exception as e:
            logger.error(f"Failed to initialize stream: {e}")
            raise
    
    async def publish_log(self, log_data: Dict[str, Any]) -> Optional[str]:
        """
        Publish a log entry to the stream
        
        Args:
            log_data: Log entry data
            
        Returns:
            Message ID if successful, None otherwise
        """
        try:
            # Convert datetime objects to ISO format strings
            serialized_data = self._serialize_log_data(log_data)
            
            # Add to stream
            message_id = await self.redis.xadd(
                name=self.stream_name,
                fields=serialized_data,
                maxlen=100000  # Keep last 100k messages
            )
            
            logger.debug(f"Published log to stream: {message_id}")
            return message_id.decode() if isinstance(message_id, bytes) else message_id
            
        except Exception as e:
            logger.error(f"Failed to publish log to stream: {e}")
            return None
    
    async def publish_batch(self, logs: List[Dict[str, Any]]) -> int:
        """
        Publish multiple logs to the stream
        
        Args:
            logs: List of log entries
            
        Returns:
            Number of successfully published logs
        """
        published_count = 0
        
        # Use pipeline for better performance
        async with self.redis.pipeline() as pipe:
            try:
                for log_data in logs:
                    serialized_data = self._serialize_log_data(log_data)
                    pipe.xadd(
                        name=self.stream_name,
                        fields=serialized_data,
                        maxlen=100000
                    )
                
                results = await pipe.execute()
                published_count = sum(1 for r in results if r)
                
                logger.info(f"Published {published_count}/{len(logs)} logs to stream")
                
            except Exception as e:
                logger.error(f"Failed to publish batch to stream: {e}")
        
        return published_count
    
    async def consume_messages(
        self,
        consumer_name: str,
        count: int = 10,
        block: int = 1000
    ) -> List[tuple]:
        """
        Consume messages from the stream
        
        Args:
            consumer_name: Name of the consumer
            count: Number of messages to read
            block: Block time in milliseconds
            
        Returns:
            List of (message_id, message_data) tuples
        """
        try:
            messages = await self.redis.xreadgroup(
                groupname=self.consumer_group,
                consumername=consumer_name,
                streams={self.stream_name: '>'},
                count=count,
                block=block
            )
            
            if not messages:
                return []
            
            # Parse messages
            parsed_messages = []
            for stream_name, stream_messages in messages:
                for msg_id, msg_data in stream_messages:
                    parsed_data = self._deserialize_log_data(msg_data)
                    parsed_messages.append((msg_id, parsed_data))
            
            return parsed_messages
            
        except Exception as e:
            logger.error(f"Failed to consume messages: {e}")
            return []
    
    async def acknowledge_message(self, message_id: str) -> bool:
        """
        Acknowledge a processed message
        
        Args:
            message_id: ID of the message to acknowledge
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = await self.redis.xack(
                self.stream_name,
                self.consumer_group,
                message_id
            )
            return result > 0
        except Exception as e:
            logger.error(f"Failed to acknowledge message {message_id}: {e}")
            return False
    
    async def get_pending_messages(
        self,
        consumer_name: str,
        count: int = 10
    ) -> List[tuple]:
        """
        Get pending messages for a consumer
        
        Args:
            consumer_name: Name of the consumer
            count: Number of messages to retrieve
            
        Returns:
            List of pending messages
        """
        try:
            # Get pending message IDs
            pending = await self.redis.xpending_range(
                name=self.stream_name,
                groupname=self.consumer_group,
                min='-',
                max='+',
                count=count,
                consumername=consumer_name
            )
            
            if not pending:
                return []
            
            # Claim and read pending messages
            message_ids = [p['message_id'] for p in pending]
            claimed = await self.redis.xclaim(
                name=self.stream_name,
                groupname=self.consumer_group,
                consumername=consumer_name,
                min_idle_time=60000,  # 1 minute
                message_ids=message_ids
            )
            
            return [(msg_id, self._deserialize_log_data(msg_data)) 
                    for msg_id, msg_data in claimed]
            
        except Exception as e:
            logger.error(f"Failed to get pending messages: {e}")
            return []
    
    async def get_stream_info(self) -> Dict[str, Any]:
        """
        Get stream information and statistics
        
        Returns:
            Stream info dictionary
        """
        try:
            info = await self.redis.xinfo_stream(self.stream_name)
            groups = await self.redis.xinfo_groups(self.stream_name)
            
            return {
                'length': info.get('length', 0),
                'first_entry': info.get('first-entry'),
                'last_entry': info.get('last-entry'),
                'groups': [
                    {
                        'name': g.get('name'),
                        'consumers': g.get('consumers'),
                        'pending': g.get('pending')
                    }
                    for g in groups
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get stream info: {e}")
            return {}
    
    def _serialize_log_data(self, log_data: Dict[str, Any]) -> Dict[str, str]:
        """Convert log data to Redis-compatible format"""
        serialized = {}
        
        for key, value in log_data.items():
            if isinstance(value, (dict, list)):
                serialized[key] = json.dumps(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif value is None:
                serialized[key] = ''
            else:
                serialized[key] = str(value)
        
        return serialized
    
    def _deserialize_log_data(self, msg_data: Dict[bytes, bytes]) -> Dict[str, Any]:
        """Convert Redis message data back to Python objects"""
        deserialized = {}
        
        for key, value in msg_data.items():
            key_str = key.decode() if isinstance(key, bytes) else key
            value_str = value.decode() if isinstance(value, bytes) else value
            
            # Try to parse JSON
            if value_str and value_str[0] in ('{', '['):
                try:
                    deserialized[key_str] = json.loads(value_str)
                    continue
                except json.JSONDecodeError:
                    pass
            
            # Try to parse datetime
            if 'timestamp' in key_str or 'time' in key_str:
                try:
                    deserialized[key_str] = datetime.fromisoformat(value_str)
                    continue
                except (ValueError, AttributeError):
                    pass
            
            deserialized[key_str] = value_str
        
        return deserialized
    
    async def cleanup_old_messages(self, max_age_seconds: int = 86400):
        """
        Clean up old messages from the stream
        
        Args:
            max_age_seconds: Maximum age of messages in seconds (default: 24 hours)
        """
        try:
            # Calculate cutoff timestamp
            cutoff_time = datetime.utcnow().timestamp() - max_age_seconds
            cutoff_id = f"{int(cutoff_time * 1000)}-0"
            
            # Trim stream
            await self.redis.xtrim(
                name=self.stream_name,
                minid=cutoff_id,
                approximate=True
            )
            
            logger.info(f"Cleaned up messages older than {max_age_seconds}s")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old messages: {e}")


# Dependency injection
async def get_stream_service() -> StreamService:
    """Get StreamService instance"""
    redis = await get_redis()
    service = StreamService(redis)
    await service.initialize_stream()
    return service