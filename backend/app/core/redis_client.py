"""
Redis client for streaming and caching
"""
import redis.asyncio as redis
from app.config import settings
import logging
import json

logger = logging.getLogger(__name__)


class RedisClient:
    client: redis.Redis = None


redis_client = RedisClient()


async def connect_to_redis():
    """Connect to Redis"""
    try:
        logger.info("Connecting to Redis...")
        redis_client.client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        
        # Test connection
        await redis_client.client.ping()
        logger.info("✓ Successfully connected to Redis")
        
        # Create consumer group if it doesn't exist
        await create_stream_group()
        
    except Exception as e:
        logger.error(f"✗ Error connecting to Redis: {e}")
        raise


async def close_redis_connection():
    """Close Redis connection"""
    try:
        logger.info("Closing Redis connection...")
        if redis_client.client:
            await redis_client.client.close()
        logger.info("✓ Redis connection closed")
    except Exception as e:
        logger.error(f"✗ Error closing Redis: {e}")


async def create_stream_group():
    """Create Redis Stream consumer group"""
    try:
        # Try to create the consumer group
        await redis_client.client.xgroup_create(
            name=settings.REDIS_STREAM_NAME,
            groupname=settings.REDIS_CONSUMER_GROUP,
            id='0',
            mkstream=True
        )
        logger.info(f"✓ Created consumer group: {settings.REDIS_CONSUMER_GROUP}")
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info(f"Consumer group already exists: {settings.REDIS_CONSUMER_GROUP}")
        else:
            logger.error(f"✗ Error creating consumer group: {e}")
            raise


async def publish_log_to_stream(log_data: dict) -> str:
    """
    Publish log to Redis Stream
    
    Args:
        log_data: Log data dictionary
        
    Returns:
        Message ID
    """
    try:
        message_id = await redis_client.client.xadd(
            name=settings.REDIS_STREAM_NAME,
            fields={"data": json.dumps(log_data)}
        )
        return message_id
    except Exception as e:
        logger.error(f"✗ Error publishing to stream: {e}")
        raise


async def get_stream_length() -> int:
    """Get number of messages in stream"""
    try:
        return await redis_client.client.xlen(settings.REDIS_STREAM_NAME)
    except Exception as e:
        logger.error(f"✗ Error getting stream length: {e}")
        return 0


def get_redis() -> redis.Redis:
    """Get Redis client instance"""
    return redis_client.client