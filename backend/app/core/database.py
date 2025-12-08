"""
Database connection and utilities
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class Database:
    client: AsyncIOMotorClient = None
    db: AsyncIOMotorDatabase = None


db = Database()


async def connect_to_mongo():
    """Connect to MongoDB"""
    try:
        logger.info("Connecting to MongoDB...")
        db.client = AsyncIOMotorClient(settings.MONGODB_URL)
        db.db = db.client[settings.MONGODB_DB_NAME]
        
        # Test connection
        await db.client.admin.command('ping')
        logger.info("✓ Successfully connected to MongoDB")
        
        # Create indexes
        await create_indexes()
        
    except Exception as e:
        logger.error(f"✗ Error connecting to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close MongoDB connection"""
    try:
        logger.info("Closing MongoDB connection...")
        if db.client:
            db.client.close()
        logger.info("✓ MongoDB connection closed")
    except Exception as e:
        logger.error(f"✗ Error closing MongoDB: {e}")


async def create_indexes():
    """Create database indexes for better performance"""
    try:
        # Logs collection indexes
        await db.db.logs.create_index("timestamp")
        await db.db.logs.create_index("level")
        await db.db.logs.create_index("service")
        await db.db.logs.create_index([("timestamp", -1)])
        
        # Anomalies collection indexes
        await db.db.anomalies.create_index("timestamp")
        await db.db.anomalies.create_index("anomaly_score")
        await db.db.anomalies.create_index([("timestamp", -1)])
        
        # Alerts collection indexes
        await db.db.alerts.create_index("created_at")
        await db.db.alerts.create_index("status")
        
        logger.info("✓ Database indexes created")
    except Exception as e:
        logger.error(f"✗ Error creating indexes: {e}")


def get_database() -> AsyncIOMotorDatabase:
    """Get database instance"""
    return db.db