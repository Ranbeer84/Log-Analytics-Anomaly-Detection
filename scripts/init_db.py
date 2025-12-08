"""
Database initialization script
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def init_database():
    """Initialize database with collections and indexes"""
    
    try:
        logger.info("🔧 Initializing database...")
        
        # Connect to MongoDB
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.MONGODB_DB_NAME]
        
        # Test connection
        await client.admin.command('ping')
        logger.info("✓ Connected to MongoDB")
        
        # Create collections
        collections = ['logs', 'anomalies', 'alerts', 'models']
        
        existing_collections = await db.list_collection_names()
        
        for collection_name in collections:
            if collection_name not in existing_collections:
                await db.create_collection(collection_name)
                logger.info(f"✓ Created collection: {collection_name}")
            else:
                logger.info(f"Collection already exists: {collection_name}")
        
        # Create indexes for logs collection
        logger.info("Creating indexes for logs collection...")
        await db.logs.create_index("timestamp")
        await db.logs.create_index("level")
        await db.logs.create_index("service")
        await db.logs.create_index([("timestamp", -1)])
        await db.logs.create_index([("service", 1), ("timestamp", -1)])
        await db.logs.create_index([("level", 1), ("timestamp", -1)])
        logger.info("✓ Logs indexes created")
        
        # Create indexes for anomalies collection
        logger.info("Creating indexes for anomalies collection...")
        await db.anomalies.create_index("timestamp")
        await db.anomalies.create_index("anomaly_score")
        await db.anomalies.create_index([("timestamp", -1)])
        await db.anomalies.create_index([("anomaly_score", -1)])
        logger.info("✓ Anomalies indexes created")
        
        # Create indexes for alerts collection
        logger.info("Creating indexes for alerts collection...")
        await db.alerts.create_index("created_at")
        await db.alerts.create_index("status")
        await db.alerts.create_index([("created_at", -1)])
        logger.info("✓ Alerts indexes created")
        
        # Get collection stats
        logs_count = await db.logs.count_documents({})
        anomalies_count = await db.anomalies.count_documents({})
        alerts_count = await db.alerts.count_documents({})
        
        logger.info("\n📊 Database Statistics:")
        logger.info(f"  Logs: {logs_count}")
        logger.info(f"  Anomalies: {anomalies_count}")
        logger.info(f"  Alerts: {alerts_count}")
        
        logger.info("\n✅ Database initialization complete!")
        
        # Close connection
        client.close()
        
    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(init_database())