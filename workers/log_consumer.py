"""
Log Consumer Worker - Consumes logs from Redis Streams
"""
import sys
import os
import time
import signal
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [LOG_CONSUMER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogConsumer:
    """Consumes log entries from Redis Streams and stores in MongoDB"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "log_analytics",
        stream_name: str = "log_stream",
        consumer_group: str = "log_processors",
        consumer_name: str = "consumer_1"
    ):
        """
        Initialize log consumer
        
        Args:
            redis_url: Redis connection URL
            mongo_uri: MongoDB connection URI
            db_name: Database name
            stream_name: Redis stream name
            consumer_group: Consumer group name
            consumer_name: Consumer instance name
        """
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.logs_collection = self.db['logs']
        
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        
        self.running = False
        self.stats = {
            'processed': 0,
            'errors': 0,
            'duplicates': 0,
            'started_at': None
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Create consumer group if not exists
        self._setup_consumer_group()
    
    def _setup_consumer_group(self) -> None:
        """Create consumer group if it doesn't exist"""
        try:
            self.redis_client.xgroup_create(
                self.stream_name,
                self.consumer_group,
                id='0',
                mkstream=True
            )
            logger.info(f"Created consumer group: {self.consumer_group}")
        except redis.ResponseError as e:
            if 'BUSYGROUP' in str(e):
                logger.info(f"Consumer group already exists: {self.consumer_group}")
            else:
                raise
    
    def start(self, batch_size: int = 10, block_ms: int = 5000) -> None:
        """
        Start consuming logs from Redis Stream
        
        Args:
            batch_size: Number of messages to read per batch
            block_ms: Blocking timeout in milliseconds
        """
        logger.info(f"Starting log consumer: {self.consumer_name}")
        self.running = True
        self.stats['started_at'] = datetime.utcnow().isoformat()
        
        while self.running:
            try:
                # Read messages from stream
                messages = self.redis_client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_name: '>'},
                    count=batch_size,
                    block=block_ms
                )
                
                if not messages:
                    continue
                
                # Process messages
                for stream_name, stream_messages in messages:
                    self._process_batch(stream_messages)
                
            except Exception as e:
                logger.error(f"Error in consumer loop: {str(e)}", exc_info=True)
                time.sleep(1)  # Backoff on error
        
        logger.info("Consumer stopped")
    
    def _process_batch(self, messages: List[tuple]) -> None:
        """
        Process a batch of messages
        
        Args:
            messages: List of (message_id, data) tuples
        """
        for message_id, data in messages:
            try:
                # Parse log data
                log_entry = self._parse_log_data(data)
                
                # Store in MongoDB
                self._store_log(log_entry)
                
                # Acknowledge message
                self.redis_client.xack(
                    self.stream_name,
                    self.consumer_group,
                    message_id
                )
                
                self.stats['processed'] += 1
                
                if self.stats['processed'] % 100 == 0:
                    logger.info(f"Processed {self.stats['processed']} logs")
                
            except DuplicateKeyError:
                # Log already exists
                self.stats['duplicates'] += 1
                self.redis_client.xack(
                    self.stream_name,
                    self.consumer_group,
                    message_id
                )
                
            except Exception as e:
                logger.error(f"Error processing message {message_id}: {str(e)}")
                self.stats['errors'] += 1
                # Don't acknowledge - message will be retried
    
    def _parse_log_data(self, data: Dict) -> Dict:
        """
        Parse log data from Redis
        
        Args:
            data: Raw message data
            
        Returns:
            Parsed log entry
        """
        # Redis stores everything as strings
        log_entry = {}
        
        for key, value in data.items():
            if key == 'metadata' or key == 'tags':
                # Parse JSON fields
                try:
                    log_entry[key] = json.loads(value)
                except:
                    log_entry[key] = value
            elif key in ['response_time', 'status_code']:
                # Convert numeric fields
                try:
                    log_entry[key] = float(value) if '.' in value else int(value)
                except:
                    log_entry[key] = value
            else:
                log_entry[key] = value
        
        # Add processing timestamp
        log_entry['processed_at'] = datetime.utcnow()
        
        return log_entry
    
    def _store_log(self, log_entry: Dict) -> None:
        """
        Store log entry in MongoDB
        
        Args:
            log_entry: Parsed log entry
        """
        self.logs_collection.insert_one(log_entry)
    
    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received shutdown signal: {signum}")
        self.running = False
    
    def get_stats(self) -> Dict:
        """Get consumer statistics"""
        return {
            **self.stats,
            'running': self.running,
            'current_time': datetime.utcnow().isoformat()
        }
    
    def stop(self) -> None:
        """Stop the consumer"""
        self.running = False
        logger.info("Stopping consumer...")


class PendingMessageProcessor:
    """Process pending messages that weren't acknowledged"""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        stream_name: str,
        consumer_group: str
    ):
        self.redis_client = redis_client
        self.stream_name = stream_name
        self.consumer_group = consumer_group
    
    def process_pending(
        self,
        consumer_name: str,
        max_retries: int = 3,
        min_idle_time: int = 60000  # 1 minute
    ) -> int:
        """
        Process pending messages
        
        Args:
            consumer_name: Consumer name to claim from
            max_retries: Maximum retry attempts
            min_idle_time: Minimum idle time in milliseconds
            
        Returns:
            Number of messages processed
        """
        processed = 0
        
        try:
            # Get pending messages
            pending = self.redis_client.xpending_range(
                self.stream_name,
                self.consumer_group,
                min='-',
                max='+',
                count=100
            )
            
            for message in pending:
                message_id = message['message_id']
                idle_time = message['time_since_delivered']
                delivery_count = message['times_delivered']
                
                # Skip if not idle long enough or too many retries
                if idle_time < min_idle_time or delivery_count > max_retries:
                    continue
                
                # Claim and reprocess
                claimed = self.redis_client.xclaim(
                    self.stream_name,
                    self.consumer_group,
                    consumer_name,
                    min_idle_time,
                    [message_id]
                )
                
                if claimed:
                    processed += 1
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing pending messages: {str(e)}")
            return processed


def main():
    """Main worker entry point"""
    logger.info("Starting Log Consumer Worker")
    
    # Configuration
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    db_name = os.getenv('DB_NAME', 'log_analytics')
    consumer_name = os.getenv('CONSUMER_NAME', f'consumer_{os.getpid()}')
    
    # Create consumer
    consumer = LogConsumer(
        redis_url=redis_url,
        mongo_uri=mongo_uri,
        db_name=db_name,
        consumer_name=consumer_name
    )
    
    try:
        # Start consuming
        consumer.start(batch_size=10, block_ms=5000)
    except Exception as e:
        logger.error(f"Consumer failed: {str(e)}", exc_info=True)
    finally:
        stats = consumer.get_stats()
        logger.info("=" * 50)
        logger.info("Consumer Statistics:")
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Errors: {stats['errors']}")
        logger.info(f"  Duplicates: {stats['duplicates']}")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()