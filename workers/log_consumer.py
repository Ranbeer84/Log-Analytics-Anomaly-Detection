"""
Enhanced Log Consumer Worker - Consumes logs from Redis Streams with production features
"""
import sys
import os
import time
import signal
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, BulkWriteError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [LOG_CONSUMER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogConsumer:
    """
    Production-ready log consumer with:
    - Bulk writes for performance
    - Pending message recovery
    - Health monitoring
    - Graceful degradation
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "log_analytics",
        stream_name: str = "log_stream",
        consumer_group: str = "log_processors",
        consumer_name: str = "consumer_1"
    ):
        """Initialize enhanced log consumer"""
        # Redis connection with retry logic
        self.redis_client = redis.from_url(
            redis_url, 
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30
        )
        
        # MongoDB connection with pooling
        self.mongo_client = MongoClient(
            mongo_uri,
            maxPoolSize=50,
            minPoolSize=10,
            socketTimeoutMS=30000,
            connectTimeoutMS=10000
        )
        self.db = self.mongo_client[db_name]
        self.logs_collection = self.db['logs']
        
        # Ensure indexes for performance
        self._ensure_indexes()
        
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        
        self.running = False
        self.stats = {
            'processed': 0,
            'errors': 0,
            'duplicates': 0,
            'retries': 0,
            'bulk_writes': 0,
            'pending_recovered': 0,
            'started_at': None,
            'last_processed_at': None
        }
        
        # Performance tracking
        self.recent_latencies = deque(maxlen=100)
        self.last_health_check = time.time()
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Setup consumer group
        self._setup_consumer_group()
    
    def _ensure_indexes(self) -> None:
        """Create indexes for optimal query performance"""
        try:
            # Compound index for timestamp-based queries
            self.logs_collection.create_index([
                ('timestamp', -1),
                ('service', 1)
            ])
            
            # Index for status-based filtering
            self.logs_collection.create_index('status_code')
            
            # Index for severity-based queries
            self.logs_collection.create_index('severity')
            
            # Text index for message search
            self.logs_collection.create_index([('message', 'text')])
            
            logger.info("Database indexes ensured")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    def _setup_consumer_group(self) -> None:
        """Create consumer group with proper error handling"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.redis_client.xgroup_create(
                    self.stream_name,
                    self.consumer_group,
                    id='0',
                    mkstream=True
                )
                logger.info(f"Created consumer group: {self.consumer_group}")
                return
            except redis.ResponseError as e:
                if 'BUSYGROUP' in str(e):
                    logger.info(f"Consumer group exists: {self.consumer_group}")
                    return
                else:
                    logger.error(f"Error creating consumer group: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        raise
            except redis.ConnectionError as e:
                logger.error(f"Redis connection error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
    
    def start(
        self, 
        batch_size: int = 10, 
        block_ms: int = 5000,
        bulk_write_threshold: int = 50,
        pending_check_interval: int = 300  # 5 minutes
    ) -> None:
        """
        Start consuming with bulk writes and pending recovery
        
        Args:
            batch_size: Messages to read per batch
            block_ms: Blocking timeout
            bulk_write_threshold: Batch writes when this many logs accumulated
            pending_check_interval: Seconds between pending message checks
        """
        logger.info(f"Starting consumer: {self.consumer_name}")
        logger.info(f"Batch size: {batch_size}, Bulk threshold: {bulk_write_threshold}")
        
        self.running = True
        self.stats['started_at'] = datetime.utcnow().isoformat()
        
        # Buffer for bulk writes
        write_buffer = []
        last_pending_check = time.time()
        
        while self.running:
            try:
                # Periodic pending message recovery
                if time.time() - last_pending_check > pending_check_interval:
                    self._recover_pending_messages()
                    last_pending_check = time.time()
                
                # Health check
                self._periodic_health_check()
                
                # Read from stream
                start_time = time.time()
                messages = self.redis_client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_name: '>'},
                    count=batch_size,
                    block=block_ms
                )
                
                if not messages:
                    # No new messages - flush buffer if exists
                    if write_buffer:
                        self._bulk_write(write_buffer)
                        write_buffer = []
                    continue
                
                # Track read latency
                read_latency = (time.time() - start_time) * 1000
                self.recent_latencies.append(read_latency)
                
                # Process messages
                for stream_name, stream_messages in messages:
                    processed = self._process_batch(stream_messages, write_buffer)
                    
                    # Bulk write if threshold reached
                    if len(write_buffer) >= bulk_write_threshold:
                        self._bulk_write(write_buffer)
                        write_buffer = []
                
                self.stats['last_processed_at'] = datetime.utcnow().isoformat()
                
            except redis.ConnectionError as e:
                logger.error(f"Redis connection error: {e}")
                time.sleep(5)  # Longer backoff for connection issues
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}", exc_info=True)
                time.sleep(1)
        
        # Final flush on shutdown
        if write_buffer:
            logger.info(f"Flushing {len(write_buffer)} remaining logs")
            self._bulk_write(write_buffer)
        
        logger.info("Consumer stopped gracefully")
    
    def _process_batch(
        self, 
        messages: List[Tuple], 
        write_buffer: List[Dict]
    ) -> int:
        """Process batch and add to write buffer"""
        message_ids = []
        
        for message_id, data in messages:
            try:
                # Parse log data
                log_entry = self._parse_log_data(data)
                log_entry['_redis_message_id'] = message_id
                
                # Add to buffer
                write_buffer.append({
                    'log': log_entry,
                    'message_id': message_id
                })
                message_ids.append(message_id)
                
            except Exception as e:
                logger.error(f"Error parsing message {message_id}: {e}")
                self.stats['errors'] += 1
                # Acknowledge bad message to prevent reprocessing
                self._safe_ack(message_id)
        
        return len(message_ids)
    
    def _bulk_write(self, buffer: List[Dict]) -> None:
        """
        Bulk write logs to MongoDB with proper error handling
        
        Args:
            buffer: List of {log, message_id} dicts
        """
        if not buffer:
            return
        
        try:
            # Extract logs for bulk insert
            logs = [item['log'] for item in buffer]
            
            # Bulk insert with ordered=False (continue on duplicates)
            result = self.logs_collection.insert_many(logs, ordered=False)
            
            inserted_count = len(result.inserted_ids)
            self.stats['processed'] += inserted_count
            self.stats['bulk_writes'] += 1
            
            # Acknowledge all messages
            message_ids = [item['message_id'] for item in buffer]
            if message_ids:
                self.redis_client.xack(
                    self.stream_name,
                    self.consumer_group,
                    *message_ids
                )
            
            if self.stats['processed'] % 500 == 0:
                logger.info(
                    f"Processed {self.stats['processed']} logs "
                    f"({self.stats['bulk_writes']} bulk writes)"
                )
            
        except BulkWriteError as e:
            # Handle partial success
            inserted_count = e.details.get('nInserted', 0)
            duplicate_count = len([
                err for err in e.details.get('writeErrors', [])
                if err['code'] == 11000  # Duplicate key error
            ])
            
            self.stats['processed'] += inserted_count
            self.stats['duplicates'] += duplicate_count
            
            # Acknowledge all (even duplicates)
            message_ids = [item['message_id'] for item in buffer]
            if message_ids:
                self.redis_client.xack(
                    self.stream_name,
                    self.consumer_group,
                    *message_ids
                )
            
            logger.warning(
                f"Bulk write partial success: {inserted_count} inserted, "
                f"{duplicate_count} duplicates"
            )
            
        except Exception as e:
            logger.error(f"Bulk write failed: {e}", exc_info=True)
            self.stats['errors'] += len(buffer)
            # Don't acknowledge - messages will be retried
    
    def _safe_ack(self, message_id: str) -> None:
        """Safely acknowledge a single message"""
        try:
            self.redis_client.xack(
                self.stream_name,
                self.consumer_group,
                message_id
            )
        except Exception as e:
            logger.error(f"Failed to acknowledge {message_id}: {e}")
    
    def _recover_pending_messages(self) -> None:
        """
        Recover pending messages that were delivered but not acknowledged
        """
        try:
            # Get pending messages info
            pending_info = self.redis_client.xpending(
                self.stream_name,
                self.consumer_group
            )
            
            if not pending_info or pending_info['pending'] == 0:
                return
            
            logger.info(f"Found {pending_info['pending']} pending messages")
            
            # Get detailed pending list
            pending = self.redis_client.xpending_range(
                self.stream_name,
                self.consumer_group,
                min='-',
                max='+',
                count=100
            )
            
            recovered = 0
            for msg in pending:
                message_id = msg['message_id']
                idle_time = msg['time_since_delivered']
                delivery_count = msg['times_delivered']
                
                # Only recover if idle > 60 seconds and delivery count <= 3
                if idle_time > 60000 and delivery_count <= 3:
                    try:
                        # Claim and reprocess
                        claimed = self.redis_client.xclaim(
                            self.stream_name,
                            self.consumer_group,
                            self.consumer_name,
                            60000,  # min idle time
                            [message_id]
                        )
                        
                        if claimed:
                            # Process claimed message
                            for msg_id, data in claimed:
                                log_entry = self._parse_log_data(data)
                                self._store_log(log_entry)
                                self._safe_ack(msg_id)
                                recovered += 1
                                
                    except Exception as e:
                        logger.error(f"Failed to recover {message_id}: {e}")
                
                elif delivery_count > 3:
                    # Dead letter - acknowledge to prevent infinite retries
                    logger.warning(
                        f"Message {message_id} exceeded retry limit, "
                        f"moving to dead letter"
                    )
                    self._safe_ack(message_id)
            
            if recovered > 0:
                self.stats['pending_recovered'] += recovered
                logger.info(f"Recovered {recovered} pending messages")
                
        except Exception as e:
            logger.error(f"Pending recovery failed: {e}", exc_info=True)
    
    def _parse_log_data(self, data: Dict) -> Dict:
        """Parse and enrich log data"""
        log_entry = {}
        
        for key, value in data.items():
            if key in ['metadata', 'tags']:
                try:
                    log_entry[key] = json.loads(value)
                except:
                    log_entry[key] = value
            elif key in ['response_time', 'status_code']:
                try:
                    log_entry[key] = float(value) if '.' in str(value) else int(value)
                except:
                    log_entry[key] = value
            else:
                log_entry[key] = value
        
        # Add processing metadata
        log_entry['processed_at'] = datetime.utcnow()
        log_entry['processor'] = self.consumer_name
        
        return log_entry
    
    def _store_log(self, log_entry: Dict) -> None:
        """Store single log entry (for recovery path)"""
        try:
            self.logs_collection.insert_one(log_entry)
            self.stats['processed'] += 1
        except DuplicateKeyError:
            self.stats['duplicates'] += 1
        except Exception as e:
            logger.error(f"Failed to store log: {e}")
            self.stats['errors'] += 1
    
    def _periodic_health_check(self) -> None:
        """Perform periodic health checks"""
        now = time.time()
        if now - self.last_health_check < 60:  # Check every minute
            return
        
        try:
            # Check Redis connection
            self.redis_client.ping()
            
            # Check MongoDB connection
            self.mongo_client.admin.command('ping')
            
            # Log health metrics
            avg_latency = (
                sum(self.recent_latencies) / len(self.recent_latencies)
                if self.recent_latencies else 0
            )
            
            logger.info(
                f"Health check OK - Avg latency: {avg_latency:.2f}ms, "
                f"Processed: {self.stats['processed']}"
            )
            
            self.last_health_check = now
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def get_stats(self) -> Dict:
        """Get comprehensive consumer statistics"""
        avg_latency = (
            sum(self.recent_latencies) / len(self.recent_latencies)
            if self.recent_latencies else 0
        )
        
        return {
            **self.stats,
            'running': self.running,
            'avg_latency_ms': round(avg_latency, 2),
            'current_time': datetime.utcnow().isoformat()
        }
    
    def stop(self) -> None:
        """Stop the consumer"""
        self.running = False
        logger.info("Consumer stop requested")


def main():
    """Main worker entry point"""
    logger.info("=" * 60)
    logger.info("Starting Enhanced Log Consumer Worker")
    logger.info("=" * 60)
    
    # Configuration from environment
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    db_name = os.getenv('DB_NAME', 'log_analytics')
    consumer_name = os.getenv('CONSUMER_NAME', f'consumer_{os.getpid()}')
    batch_size = int(os.getenv('BATCH_SIZE', '10'))
    bulk_threshold = int(os.getenv('BULK_WRITE_THRESHOLD', '50'))
    
    logger.info(f"Configuration:")
    logger.info(f"  Consumer Name: {consumer_name}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Bulk Write Threshold: {bulk_threshold}")
    logger.info(f"  Redis: {redis_url}")
    logger.info(f"  MongoDB: {mongo_uri}")
    
    # Create and start consumer
    consumer = LogConsumer(
        redis_url=redis_url,
        mongo_uri=mongo_uri,
        db_name=db_name,
        consumer_name=consumer_name
    )
    
    try:
        consumer.start(
            batch_size=batch_size,
            bulk_write_threshold=bulk_threshold
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Consumer failed: {e}", exc_info=True)
    finally:
        stats = consumer.get_stats()
        logger.info("=" * 60)
        logger.info("Final Consumer Statistics:")
        logger.info(f"  Total Processed: {stats['processed']}")
        logger.info(f"  Bulk Writes: {stats['bulk_writes']}")
        logger.info(f"  Duplicates: {stats['duplicates']}")
        logger.info(f"  Pending Recovered: {stats['pending_recovered']}")
        logger.info(f"  Errors: {stats['errors']}")
        logger.info(f"  Avg Latency: {stats['avg_latency_ms']:.2f}ms")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()