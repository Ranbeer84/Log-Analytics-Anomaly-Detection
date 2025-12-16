"""
Anomaly Detection Worker - Real-time ML inference on log streams
"""
import sys
import os
import time
import signal
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from pymongo import MongoClient

from ml.inference.anomaly_detector import AnomalyDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ANOMALY_WORKER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnomalyWorker:
    """Worker for real-time anomaly detection"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "log_analytics",
        model_path: Optional[str] = None,
        stream_name: str = "log_stream",
        consumer_group: str = "anomaly_detectors",
        consumer_name: str = "detector_1"
    ):
        """
        Initialize anomaly detection worker
        
        Args:
            redis_url: Redis connection URL
            mongo_uri: MongoDB connection URI
            db_name: Database name
            model_path: Path to trained model
            stream_name: Redis stream to consume
            consumer_group: Consumer group name
            consumer_name: Consumer instance name
        """
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        
        self.anomalies_collection = self.db['anomalies']
        self.logs_collection = self.db['logs']
        
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        
        # Initialize anomaly detector
        self.detector = AnomalyDetector(model_path=model_path)
        
        self.running = False
        self.stats = {
            'processed': 0,
            'anomalies_detected': 0,
            'errors': 0,
            'started_at': None
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Setup consumer group
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
                logger.info(f"Consumer group exists: {self.consumer_group}")
            else:
                raise
    
    def start(
        self,
        batch_size: int = 5,
        block_ms: int = 5000,
        use_sequence: bool = True
    ) -> None:
        """
        Start processing logs for anomaly detection
        
        Args:
            batch_size: Messages per batch
            block_ms: Blocking timeout
            use_sequence: Use sequence features
        """
        logger.info(f"Starting anomaly worker: {self.consumer_name}")
        
        if not self.detector.model.is_trained:
            logger.warning("Model not trained! Anomalies won't be detected.")
            logger.warning("Train a model first using ml/training/train_isolation_forest.py")
        
        self.running = True
        self.stats['started_at'] = datetime.utcnow().isoformat()
        
        while self.running:
            try:
                # Read from stream
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
                    self._process_batch(stream_messages, use_sequence)
                
            except Exception as e:
                logger.error(f"Error in worker loop: {str(e)}", exc_info=True)
                time.sleep(1)
        
        logger.info("Anomaly worker stopped")
    
    def _process_batch(self, messages: list, use_sequence: bool) -> None:
        """Process batch of messages for anomaly detection"""
        for message_id, data in messages:
            try:
                # Parse log data
                log_entry = self._parse_log_data(data)
                
                # Detect anomaly
                result = self.detector.detect_anomaly(
                    log_entry,
                    use_sequence=use_sequence
                )
                
                # Store anomaly if detected
                if result.get('is_anomaly', False):
                    self._store_anomaly(log_entry, result)
                    self.stats['anomalies_detected'] += 1
                    
                    logger.info(
                        f"Anomaly detected! Score: {result['anomaly_score']:.3f}, "
                        f"Severity: {result['severity']}"
                    )
                
                # Acknowledge message
                self.redis_client.xack(
                    self.stream_name,
                    self.consumer_group,
                    message_id
                )
                
                self.stats['processed'] += 1
                
                if self.stats['processed'] % 50 == 0:
                    logger.info(f"Processed {self.stats['processed']} logs")
                
            except Exception as e:
                logger.error(f"Error processing {message_id}: {str(e)}")
                self.stats['errors'] += 1
    
    def _parse_log_data(self, data: Dict) -> Dict:
        """Parse log data from Redis"""
        log_entry = {}
        
        for key, value in data.items():
            if key in ['metadata', 'tags']:
                try:
                    log_entry[key] = json.loads(value)
                except:
                    log_entry[key] = value
            elif key in ['response_time', 'status_code']:
                try:
                    log_entry[key] = float(value) if '.' in value else int(value)
                except:
                    log_entry[key] = value
            else:
                log_entry[key] = value
        
        return log_entry
    
    def _store_anomaly(self, log_entry: Dict, detection_result: Dict) -> None:
        """
        Store detected anomaly in database
        
        Args:
            log_entry: Original log entry
            detection_result: Detection result from ML model
        """
        anomaly_doc = {
            'log_id': log_entry.get('_id'),
            'timestamp': log_entry.get('timestamp', datetime.utcnow()),
            'service': log_entry.get('service'),
            'severity': log_entry.get('severity'),
            'message': log_entry.get('message'),
            'status_code': log_entry.get('status_code'),
            'response_time': log_entry.get('response_time'),
            
            # Anomaly detection metadata
            'anomaly_score': detection_result['anomaly_score'],
            'anomaly_severity': detection_result['severity'],
            'confidence': detection_result.get('confidence', 0),
            'detection_method': detection_result.get('detection_method', 'isolation_forest'),
            
            # Additional info
            'detected_at': datetime.utcnow(),
            'status': 'new',
            'investigated': False,
            'false_positive': False,
            
            # Full log for reference
            'original_log': log_entry
        }
        
        try:
            self.anomalies_collection.insert_one(anomaly_doc)
        except Exception as e:
            logger.error(f"Failed to store anomaly: {str(e)}")
    
    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received shutdown signal: {signum}")
        self.running = False
    
    def get_stats(self) -> Dict:
        """Get worker statistics"""
        detector_stats = self.detector.get_statistics()
        
        return {
            **self.stats,
            'detector_stats': detector_stats,
            'running': self.running,
            'current_time': datetime.utcnow().isoformat()
        }
    
    def stop(self) -> None:
        """Stop the worker"""
        self.running = False
        logger.info("Stopping anomaly worker...")


def main():
    """Main worker entry point"""
    logger.info("Starting Anomaly Detection Worker")
    
    # Configuration
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    db_name = os.getenv('DB_NAME', 'log_analytics')
    consumer_name = os.getenv('CONSUMER_NAME', f'detector_{os.getpid()}')
    
    # Find latest model
    models_dir = Path(__file__).parent.parent / 'ml' / 'saved_models'
    model_files = list(models_dir.glob('isolation_forest_*.pkl'))
    
    model_path = None
    if model_files:
        model_path = str(sorted(model_files)[-1])  # Use latest
        logger.info(f"Using model: {model_path}")
    else:
        logger.warning("No trained model found!")
    
    # Create worker
    worker = AnomalyWorker(
        redis_url=redis_url,
        mongo_uri=mongo_uri,
        db_name=db_name,
        model_path=model_path,
        consumer_name=consumer_name
    )
    
    try:
        # Start processing
        worker.start(batch_size=5, use_sequence=True)
    except Exception as e:
        logger.error(f"Worker failed: {str(e)}", exc_info=True)
    finally:
        stats = worker.get_stats()
        logger.info("=" * 50)
        logger.info("Worker Statistics:")
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Anomalies: {stats['anomalies_detected']}")
        logger.info(f"  Errors: {stats['errors']}")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()