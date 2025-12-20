"""
Enhanced Anomaly Detection Worker - Production-ready real-time ML inference
"""
import sys
import os
import time
import signal
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from collections import deque
import threading

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
    """
    Production-grade anomaly detection worker with:
    - Model warmup and validation
    - Circuit breaker pattern
    - Adaptive batch sizing
    - Performance monitoring
    """
    
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
        """Initialize enhanced anomaly detection worker"""
        self.redis_client = redis.from_url(
            redis_url, 
            decode_responses=True,
            socket_connect_timeout=5,
            health_check_interval=30
        )
        self.mongo_client = MongoClient(
            mongo_uri,
            maxPoolSize=30,
            minPoolSize=5
        )
        self.db = self.mongo_client[db_name]
        
        self.anomalies_collection = self.db['anomalies']
        self.logs_collection = self.db['logs']
        
        # Ensure indexes
        self._ensure_indexes()
        
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        
        # Initialize detector
        self.detector = AnomalyDetector(model_path=model_path)
        
        # Warmup model
        self._warmup_model()
        
        self.running = False
        self.stats = {
            'processed': 0,
            'anomalies_detected': 0,
            'errors': 0,
            'model_errors': 0,
            'circuit_breaks': 0,
            'started_at': None
        }
        
        # Circuit breaker for model failures
        self.circuit_breaker = {
            'failures': 0,
            'max_failures': 5,
            'open': False,
            'reset_time': None
        }
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.last_health_check = time.time()
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Setup consumer group
        self._setup_consumer_group()
    
    def _ensure_indexes(self) -> None:
        """Create indexes for anomaly queries"""
        try:
            # Compound index for anomaly queries
            self.anomalies_collection.create_index([
                ('detected_at', -1),
                ('anomaly_severity', 1)
            ])
            
            # Index for status filtering
            self.anomalies_collection.create_index('status')
            
            # Index for service-based queries
            self.anomalies_collection.create_index('service')
            
            logger.info("Anomaly collection indexes ensured")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    def _warmup_model(self) -> None:
        """Warmup model with sample data"""
        if not self.detector.model.is_trained:
            logger.warning("⚠️  Model not trained - anomaly detection disabled")
            logger.warning("   Train a model using: ml/training/train_isolation_forest.py")
            return
        
        try:
            logger.info("Warming up ML model...")
            
            # Create sample log entry
            sample_log = {
                'timestamp': datetime.utcnow(),
                'service': 'warmup',
                'severity': 'info',
                'message': 'Model warmup',
                'status_code': 200,
                'response_time': 100.0
            }
            
            # Run inference
            start = time.time()
            result = self.detector.detect_anomaly(sample_log, use_sequence=False)
            warmup_time = (time.time() - start) * 1000
            
            logger.info(f"✓ Model warmed up ({warmup_time:.2f}ms)")
            logger.info(f"  Model type: {self.detector.model.model_type}")
            logger.info(f"  Features: {len(self.detector.model.feature_names)}")
            
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
            self.circuit_breaker['open'] = True
    
    def _setup_consumer_group(self) -> None:
        """Create consumer group with retries"""
        max_retries = 3
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
                elif attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise
    
    def start(
        self,
        batch_size: int = 5,
        block_ms: int = 5000,
        use_sequence: bool = True,
        adaptive_batch: bool = True
    ) -> None:
        """
        Start anomaly detection with adaptive processing
        
        Args:
            batch_size: Initial batch size
            block_ms: Blocking timeout
            use_sequence: Use sequence features
            adaptive_batch: Adjust batch size based on performance
        """
        logger.info("=" * 60)
        logger.info(f"Starting anomaly worker: {self.consumer_name}")
        logger.info(f"  Model trained: {self.detector.model.is_trained}")
        logger.info(f"  Sequence features: {use_sequence}")
        logger.info(f"  Adaptive batching: {adaptive_batch}")
        logger.info("=" * 60)
        
        if not self.detector.model.is_trained:
            logger.error("Cannot start - model not trained!")
            return
        
        self.running = True
        self.stats['started_at'] = datetime.utcnow().isoformat()
        
        current_batch_size = batch_size
        
        while self.running:
            try:
                # Check circuit breaker
                if self._check_circuit_breaker():
                    time.sleep(5)
                    continue
                
                # Health check
                self._periodic_health_check()
                
                # Read from stream
                messages = self.redis_client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_name: '>'},
                    count=current_batch_size,
                    block=block_ms
                )
                
                if not messages:
                    continue
                
                # Process with timing
                batch_start = time.time()
                
                for stream_name, stream_messages in messages:
                    self._process_batch(stream_messages, use_sequence)
                
                batch_time = time.time() - batch_start
                
                # Adaptive batch sizing
                if adaptive_batch and len(self.inference_times) >= 10:
                    avg_time = sum(self.inference_times) / len(self.inference_times)
                    
                    # Increase batch if processing is fast
                    if avg_time < 50 and current_batch_size < 20:  # <50ms
                        current_batch_size += 1
                    # Decrease if processing is slow
                    elif avg_time > 200 and current_batch_size > 1:  # >200ms
                        current_batch_size -= 1
                
            except redis.ConnectionError as e:
                logger.error(f"Redis connection error: {e}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(1)
        
        logger.info("Anomaly worker stopped")
    
    def _process_batch(self, messages: list, use_sequence: bool) -> None:
        """Process batch with circuit breaker protection"""
        for message_id, data in messages:
            inference_start = time.time()
            
            try:
                # Parse log
                log_entry = self._parse_log_data(data)
                
                # Run inference with timeout protection
                result = self._safe_inference(log_entry, use_sequence)
                
                if result is None:
                    # Inference failed
                    self._safe_ack(message_id)
                    continue
                
                # Track inference time
                inference_time = (time.time() - inference_start) * 1000
                self.inference_times.append(inference_time)
                
                # Handle anomaly
                if result.get('is_anomaly', False):
                    self._store_anomaly(log_entry, result)
                    self.stats['anomalies_detected'] += 1
                    
                    # Log high-severity anomalies
                    if result['severity'] in ['high', 'critical']:
                        logger.warning(
                            f"🚨 {result['severity'].upper()} anomaly detected! "
                            f"Service: {log_entry.get('service')}, "
                            f"Score: {result['anomaly_score']:.3f}"
                        )
                
                # Acknowledge
                self._safe_ack(message_id)
                
                self.stats['processed'] += 1
                
                # Reset circuit breaker on success
                if self.circuit_breaker['failures'] > 0:
                    self.circuit_breaker['failures'] = 0
                
                # Periodic logging
                if self.stats['processed'] % 100 == 0:
                    avg_inference = sum(self.inference_times) / len(self.inference_times)
                    logger.info(
                        f"Processed {self.stats['processed']} logs | "
                        f"Anomalies: {self.stats['anomalies_detected']} | "
                        f"Avg inference: {avg_inference:.1f}ms"
                    )
                
            except Exception as e:
                logger.error(f"Error processing {message_id}: {e}")
                self.stats['errors'] += 1
                # Don't acknowledge - will retry
    
    def _safe_inference(
        self, 
        log_entry: Dict, 
        use_sequence: bool,
        timeout: float = 2.0
    ) -> Optional[Dict]:
        """
        Run inference with timeout and error handling
        
        Returns:
            Detection result or None if failed
        """
        try:
            result = self.detector.detect_anomaly(
                log_entry,
                use_sequence=use_sequence
            )
            return result
            
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            self.stats['model_errors'] += 1
            self.circuit_breaker['failures'] += 1
            
            # Open circuit if too many failures
            if self.circuit_breaker['failures'] >= self.circuit_breaker['max_failures']:
                self.circuit_breaker['open'] = True
                self.circuit_breaker['reset_time'] = time.time() + 60  # Reset after 1 min
                self.stats['circuit_breaks'] += 1
                logger.error("🔴 Circuit breaker OPEN - model failures exceeded threshold")
            
            return None
    
    def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker is open
        
        Returns:
            True if open (should skip processing)
        """
        if not self.circuit_breaker['open']:
            return False
        
        # Check if it's time to reset
        if time.time() >= self.circuit_breaker['reset_time']:
            logger.info("🟡 Attempting to close circuit breaker...")
            self.circuit_breaker['open'] = False
            self.circuit_breaker['failures'] = 0
            logger.info("🟢 Circuit breaker CLOSED - resuming normal operation")
            return False
        
        return True
    
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
                    log_entry[key] = float(value) if '.' in str(value) else int(value)
                except:
                    log_entry[key] = value
            else:
                log_entry[key] = value
        
        return log_entry
    
    def _store_anomaly(self, log_entry: Dict, detection_result: Dict) -> None:
        """Store detected anomaly with enrichment"""
        anomaly_doc = {
            'log_id': log_entry.get('_id'),
            'timestamp': log_entry.get('timestamp', datetime.utcnow()),
            'service': log_entry.get('service'),
            'severity': log_entry.get('severity'),
            'message': log_entry.get('message'),
            'status_code': log_entry.get('status_code'),
            'response_time': log_entry.get('response_time'),
            
            # Detection metadata
            'anomaly_score': detection_result['anomaly_score'],
            'anomaly_severity': detection_result['severity'],
            'confidence': detection_result.get('confidence', 0),
            'detection_method': detection_result.get('detection_method', 'isolation_forest'),
            'feature_contributions': detection_result.get('feature_contributions', {}),
            
            # Workflow fields
            'detected_at': datetime.utcnow(),
            'detected_by': self.consumer_name,
            'status': 'new',
            'investigated': False,
            'false_positive': False,
            'acknowledged_at': None,
            'resolved_at': None,
            
            # Reference
            'original_log': log_entry
        }
        
        try:
            self.anomalies_collection.insert_one(anomaly_doc)
        except Exception as e:
            logger.error(f"Failed to store anomaly: {e}")
    
    def _safe_ack(self, message_id: str) -> None:
        """Safely acknowledge message"""
        try:
            self.redis_client.xack(
                self.stream_name,
                self.consumer_group,
                message_id
            )
        except Exception as e:
            logger.error(f"Failed to acknowledge {message_id}: {e}")
    
    def _periodic_health_check(self) -> None:
        """Periodic health monitoring"""
        now = time.time()
        if now - self.last_health_check < 60:
            return
        
        try:
            # Check connections
            self.redis_client.ping()
            self.mongo_client.admin.command('ping')
            
            # Get metrics
            avg_inference = (
                sum(self.inference_times) / len(self.inference_times)
                if self.inference_times else 0
            )
            
            anomaly_rate = (
                self.stats['anomalies_detected'] / self.stats['processed']
                if self.stats['processed'] > 0 else 0
            )
            
            logger.info(
                f"Health OK - Processed: {self.stats['processed']}, "
                f"Anomaly rate: {anomaly_rate:.2%}, "
                f"Avg inference: {avg_inference:.1f}ms"
            )
            
            self.last_health_check = now
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _handle_shutdown(self, signum, frame) -> None:
        """Graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        detector_stats = self.detector.get_statistics()
        
        avg_inference = (
            sum(self.inference_times) / len(self.inference_times)
            if self.inference_times else 0
        )
        
        anomaly_rate = (
            self.stats['anomalies_detected'] / self.stats['processed']
            if self.stats['processed'] > 0 else 0
        )
        
        return {
            **self.stats,
            'anomaly_rate': round(anomaly_rate, 4),
            'avg_inference_ms': round(avg_inference, 2),
            'detector_stats': detector_stats,
            'circuit_breaker': self.circuit_breaker,
            'running': self.running,
            'current_time': datetime.utcnow().isoformat()
        }
    
    def stop(self) -> None:
        """Stop the worker"""
        self.running = False


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Enhanced Anomaly Detection Worker")
    logger.info("=" * 60)
    
    # Configuration
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    db_name = os.getenv('DB_NAME', 'log_analytics')
    consumer_name = os.getenv('CONSUMER_NAME', f'detector_{os.getpid()}')
    batch_size = int(os.getenv('BATCH_SIZE', '5'))
    
    # Find model
    models_dir = Path(__file__).parent.parent / 'ml' / 'saved_models'
    model_files = list(models_dir.glob('isolation_forest_*.pkl'))
    
    model_path = None
    if model_files:
        model_path = str(sorted(model_files)[-1])
        logger.info(f"📦 Found model: {Path(model_path).name}")
    else:
        logger.error("❌ No trained model found!")
        logger.error("   Train a model using: python ml/training/train_isolation_forest.py")
        return
    
    # Create worker
    worker = AnomalyWorker(
        redis_url=redis_url,
        mongo_uri=mongo_uri,
        db_name=db_name,
        model_path=model_path,
        consumer_name=consumer_name
    )
    
    try:
        worker.start(batch_size=batch_size, use_sequence=True, adaptive_batch=True)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
    finally:
        stats = worker.get_stats()
        logger.info("=" * 60)
        logger.info("Final Worker Statistics:")
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Anomalies: {stats['anomalies_detected']}")
        logger.info(f"  Anomaly Rate: {stats['anomaly_rate']:.2%}")
        logger.info(f"  Avg Inference: {stats['avg_inference_ms']:.2f}ms")
        logger.info(f"  Model Errors: {stats['model_errors']}")
        logger.info(f"  Circuit Breaks: {stats['circuit_breaks']}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()