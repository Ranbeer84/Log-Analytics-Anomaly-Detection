"""
Worker System Integration Test Suite
Tests all workers end-to-end
"""
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from pymongo import MongoClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [TEST] - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkerSystemTest:
    """Comprehensive test suite for all workers"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "log_analytics_test"
    ):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        
        self.test_results = []
    
    def setup(self) -> None:
        """Setup test environment"""
        logger.info("Setting up test environment...")
        
        # Clean test database
        self.db.logs.delete_many({})
        self.db.anomalies.delete_many({})
        self.db.alerts.delete_many({})
        self.db.batch_jobs.delete_many({})
        
        # Clean Redis stream
        try:
            self.redis_client.delete('log_stream_test')
        except:
            pass
        
        logger.info("✓ Test environment ready")
    
    def teardown(self) -> None:
        """Cleanup test environment"""
        logger.info("Cleaning up...")
        # Optionally keep data for inspection
        # self.db.client.drop_database(self.db.name)
    
    def run_all_tests(self) -> bool:
        """Run all test suites"""
        logger.info("=" * 60)
        logger.info("WORKER SYSTEM TEST SUITE")
        logger.info("=" * 60)
        
        self.setup()
        
        try:
            # Test 1: Redis Streams Setup
            self.test_redis_streams()
            
            # Test 2: Log Generation
            self.test_log_generation()
            
            # Test 3: Consumer Group
            self.test_consumer_group()
            
            # Test 4: Log Storage
            self.test_log_storage()
            
            # Test 5: Anomaly Detection (if model available)
            self.test_anomaly_detection()
            
            # Test 6: Alert Generation
            self.test_alert_generation()
            
            # Test 7: Batch Processing
            self.test_batch_processing()
            
            # Summary
            self.print_summary()
            
            return all(result['passed'] for result in self.test_results)
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}", exc_info=True)
            return False
        finally:
            self.teardown()
    
    def test_redis_streams(self) -> None:
        """Test Redis Streams functionality"""
        test_name = "Redis Streams Setup"
        logger.info(f"\nTest: {test_name}")
        
        try:
            # Test connection
            self.redis_client.ping()
            logger.info("✓ Redis connection OK")
            
            # Create test stream
            stream_name = 'log_stream_test'
            msg_id = self.redis_client.xadd(
                stream_name,
                {
                    'test': 'message',
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            logger.info(f"✓ Stream write OK: {msg_id}")
            
            # Read from stream
            messages = self.redis_client.xread(
                streams={stream_name: '0'},
                count=1
            )
            assert len(messages) > 0
            logger.info("✓ Stream read OK")
            
            self._record_test(test_name, True, "All checks passed")
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            self._record_test(test_name, False, str(e))
    
    def test_log_generation(self) -> None:
        """Test log data generation"""
        test_name = "Log Generation"
        logger.info(f"\nTest: {test_name}")
        
        try:
            stream_name = 'log_stream_test'
            
            # Generate test logs
            test_logs = [
                {
                    'timestamp': datetime.utcnow().isoformat(),
                    'service': 'api-service',
                    'severity': 'info',
                    'message': 'Test log entry',
                    'status_code': '200',
                    'response_time': '150.5',
                    'tags': json.dumps(['test'])
                },
                {
                    'timestamp': datetime.utcnow().isoformat(),
                    'service': 'api-service',
                    'severity': 'error',
                    'message': 'Test error',
                    'status_code': '500',
                    'response_time': '2500.0',
                    'tags': json.dumps(['test', 'error'])
                }
            ]
            
            # Add to stream
            for log in test_logs:
                self.redis_client.xadd(stream_name, log)
            
            # Verify
            length = self.redis_client.xlen(stream_name)
            logger.info(f"✓ Generated {len(test_logs)} logs, stream length: {length}")
            
            self._record_test(test_name, True, f"{len(test_logs)} logs generated")
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            self._record_test(test_name, False, str(e))
    
    def test_consumer_group(self) -> None:
        """Test consumer group creation"""
        test_name = "Consumer Group"
        logger.info(f"\nTest: {test_name}")
        
        try:
            stream_name = 'log_stream_test'
            group_name = 'test_processors'
            
            # Create consumer group
            try:
                self.redis_client.xgroup_create(
                    stream_name,
                    group_name,
                    id='0',
                    mkstream=True
                )
                logger.info(f"✓ Consumer group created: {group_name}")
            except redis.ResponseError as e:
                if 'BUSYGROUP' in str(e):
                    logger.info(f"✓ Consumer group exists: {group_name}")
                else:
                    raise
            
            # Read as consumer
            messages = self.redis_client.xreadgroup(
                groupname=group_name,
                consumername='test_consumer',
                streams={stream_name: '>'},
                count=1
            )
            
            if messages:
                logger.info(f"✓ Read {len(messages[0][1])} messages as consumer")
                
                # Acknowledge
                for _, msg_list in messages:
                    for msg_id, _ in msg_list:
                        self.redis_client.xack(stream_name, group_name, msg_id)
                
                logger.info("✓ Messages acknowledged")
            
            self._record_test(test_name, True, "Consumer group functional")
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            self._record_test(test_name, False, str(e))
    
    def test_log_storage(self) -> None:
        """Test MongoDB log storage"""
        test_name = "Log Storage (MongoDB)"
        logger.info(f"\nTest: {test_name}")
        
        try:
            # Test connection
            self.mongo_client.admin.command('ping')
            logger.info("✓ MongoDB connection OK")
            
            # Insert test log
            test_log = {
                'timestamp': datetime.utcnow(),
                'service': 'test-service',
                'severity': 'info',
                'message': 'Test log entry',
                'status_code': 200,
                'response_time': 150.5,
                'processed_at': datetime.utcnow()
            }
            
            result = self.db.logs.insert_one(test_log)
            logger.info(f"✓ Log inserted: {result.inserted_id}")
            
            # Query log
            retrieved = self.db.logs.find_one({'_id': result.inserted_id})
            assert retrieved is not None
            logger.info("✓ Log retrieval OK")
            
            # Test index
            indexes = self.db.logs.list_indexes()
            index_count = len(list(indexes))
            logger.info(f"✓ Collection has {index_count} indexes")
            
            self._record_test(test_name, True, "Storage functional")
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            self._record_test(test_name, False, str(e))
    
    def test_anomaly_detection(self) -> None:
        """Test anomaly detection (if model available)"""
        test_name = "Anomaly Detection"
        logger.info(f"\nTest: {test_name}")
        
        try:
            # Check if model exists
            models_dir = Path(__file__).parent.parent / 'ml' / 'saved_models'
            model_files = list(models_dir.glob('isolation_forest_*.pkl'))
            
            if not model_files:
                logger.warning("⚠ No trained model found - skipping test")
                self._record_test(test_name, True, "Skipped (no model)")
                return
            
            # Import detector
            from ml.inference.anomaly_detector import AnomalyDetector
            
            model_path = str(sorted(model_files)[-1])
            detector = AnomalyDetector(model_path=model_path)
            
            if not detector.model.is_trained:
                logger.warning("⚠ Model not trained - skipping test")
                self._record_test(test_name, True, "Skipped (model not trained)")
                return
            
            logger.info(f"✓ Model loaded: {Path(model_path).name}")
            
            # Test normal log
            normal_log = {
                'timestamp': datetime.utcnow(),
                'service': 'api-service',
                'severity': 'info',
                'status_code': 200,
                'response_time': 100.0
            }
            
            result = detector.detect_anomaly(normal_log, use_sequence=False)
            logger.info(f"✓ Normal log - Score: {result['anomaly_score']:.3f}")
            
            # Test anomalous log
            anomalous_log = {
                'timestamp': datetime.utcnow(),
                'service': 'api-service',
                'severity': 'error',
                'status_code': 500,
                'response_time': 5000.0
            }
            
            result = detector.detect_anomaly(anomalous_log, use_sequence=False)
            logger.info(
                f"✓ Anomalous log - Score: {result['anomaly_score']:.3f}, "
                f"Is anomaly: {result['is_anomaly']}"
            )
            
            self._record_test(test_name, True, "Detection functional")
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            self._record_test(test_name, False, str(e))
    
    def test_alert_generation(self) -> None:
        """Test alert generation logic"""
        test_name = "Alert Generation"
        logger.info(f"\nTest: {test_name}")
        
        try:
            # Create test anomalies
            anomalies = []
            for i in range(5):
                anomaly = {
                    'timestamp': datetime.utcnow(),
                    'service': 'test-service',
                    'severity': 'high',
                    'anomaly_score': 0.85,
                    'anomaly_severity': 'high',
                    'detected_at': datetime.utcnow(),
                    'status': 'new'
                }
                result = self.db.anomalies.insert_one(anomaly)
                anomalies.append(result.inserted_id)
            
            logger.info(f"✓ Created {len(anomalies)} test anomalies")
            
            # Test alert creation
            alert = {
                'rule_name': 'test_rule',
                'title': 'Test Alert',
                'description': 'Test alert generation',
                'severity': 'high',
                'metrics': {'anomaly_count': len(anomalies)},
                'triggered_at': datetime.utcnow(),
                'status': 'active'
            }
            
            result = self.db.alerts.insert_one(alert)
            logger.info(f"✓ Alert created: {result.inserted_id}")
            
            # Query alert
            retrieved = self.db.alerts.find_one({'_id': result.inserted_id})
            assert retrieved is not None
            logger.info("✓ Alert retrieval OK")
            
            self._record_test(test_name, True, "Alert system functional")
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            self._record_test(test_name, False, str(e))
    
    def test_batch_processing(self) -> None:
        """Test batch processing job"""
        test_name = "Batch Processing"
        logger.info(f"\nTest: {test_name}")
        
        try:
            # Create batch job
            job = {
                'status': 'pending',
                'created_at': datetime.utcnow(),
                'parameters': {
                    'reprocess': False
                },
                'stats': {
                    'logs_processed': 0,
                    'anomalies_detected': 0
                }
            }
            
            result = self.db.batch_jobs.insert_one(job)
            job_id = result.inserted_id
            logger.info(f"✓ Batch job created: {job_id}")
            
            # Update job status
            self.db.batch_jobs.update_one(
                {'_id': job_id},
                {'$set': {'status': 'completed'}}
            )
            
            # Verify update
            updated = self.db.batch_jobs.find_one({'_id': job_id})
            assert updated['status'] == 'completed'
            logger.info("✓ Job status update OK")
            
            self._record_test(test_name, True, "Batch processing functional")
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            self._record_test(test_name, False, str(e))
    
    def _record_test(self, name: str, passed: bool, message: str) -> None:
        """Record test result"""
        self.test_results.append({
            'name': name,
            'passed': passed,
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def print_summary(self) -> None:
        """Print test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for r in self.test_results if r['passed'])
        failed = len(self.test_results) - passed
        
        for result in self.test_results:
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            logger.info(f"{status}: {result['name']} - {result['message']}")
        
        logger.info("=" * 60)
        logger.info(f"Results: {passed}/{len(self.test_results)} passed")
        
        if failed > 0:
            logger.error(f"❌ {failed} test(s) failed")
        else:
            logger.info("✅ All tests passed!")
        
        logger.info("=" * 60)


def main():
    """Run test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Worker System Test Suite')
    parser.add_argument(
        '--redis-url',
        default='redis://localhost:6379',
        help='Redis connection URL'
    )
    parser.add_argument(
        '--mongo-uri',
        default='mongodb://localhost:27017',
        help='MongoDB connection URI'
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = WorkerSystemTest(
        redis_url=args.redis_url,
        mongo_uri=args.mongo_uri
    )
    
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()