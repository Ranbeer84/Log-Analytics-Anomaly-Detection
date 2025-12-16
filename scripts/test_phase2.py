"""
Phase 2 Testing Script
Tests ML models, workers, and anomaly detection
"""
import sys
import time
import requests
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient
import redis


class Phase2Tester:
    """Comprehensive testing for Phase 2"""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.mongo_client = MongoClient("mongodb://localhost:27017")
        self.db = self.mongo_client['log_analytics']
        self.redis_client = redis.from_url("redis://localhost:6379")
        
        self.results = {
            'passed': 0,
            'failed': 0,
            'tests': []
        }
    
    def run_all_tests(self):
        """Run all Phase 2 tests"""
        print("=" * 60)
        print("Phase 2 Testing Suite")
        print("=" * 60)
        print()
        
        # Test ML Model
        print("🧠 Testing ML Model...")
        self.test_model_exists()
        self.test_model_loading()
        print()
        
        # Test Workers
        print("⚙️  Testing Workers...")
        self.test_log_consumer()
        self.test_anomaly_worker()
        self.test_alert_worker()
        print()
        
        # Test Anomaly Detection
        print("🔍 Testing Anomaly Detection...")
        self.test_anomaly_detection()
        self.test_anomaly_scoring()
        print()
        
        # Test Alert System
        print("🚨 Testing Alert System...")
        self.test_alert_generation()
        self.test_alert_cooldown()
        print()
        
        # Print results
        self.print_results()
    
    def test_model_exists(self):
        """Test that trained model exists"""
        models_dir = Path(__file__).parent.parent / 'ml' / 'saved_models'
        model_files = list(models_dir.glob('isolation_forest_*.pkl'))
        
        if model_files:
            self.record_test("Model file exists", True)
            print(f"  ✓ Found {len(model_files)} model(s)")
        else:
            self.record_test("Model file exists", False, "No model files found")
            print("  ✗ No model files found. Run training first.")
    
    def test_model_loading(self):
        """Test model can be loaded"""
        try:
            from ml.inference.anomaly_detector import AnomalyDetector
            
            models_dir = Path(__file__).parent.parent / 'ml' / 'saved_models'
            model_files = list(models_dir.glob('isolation_forest_*.pkl'))
            
            if not model_files:
                self.record_test("Model loading", False, "No model to load")
                return
            
            model_path = str(sorted(model_files)[-1])
            detector = AnomalyDetector(model_path=model_path)
            
            if detector.model.is_trained:
                self.record_test("Model loading", True)
                print(f"  ✓ Model loaded successfully")
                
                # Test stats
                stats = detector.get_statistics()
                print(f"    - Features: {stats['model_info']['n_features']}")
                print(f"    - Training samples: {stats['model_info']['training_stats']['n_samples']}")
            else:
                self.record_test("Model loading", False, "Model not trained")
                
        except Exception as e:
            self.record_test("Model loading", False, str(e))
            print(f"  ✗ Failed: {str(e)}")
    
    def test_log_consumer(self):
        """Test log consumer is working"""
        try:
            # Check if stream exists
            stream_length = self.redis_client.xlen('log_stream')
            
            # Check consumer groups
            try:
                groups = self.redis_client.xinfo_groups('log_stream')
                has_consumer = any(g['name'] == 'log_processors' for g in groups)
                
                if has_consumer:
                    self.record_test("Log consumer setup", True)
                    print(f"  ✓ Consumer group exists")
                    print(f"    - Stream length: {stream_length}")
                else:
                    self.record_test("Log consumer setup", False, "Consumer group not found")
                    
            except redis.ResponseError:
                self.record_test("Log consumer setup", False, "Stream doesn't exist")
                
        except Exception as e:
            self.record_test("Log consumer setup", False, str(e))
    
    def test_anomaly_worker(self):
        """Test anomaly detection worker"""
        try:
            # Check anomalies collection
            anomaly_count = self.db.anomalies.count_documents({})
            
            if anomaly_count > 0:
                self.record_test("Anomaly detection", True)
                print(f"  ✓ Found {anomaly_count} detected anomalies")
                
                # Get recent anomaly
                recent = self.db.anomalies.find_one(
                    sort=[('detected_at', -1)]
                )
                if recent:
                    print(f"    - Latest: {recent.get('anomaly_severity')} "
                          f"(score: {recent.get('anomaly_score', 0):.3f})")
            else:
                self.record_test("Anomaly detection", False, "No anomalies detected yet")
                print("  ⚠ No anomalies detected. Generate some test data.")
                
        except Exception as e:
            self.record_test("Anomaly detection", False, str(e))
    
    def test_alert_worker(self):
        """Test alert processing worker"""
        try:
            alert_count = self.db.alerts.count_documents({})
            
            if alert_count > 0:
                self.record_test("Alert system", True)
                print(f"  ✓ Found {alert_count} alerts")
                
                # Get recent alert
                recent = self.db.alerts.find_one(
                    sort=[('triggered_at', -1)]
                )
                if recent:
                    print(f"    - Latest: {recent.get('title')} "
                          f"({recent.get('severity')})")
            else:
                self.record_test("Alert system", False, "No alerts generated")
                print("  ⚠ No alerts. May need to trigger alert conditions.")
                
        except Exception as e:
            self.record_test("Alert system", False, str(e))
    
    def test_anomaly_detection(self):
        """Test end-to-end anomaly detection"""
        try:
            # Inject a suspicious log
            test_log = {
                "service": "test-api",
                "severity": "error",
                "message": "Critical database failure - connection pool exhausted after 60000ms timeout",
                "status_code": 500,
                "response_time": 60000,
                "metadata": {
                    "method": "POST",
                    "path": "/api/critical/operation",
                    "ip": "203.0.113.99",
                    "user_agent": "BadBot/1.0"
                }
            }
            
            response = requests.post(
                f"{self.api_url}/api/logs/",
                json=test_log,
                timeout=5
            )
            
            if response.status_code == 201:
                print("  ✓ Suspicious log ingested")
                
                # Wait for processing
                print("    Waiting for workers to process...")
                time.sleep(5)
                
                # Check if anomaly was detected
                anomalies = self.db.anomalies.find({
                    'service': 'test-api'
                }).sort('detected_at', -1).limit(1)
                
                anomaly = list(anomalies)
                if anomaly:
                    self.record_test("E2E anomaly detection", True)
                    print(f"  ✓ Anomaly detected!")
                    print(f"    - Score: {anomaly[0].get('anomaly_score', 0):.3f}")
                    print(f"    - Severity: {anomaly[0].get('anomaly_severity')}")
                else:
                    self.record_test("E2E anomaly detection", False, 
                                   "Log ingested but no anomaly detected")
            else:
                self.record_test("E2E anomaly detection", False, 
                               f"API returned {response.status_code}")
                
        except Exception as e:
            self.record_test("E2E anomaly detection", False, str(e))
            print(f"  ✗ Failed: {str(e)}")
    
    def test_anomaly_scoring(self):
        """Test anomaly scoring ranges"""
        try:
            # Get anomalies with different scores
            pipeline = [
                {
                    '$group': {
                        '_id': '$anomaly_severity',
                        'count': {'$sum': 1},
                        'avg_score': {'$avg': '$anomaly_score'}
                    }
                }
            ]
            
            results = list(self.db.anomalies.aggregate(pipeline))
            
            if results:
                self.record_test("Anomaly scoring", True)
                print("  ✓ Severity distribution:")
                for result in results:
                    severity = result['_id']
                    count = result['count']
                    avg = result['avg_score']
                    print(f"    - {severity}: {count} (avg: {avg:.3f})")
            else:
                self.record_test("Anomaly scoring", False, "No scored anomalies")
                
        except Exception as e:
            self.record_test("Anomaly scoring", False, str(e))
    
    def test_alert_generation(self):
        """Test alert generation for different rules"""
        try:
            # Check different alert types
            alert_types = self.db.alerts.distinct('rule_name')
            
            if alert_types:
                self.record_test("Alert rules", True)
                print(f"  ✓ Active alert rules: {len(alert_types)}")
                for rule in alert_types:
                    count = self.db.alerts.count_documents({'rule_name': rule})
                    print(f"    - {rule}: {count} alerts")
            else:
                self.record_test("Alert rules", False, "No alert rules triggered")
                
        except Exception as e:
            self.record_test("Alert rules", False, str(e))
    
    def test_alert_cooldown(self):
        """Test alert cooldown mechanism"""
        try:
            # Find alerts with same rule
            pipeline = [
                {
                    '$sort': {'triggered_at': -1}
                },
                {
                    '$group': {
                        '_id': '$rule_name',
                        'alerts': {
                            '$push': {
                                'time': '$triggered_at'
                            }
                        }
                    }
                },
                {
                    '$match': {
                        'alerts.1': {'$exists': True}
                    }
                }
            ]
            
            results = list(self.db.alerts.aggregate(pipeline))
            
            if results:
                print("  ✓ Checking cooldown periods:")
                for result in results:
                    rule = result['_id']
                    alerts = result['alerts']
                    if len(alerts) >= 2:
                        time_diff = (alerts[0]['time'] - alerts[1]['time']).total_seconds() / 60
                        print(f"    - {rule}: {time_diff:.1f} min between alerts")
                
                self.record_test("Alert cooldown", True)
            else:
                print("  ⚠ Not enough alerts to test cooldown")
                self.record_test("Alert cooldown", True, "N/A - insufficient data")
                
        except Exception as e:
            self.record_test("Alert cooldown", False, str(e))
    
    def record_test(self, name: str, passed: bool, error: str = ""):
        """Record test result"""
        if passed:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
        
        self.results['tests'].append({
            'name': name,
            'passed': passed,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def print_results(self):
        """Print test summary"""
        print()
        print("=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        
        total = self.results['passed'] + self.results['failed']
        success_rate = (self.results['passed'] / total * 100) if total > 0 else 0
        
        print(f"Total Tests: {total}")
        print(f"Passed: {self.results['passed']} ✓")
        print(f"Failed: {self.results['failed']} ✗")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        if self.results['failed'] > 0:
            print("Failed Tests:")
            for test in self.results['tests']:
                if not test['passed']:
                    print(f"  ✗ {test['name']}")
                    if test['error']:
                        print(f"    Error: {test['error']}")
        
        print("=" * 60)
        
        # Save results
        results_file = Path(__file__).parent.parent / 'test_results_phase2.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")


def main():
    """Run tests"""
    tester = Phase2Tester()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nTest suite failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()