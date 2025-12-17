"""
Comprehensive test script for Phases 1-3
Tests all components to ensure they work together
"""
import asyncio
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import httpx
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis

# Color codes for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class PhaseValidator:
    """Validator for all phases"""
    
    def __init__(self):
        self.results = {
            "phase1": {"passed": 0, "failed": 0, "tests": []},
            "phase2": {"passed": 0, "failed": 0, "tests": []},
            "phase3": {"passed": 0, "failed": 0, "tests": []}
        }
        self.api_base = "http://localhost:8000"
        self.mongo_uri = "mongodb://localhost:27017"
        self.redis_uri = "redis://localhost:6379"
    
    def print_header(self, text: str):
        """Print colored header"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
    
    def print_test(self, phase: str, test_name: str, passed: bool, message: str = ""):
        """Print test result"""
        status = f"{Colors.OKGREEN}✓ PASS{Colors.ENDC}" if passed else f"{Colors.FAIL}✗ FAIL{Colors.ENDC}"
        print(f"  {status} - {test_name}")
        if message:
            print(f"    {Colors.OKCYAN}→{Colors.ENDC} {message}")
        
        self.results[phase]["tests"].append({
            "name": test_name,
            "passed": passed,
            "message": message
        })
        
        if passed:
            self.results[phase]["passed"] += 1
        else:
            self.results[phase]["failed"] += 1
    
    def print_summary(self):
        """Print final summary"""
        self.print_header("TEST SUMMARY")
        
        total_passed = 0
        total_failed = 0
        
        for phase, data in self.results.items():
            passed = data["passed"]
            failed = data["failed"]
            total = passed + failed
            total_passed += passed
            total_failed += failed
            
            phase_name = phase.upper().replace("PHASE", "PHASE ")
            status_color = Colors.OKGREEN if failed == 0 else Colors.WARNING
            
            print(f"{status_color}{phase_name}:{Colors.ENDC} {passed}/{total} tests passed")
        
        print(f"\n{Colors.BOLD}OVERALL:{Colors.ENDC} {total_passed}/{total_passed + total_failed} tests passed")
        
        if total_failed == 0:
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! 🎉{Colors.ENDC}\n")
            return True
        else:
            print(f"\n{Colors.FAIL}{Colors.BOLD}❌ {total_failed} TEST(S) FAILED{Colors.ENDC}\n")
            return False
    
    async def test_phase1_database(self):
        """Test Phase 1: Database connections"""
        self.print_header("PHASE 1: Core Infrastructure")
        
        # Test MongoDB connection
        try:
            client = AsyncIOMotorClient(self.mongo_uri)
            await client.admin.command('ping')
            db_list = await client.list_database_names()
            await client.close()
            self.print_test("phase1", "MongoDB Connection", True, f"Found {len(db_list)} databases")
        except Exception as e:
            self.print_test("phase1", "MongoDB Connection", False, str(e))
        
        # Test Redis connection
        try:
            redis = Redis.from_url(self.redis_uri, decode_responses=True)
            await redis.ping()
            info = await redis.info()
            await redis.close()
            self.print_test("phase1", "Redis Connection", True, f"Version: {info.get('redis_version', 'unknown')}")
        except Exception as e:
            self.print_test("phase1", "Redis Connection", False, str(e))
    
    async def test_phase1_api(self):
        """Test Phase 1: Basic API endpoints"""
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            try:
                response = await client.get(f"{self.api_base}/api/health")
                if response.status_code == 200:
                    data = response.json()
                    self.print_test("phase1", "Health Endpoint", True, 
                                  f"Status: {data.get('status', 'unknown')}")
                else:
                    self.print_test("phase1", "Health Endpoint", False, 
                                  f"Status code: {response.status_code}")
            except Exception as e:
                self.print_test("phase1", "Health Endpoint", False, str(e))
            
            # Test detailed health endpoint
            try:
                response = await client.get(f"{self.api_base}/api/health/detailed")
                if response.status_code == 200:
                    data = response.json()
                    services = data.get('services', {})
                    healthy = all(s.get('status') == 'healthy' for s in services.values())
                    self.print_test("phase1", "Detailed Health Check", healthy,
                                  f"Services: {', '.join(services.keys())}")
                else:
                    self.print_test("phase1", "Detailed Health Check", False,
                                  f"Status code: {response.status_code}")
            except Exception as e:
                self.print_test("phase1", "Detailed Health Check", False, str(e))
    
    async def test_phase2_log_ingestion(self):
        """Test Phase 2: Log ingestion and processing"""
        self.print_header("PHASE 2: Log Processing")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test single log ingestion
            try:
                log_data = {
                    "message": "Test log message from validation script",
                    "level": "INFO",
                    "service": "test-service",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                response = await client.post(
                    f"{self.api_base}/api/logs/ingest",
                    json=log_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.print_test("phase2", "Single Log Ingestion", True,
                                  f"Log ID: {result.get('log_id', 'N/A')}")
                else:
                    self.print_test("phase2", "Single Log Ingestion", False,
                                  f"Status: {response.status_code}")
            except Exception as e:
                self.print_test("phase2", "Single Log Ingestion", False, str(e))
            
            # Test batch log ingestion
            try:
                batch_logs = [
                    {
                        "message": f"Batch test log {i}",
                        "level": "INFO" if i % 2 == 0 else "ERROR",
                        "service": "batch-test-service",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    for i in range(5)
                ]
                
                response = await client.post(
                    f"{self.api_base}/api/logs/batch",
                    json={"logs": batch_logs}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    count = result.get('inserted_count', 0)
                    self.print_test("phase2", "Batch Log Ingestion", count == 5,
                                  f"Inserted {count}/5 logs")
                else:
                    self.print_test("phase2", "Batch Log Ingestion", False,
                                  f"Status: {response.status_code}")
            except Exception as e:
                self.print_test("phase2", "Batch Log Ingestion", False, str(e))
            
            # Test log query
            try:
                response = await client.get(
                    f"{self.api_base}/api/logs",
                    params={"limit": 10}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logs = result.get('data', [])
                    self.print_test("phase2", "Log Query", True,
                                  f"Retrieved {len(logs)} logs")
                else:
                    self.print_test("phase2", "Log Query", False,
                                  f"Status: {response.status_code}")
            except Exception as e:
                self.print_test("phase2", "Log Query", False, str(e))
            
            # Test log search
            try:
                response = await client.get(
                    f"{self.api_base}/api/logs/search",
                    params={"query": "test", "limit": 5}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logs = result.get('data', [])
                    self.print_test("phase2", "Log Search", True,
                                  f"Found {len(logs)} matching logs")
                else:
                    self.print_test("phase2", "Log Search", False,
                                  f"Status: {response.status_code}")
            except Exception as e:
                self.print_test("phase2", "Log Search", False, str(e))
    
    async def test_phase2_models(self):
        """Test Phase 2: Data models"""
        try:
            from app.models.log_entry import LogLevel, LogEntry
            from app.models.anomaly import AnomalyType, Anomaly
            from app.models.alert import AlertStatus, AlertSeverity, Alert
            
            # Test enum imports
            levels = [LogLevel.INFO, LogLevel.ERROR]
            self.print_test("phase2", "Model Imports", True,
                          f"Successfully imported all models")
        except Exception as e:
            self.print_test("phase2", "Model Imports", False, str(e))
    
    async def test_phase3_analytics(self):
        """Test Phase 3: Analytics endpoints"""
        self.print_header("PHASE 3: Analytics & Alerts")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test log volume analytics
            try:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=24)
                
                response = await client.get(
                    f"{self.api_base}/api/analytics/volume",
                    params={
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "interval": "1h"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    data_points = len(result.get('data', []))
                    self.print_test("phase3", "Log Volume Analytics", True,
                                  f"Retrieved {data_points} data points")
                else:
                    self.print_test("phase3", "Log Volume Analytics", False,
                                  f"Status: {response.status_code}")
            except Exception as e:
                self.print_test("phase3", "Log Volume Analytics", False, str(e))
            
            # Test error rate
            try:
                response = await client.get(f"{self.api_base}/api/analytics/error-rate")
                
                if response.status_code == 200:
                    result = response.json()
                    error_rate = result.get('data', {}).get('error_rate', 0)
                    self.print_test("phase3", "Error Rate Analytics", True,
                                  f"Error rate: {error_rate}%")
                else:
                    self.print_test("phase3", "Error Rate Analytics", False,
                                  f"Status: {response.status_code}")
            except Exception as e:
                self.print_test("phase3", "Error Rate Analytics", False, str(e))
            
            # Test service metrics
            try:
                response = await client.get(f"{self.api_base}/api/analytics/services")
                
                if response.status_code == 200:
                    result = response.json()
                    services = len(result.get('data', []))
                    self.print_test("phase3", "Service Metrics", True,
                                  f"Found {services} services")
                else:
                    self.print_test("phase3", "Service Metrics", False,
                                  f"Status: {response.status_code}")
            except Exception as e:
                self.print_test("phase3", "Service Metrics", False, str(e))
            
            # Test dashboard endpoint
            try:
                response = await client.get(f"{self.api_base}/api/analytics/dashboard")
                
                if response.status_code == 200:
                    result = response.json()
                    data = result.get('data', {})
                    sections = len(data.keys())
                    self.print_test("phase3", "Dashboard Analytics", True,
                                  f"Retrieved {sections} metric sections")
                else:
                    self.print_test("phase3", "Dashboard Analytics", False,
                                  f"Status: {response.status_code}")
            except Exception as e:
                self.print_test("phase3", "Dashboard Analytics", False, str(e))
    
    async def test_phase3_alerts(self):
        """Test Phase 3: Alert system"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test create alert rule
            try:
                rule_data = {
                    "name": "Test Alert Rule",
                    "description": "Test rule for validation",
                    "condition": {
                        "metric": "error_rate",
                        "operator": "gt",
                        "threshold": 10
                    },
                    "severity": "WARNING",
                    "enabled": True,
                    "threshold": {
                        "value": 10,
                        "window_minutes": 5
                    }
                }
                
                response = await client.post(
                    f"{self.api_base}/api/alerts/rules",
                    json=rule_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    rule_id = result.get('rule_id')
                    self.print_test("phase3", "Create Alert Rule", True,
                                  f"Rule ID: {rule_id}")
                    
                    # Store for cleanup
                    self.test_rule_id = rule_id
                else:
                    self.print_test("phase3", "Create Alert Rule", False,
                                  f"Status: {response.status_code}")
            except Exception as e:
                self.print_test("phase3", "Create Alert Rule", False, str(e))
            
            # Test list alert rules
            try:
                response = await client.get(f"{self.api_base}/api/alerts/rules")
                
                if response.status_code == 200:
                    result = response.json()
                    rules = len(result.get('data', []))
                    self.print_test("phase3", "List Alert Rules", True,
                                  f"Found {rules} alert rules")
                else:
                    self.print_test("phase3", "List Alert Rules", False,
                                  f"Status: {response.status_code}")
            except Exception as e:
                self.print_test("phase3", "List Alert Rules", False, str(e))
            
            # Test alert statistics
            try:
                response = await client.get(
                    f"{self.api_base}/api/alerts/statistics/summary"
                )
                
                if response.status_code == 200:
                    result = response.json()
                    total = result.get('data', {}).get('total', 0)
                    self.print_test("phase3", "Alert Statistics", True,
                                  f"Total alerts: {total}")
                else:
                    self.print_test("phase3", "Alert Statistics", False,
                                  f"Status: {response.status_code}")
            except Exception as e:
                self.print_test("phase3", "Alert Statistics", False, str(e))
            
            # Cleanup test rule
            if hasattr(self, 'test_rule_id') and self.test_rule_id:
                try:
                    await client.delete(
                        f"{self.api_base}/api/alerts/rules/{self.test_rule_id}"
                    )
                    print(f"    {Colors.OKCYAN}→{Colors.ENDC} Cleaned up test rule")
                except:
                    pass
    
    async def test_phase3_utilities(self):
        """Test Phase 3: Utility functions"""
        try:
            from app.utils.log_parser import parse_log_line, LogFormat
            from app.utils.validators import LogValidator, ValidationError
            from app.utils.helpers import generate_hash, chunk_list
            
            # Test log parser
            json_log = '{"message": "test", "level": "INFO"}'
            parsed = parse_log_line(json_log)
            parser_works = parsed is not None and parsed.get('level') == 'INFO'
            self.print_test("phase3", "Log Parser", parser_works,
                          f"Parsed format: {parsed.get('_format', 'unknown')}")
            
            # Test validator
            try:
                level = LogValidator.validate_log_level("INFO")
                self.print_test("phase3", "Validators", True,
                              "Validation working correctly")
            except ValidationError:
                self.print_test("phase3", "Validators", False,
                              "Validation failed")
            
            # Test helpers
            hash_val = generate_hash("test data")
            chunks = chunk_list([1, 2, 3, 4, 5], 2)
            helpers_work = len(hash_val) == 64 and len(chunks) == 3
            self.print_test("phase3", "Helper Functions", helpers_work,
                          "Hash and chunk functions working")
            
        except Exception as e:
            self.print_test("phase3", "Utility Functions", False, str(e))
    
    async def test_redis_streams(self):
        """Test Redis Streams functionality"""
        try:
            from app.services.stream_service import StreamService
            from app.core.redis_client import get_redis
            
            redis = await get_redis()
            service = StreamService(redis)
            await service.initialize_stream()
            
            # Test publishing
            log_data = {
                "message": "Stream test",
                "level": "INFO",
                "timestamp": datetime.utcnow()
            }
            
            msg_id = await service.publish_log(log_data)
            stream_works = msg_id is not None
            
            await redis.close()
            
            self.print_test("phase3", "Redis Streams", stream_works,
                          f"Message ID: {msg_id}")
            
        except Exception as e:
            self.print_test("phase3", "Redis Streams", False, str(e))
    
    async def run_all_tests(self):
        """Run all validation tests"""
        print(f"\n{Colors.BOLD}{Colors.OKBLUE}")
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║                                                           ║")
        print("║       AI LOG ANALYTICS - PHASE VALIDATION SUITE          ║")
        print("║                                                           ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print(Colors.ENDC)
        
        try:
            # Phase 1 tests
            await self.test_phase1_database()
            await self.test_phase1_api()
            
            # Phase 2 tests
            await self.test_phase2_log_ingestion()
            await self.test_phase2_models()
            
            # Phase 3 tests
            await self.test_phase3_analytics()
            await self.test_phase3_alerts()
            await self.test_phase3_utilities()
            await self.test_redis_streams()
            
            # Print summary
            all_passed = self.print_summary()
            
            return all_passed
            
        except Exception as e:
            print(f"\n{Colors.FAIL}Critical error during testing: {e}{Colors.ENDC}\n")
            return False


async def main():
    """Main test runner"""
    validator = PhaseValidator()
    
    print(f"{Colors.WARNING}⚠️  Make sure all services are running:{Colors.ENDC}")
    print(f"  • docker-compose up -d")
    print(f"  • Backend API on port 8000")
    print(f"  • MongoDB on port 27017")
    print(f"  • Redis on port 6379\n")
    
    input(f"{Colors.OKCYAN}Press Enter to start tests...{Colors.ENDC}")
    
    success = await validator.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())