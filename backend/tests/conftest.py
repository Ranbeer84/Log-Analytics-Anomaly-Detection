"""
Pytest Configuration and Fixtures
Shared test fixtures for the entire test suite
"""
import sys
import asyncio
from pathlib import Path
from typing import Generator, AsyncGenerator
from datetime import datetime, timedelta
import pytest
from faker import Faker
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.config import settings
from app.core.database import get_database
from app.core.redis_client import get_redis

fake = Faker()

# ===== Event Loop Configuration =====

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ===== Test Database Configuration =====

@pytest.fixture(scope="session")
async def test_mongodb():
    """
    Test MongoDB connection
    Creates a separate test database
    """
    test_db_name = f"{settings.MONGODB_DB_NAME}_test"
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[test_db_name]
    
    yield db
    
    # Cleanup: Drop test database after all tests
    await client.drop_database(test_db_name)
    client.close()


@pytest.fixture(scope="session")
async def test_redis():
    """
    Test Redis connection
    Uses a separate Redis database (default is 0, test uses 1)
    """
    test_redis_url = f"{settings.REDIS_URL}/1"  # Use database 1 for tests
    client = redis.from_url(test_redis_url, decode_responses=True)
    
    yield client
    
    # Cleanup: Flush test database after all tests
    await client.flushdb()
    await client.close()


# ===== FastAPI Test Client =====

@pytest.fixture
def client() -> Generator:
    """
    Synchronous test client for FastAPI
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client() -> AsyncGenerator:
    """
    Asynchronous test client for FastAPI
    """
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# ===== Mock Data Generators =====

@pytest.fixture
def sample_log_entry():
    """Generate a sample log entry"""
    return {
        "timestamp": datetime.utcnow(),
        "level": fake.random_element(["INFO", "WARN", "ERROR", "DEBUG"]),
        "service": fake.random_element([
            "api-gateway", "auth-service", "user-service",
            "payment-service", "notification-service"
        ]),
        "message": fake.sentence(),
        "endpoint": f"/api/v1/{fake.word()}",
        "status_code": fake.random_element([200, 201, 400, 401, 500]),
        "response_time": fake.random.uniform(50, 500),
        "ip_address": fake.ipv4(),
        "user_id": f"user_{fake.random_number(digits=4)}",
        "metadata": {
            "environment": "test",
            "version": fake.random_element(["v1.0.0", "v1.1.0", "v2.0.0"]),
            "region": fake.random_element(["us-east-1", "us-west-2", "eu-west-1"])
        }
    }


@pytest.fixture
def sample_log_batch(sample_log_entry):
    """Generate a batch of sample log entries"""
    def _generate_batch(count: int = 10):
        return [
            {
                **sample_log_entry,
                "timestamp": datetime.utcnow() - timedelta(minutes=i),
                "message": fake.sentence()
            }
            for i in range(count)
        ]
    return _generate_batch


@pytest.fixture
def sample_anomaly_log():
    """Generate a sample anomalous log entry"""
    return {
        "timestamp": datetime.utcnow(),
        "level": "ERROR",
        "service": "payment-service",
        "message": "Connection timeout to database",
        "endpoint": "/api/v1/payments/process",
        "status_code": 500,
        "response_time": 15000,  # Very high response time
        "ip_address": fake.ipv4(),
        "user_id": f"user_{fake.random_number(digits=4)}",
        "metadata": {
            "error_type": "timeout",
            "retry_count": 3
        }
    }


@pytest.fixture
def sample_alert():
    """Generate a sample alert"""
    return {
        "alert_type": "high_error_rate",
        "severity": "critical",
        "service": "payment-service",
        "message": "High error rate detected",
        "threshold": 0.05,
        "current_value": 0.15,
        "created_at": datetime.utcnow(),
        "status": "open",
        "metadata": {
            "time_window": "5m",
            "affected_endpoints": ["/api/v1/payments/process"]
        }
    }


# ===== Database Cleanup Fixtures =====

@pytest.fixture(autouse=True)
async def cleanup_test_data(test_mongodb):
    """
    Automatically cleanup test data after each test
    """
    yield
    
    # Clean up all collections
    collections = await test_mongodb.list_collection_names()
    for collection in collections:
        await test_mongodb[collection].delete_many({})


# ===== Mock External Services =====

@pytest.fixture
def mock_ml_service(monkeypatch):
    """
    Mock ML service responses
    """
    async def mock_detect_anomaly(*args, **kwargs):
        return {
            "is_anomaly": False,
            "anomaly_score": 0.3,
            "confidence": 0.85,
            "severity": "low"
        }
    
    async def mock_train_model(*args, **kwargs):
        return {
            "status": "success",
            "model_id": "test_model_v1",
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.93
            }
        }
    
    return {
        "detect_anomaly": mock_detect_anomaly,
        "train_model": mock_train_model
    }


@pytest.fixture
def mock_email_service(monkeypatch):
    """
    Mock email service
    """
    sent_emails = []
    
    async def mock_send_email(to: str, subject: str, body: str):
        sent_emails.append({
            "to": to,
            "subject": subject,
            "body": body,
            "sent_at": datetime.utcnow()
        })
        return True
    
    return {
        "send_email": mock_send_email,
        "sent_emails": sent_emails
    }


# ===== Authentication Fixtures =====

@pytest.fixture
def auth_headers():
    """
    Generate authentication headers for testing
    """
    # In a real app, generate a valid JWT token
    return {
        "Authorization": "Bearer test_token_12345"
    }


@pytest.fixture
def admin_headers():
    """
    Generate admin authentication headers for testing
    """
    return {
        "Authorization": "Bearer admin_token_12345",
        "X-Admin": "true"
    }


# ===== Time-based Fixtures =====

@pytest.fixture
def fixed_datetime(monkeypatch):
    """
    Fix datetime to a specific value for testing
    """
    fixed_time = datetime(2025, 2, 12, 12, 0, 0)
    
    class MockDateTime:
        @classmethod
        def utcnow(cls):
            return fixed_time
        
        @classmethod
        def now(cls):
            return fixed_time
    
    monkeypatch.setattr("datetime.datetime", MockDateTime)
    return fixed_time


# ===== Helper Functions =====

@pytest.fixture
def create_test_logs():
    """
    Helper function to create test logs in database
    """
    async def _create_logs(db, count: int = 10, **kwargs):
        logs = []
        for i in range(count):
            log = {
                "timestamp": datetime.utcnow() - timedelta(minutes=i),
                "level": "INFO",
                "service": "test-service",
                "message": f"Test log message {i}",
                "processed": False,
                **kwargs
            }
            logs.append(log)
        
        result = await db.logs.insert_many(logs)
        return result.inserted_ids
    
    return _create_logs


@pytest.fixture
def assert_log_structure():
    """
    Helper function to assert log entry structure
    """
    def _assert(log_entry: dict):
        required_fields = [
            "timestamp", "level", "service", "message"
        ]
        for field in required_fields:
            assert field in log_entry, f"Missing required field: {field}"
        
        assert log_entry["level"] in ["INFO", "WARN", "ERROR", "DEBUG", "CRITICAL"]
        assert isinstance(log_entry["timestamp"], (datetime, str))
    
    return _assert


# ===== Performance Testing Fixtures =====

@pytest.fixture
def benchmark_timer():
    """
    Simple timer for benchmarking operations
    """
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed()
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# ===== Pytest Configuration =====

def pytest_configure(config):
    """
    Pytest configuration hook
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests related to ML models"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers based on test location
    """
    for item in items:
        # Add markers based on test file location
        if "test_api" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_services" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_ml" in str(item.fspath):
            item.add_marker(pytest.mark.ml)