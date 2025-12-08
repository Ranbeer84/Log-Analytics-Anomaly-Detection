"""
Synthetic log generator for testing
"""
import asyncio
import sys
import os
import random
from datetime import datetime, timedelta
import httpx

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample data
SERVICES = [
    "api-gateway", "auth-service", "user-service", "payment-service",
    "notification-service", "analytics-service", "database-service", "cache-service"
]

LOG_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]

MESSAGES = {
    "INFO": [
        "Request processed successfully",
        "User logged in",
        "Transaction completed",
        "Cache hit for key",
        "Data synchronized",
        "Email sent successfully",
        "API call completed",
        "Resource created",
    ],
    "WARN": [
        "High response time detected",
        "Cache miss for key",
        "Retry attempt",
        "Deprecated API used",
        "Low memory warning",
        "Connection pool nearly full",
        "Rate limit approaching",
        "Slow query detected",
    ],
    "ERROR": [
        "Failed to connect to database",
        "Authentication failed",
        "Payment processing failed",
        "Invalid input data",
        "Service unavailable",
        "Timeout error",
        "Failed to send email",
        "Database query failed",
    ],
    "DEBUG": [
        "Processing request",
        "Loading configuration",
        "Initializing connection",
        "Validating input",
        "Executing query",
        "Checking cache",
        "Building response",
        "Cleaning up resources",
    ]
}

ENDPOINTS = [
    "/api/login", "/api/logout", "/api/users", "/api/products",
    "/api/orders", "/api/payments", "/api/analytics", "/api/settings"
]

STATUS_CODES = {
    "INFO": [200, 201, 204],
    "WARN": [429, 408],
    "ERROR": [400, 401, 403, 404, 500, 502, 503],
    "DEBUG": [200]
}


def generate_log(timestamp: datetime) -> dict:
    """Generate a single random log entry"""
    
    level = random.choices(
        LOG_LEVELS,
        weights=[70, 15, 10, 5]  # More INFO, fewer ERRORs
    )[0]
    
    service = random.choice(SERVICES)
    message = random.choice(MESSAGES[level])
    endpoint = random.choice(ENDPOINTS)
    
    log = {
        "timestamp": timestamp.isoformat() + "Z",
        "level": level,
        "service": service,
        "message": message,
        "endpoint": endpoint,
        "response_time": round(random.uniform(10, 500), 2),
        "status_code": random.choice(STATUS_CODES[level]),
        "user_id": f"user_{random.randint(1, 1000)}",
        "ip_address": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
        "metadata": {
            "request_id": f"req_{random.randint(10000, 99999)}",
            "trace_id": f"trace_{random.randint(100000, 999999)}"
        }
    }
    
    return log


def generate_anomaly_log(timestamp: datetime) -> dict:
    """Generate an anomalous log entry"""
    
    service = random.choice(SERVICES)
    
    # Create anomalous patterns
    anomaly_type = random.choice(["error_spike", "slow_response", "auth_failure"])
    
    if anomaly_type == "error_spike":
        log = {
            "timestamp": timestamp.isoformat() + "Z",
            "level": "ERROR",
            "service": service,
            "message": "Critical system failure",
            "endpoint": random.choice(ENDPOINTS),
            "response_time": round(random.uniform(1000, 3000), 2),  # Very slow
            "status_code": 500,
            "user_id": f"user_{random.randint(1, 1000)}",
            "ip_address": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "metadata": {
                "error_code": "CRITICAL_ERROR",
                "request_id": f"req_{random.randint(10000, 99999)}"
            }
        }
    elif anomaly_type == "slow_response":
        log = {
            "timestamp": timestamp.isoformat() + "Z",
            "level": "WARN",
            "service": service,
            "message": "Extremely slow response time",
            "endpoint": random.choice(ENDPOINTS),
            "response_time": round(random.uniform(2000, 5000), 2),  # Very slow
            "status_code": 200,
            "user_id": f"user_{random.randint(1, 1000)}",
            "ip_address": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "metadata": {
                "performance_issue": True,
                "request_id": f"req_{random.randint(10000, 99999)}"
            }
        }
    else:  # auth_failure
        log = {
            "timestamp": timestamp.isoformat() + "Z",
            "level": "ERROR",
            "service": "auth-service",
            "message": "Multiple failed login attempts",
            "endpoint": "/api/login",
            "response_time": round(random.uniform(50, 200), 2),
            "status_code": 401,
            "user_id": f"user_{random.randint(1, 100)}",
            "ip_address": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "metadata": {
                "attempt_count": random.randint(5, 20),
                "potential_attack": True,
                "request_id": f"req_{random.randint(10000, 99999)}"
            }
        }
    
    return log


async def send_log(log: dict, api_url: str):
    """Send log to API"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{api_url}/api/v1/logs", json=log, timeout=10.0)
            return response.status_code == 201
        except Exception as e:
            logger.error(f"Error sending log: {e}")
            return False


async def generate_logs_batch(num_logs: int = 1000, include_anomalies: bool = True):
    """Generate and send a batch of logs"""
    
    api_url = "http://localhost:8000"
    
    logger.info(f"🔧 Generating {num_logs} logs...")
    
    # Generate time range (last 7 days)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    
    success_count = 0
    fail_count = 0
    
    for i in range(num_logs):
        # Random timestamp within range
        random_seconds = random.randint(0, int((end_time - start_time).total_seconds()))
        timestamp = start_time + timedelta(seconds=random_seconds)
        
        # 5% chance of anomaly if enabled
        if include_anomalies and random.random() < 0.05:
            log = generate_anomaly_log(timestamp)
        else:
            log = generate_log(timestamp)
        
        # Send log
        success = await send_log(log, api_url)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i + 1}/{num_logs} logs sent")
    
    logger.info(f"\n✅ Log generation complete!")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed: {fail_count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic logs")
    parser.add_argument("--count", type=int, default=1000, help="Number of logs to generate")
    parser.add_argument("--no-anomalies", action="store_true", help="Disable anomaly generation")
    
    args = parser.parse_args()
    
    asyncio.run(generate_logs_batch(
        num_logs=args.count,
        include_anomalies=not args.no_anomalies
    ))