# API Documentation

Complete REST API reference for the AI Log Analytics Platform.

## Base URL

```
Development: http://localhost:8000
Production: https://your-domain.com
```

## Authentication

Most endpoints require authentication using JWT tokens.

```bash
# Get access token (if implemented)
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "your-username",
  "password": "your-password"
}

# Use token in requests
Authorization: Bearer <your-token>
```

---

## Endpoints

### Health & Status

#### GET /api/v1/health

Check API health status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-02-12T14:30:00Z",
  "version": "1.0.0",
  "services": {
    "mongodb": "healthy",
    "redis": "healthy",
    "ml_service": "healthy"
  }
}
```

#### GET /api/v1/health/metrics

Get system metrics.

**Response**:
```json
{
  "logs_per_second": 156.4,
  "total_logs": 1523456,
  "anomaly_rate": 0.032,
  "avg_response_time": 145.2,
  "active_alerts": 3
}
```

---

### Log Management

#### POST /api/v1/logs/ingest

Ingest a single log entry.

**Request Body**:
```json
{
  "timestamp": "2025-02-12T14:30:00Z",
  "level": "ERROR",
  "service": "payment-service",
  "message": "Payment processing failed",
  "endpoint": "/api/v1/payments/process",
  "status_code": 500,
  "response_time": 5432.1,
  "ip_address": "192.168.1.100",
  "user_id": "user_12345",
  "metadata": {
    "error_code": "TIMEOUT",
    "retry_count": 3,
    "transaction_id": "txn_abc123"
  }
}
```

**Response**:
```json
{
  "status": "success",
  "log_id": "65ca123...",
  "message": "Log ingested successfully",
  "anomaly_detected": false
}
```

**Status Codes**:
- `201`: Log created successfully
- `400`: Invalid request body
- `500`: Server error

#### POST /api/v1/logs/bulk

Ingest multiple logs in a single request.

**Request Body**:
```json
{
  "logs": [
    {
      "timestamp": "2025-02-12T14:30:00Z",
      "level": "INFO",
      "service": "api-gateway",
      "message": "Request processed"
    },
    {
      "timestamp": "2025-02-12T14:30:01Z",
      "level": "ERROR",
      "service": "payment-service",
      "message": "Payment failed"
    }
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "ingested_count": 2,
  "failed_count": 0,
  "log_ids": ["65ca123...", "65ca124..."]
}
```

#### GET /api/v1/logs

Retrieve logs with filtering and pagination.

**Query Parameters**:
- `level` (string): Filter by log level (INFO, WARN, ERROR, etc.)
- `service` (string): Filter by service name
- `start_date` (ISO 8601): Start of time range
- `end_date` (ISO 8601): End of time range
- `limit` (integer): Number of logs to return (default: 100)
- `skip` (integer): Number of logs to skip (pagination)
- `sort` (string): Sort field (default: timestamp)
- `order` (string): Sort order (asc/desc, default: desc)

**Example Request**:
```bash
GET /api/v1/logs?level=ERROR&service=payment-service&limit=50&sort=timestamp&order=desc
```

**Response**:
```json
{
  "logs": [
    {
      "_id": "65ca123...",
      "timestamp": "2025-02-12T14:30:00Z",
      "level": "ERROR",
      "service": "payment-service",
      "message": "Payment processing failed",
      "endpoint": "/api/v1/payments/process",
      "status_code": 500,
      "response_time": 5432.1,
      "metadata": {...}
    }
  ],
  "total": 234,
  "limit": 50,
  "skip": 0,
  "has_more": true
}
```

#### GET /api/v1/logs/{log_id}

Get a specific log entry by ID.

**Response**:
```json
{
  "_id": "65ca123...",
  "timestamp": "2025-02-12T14:30:00Z",
  "level": "ERROR",
  "service": "payment-service",
  "message": "Payment processing failed",
  "metadata": {...},
  "anomaly_score": 0.87,
  "is_anomaly": true
}
```

**Status Codes**:
- `200`: Success
- `404`: Log not found

#### POST /api/v1/logs/search

Advanced log search with full-text and filters.

**Request Body**:
```json
{
  "query": "payment failed",
  "filters": {
    "level": ["ERROR", "CRITICAL"],
    "service": ["payment-service", "checkout-service"],
    "status_code": [500, 502, 503]
  },
  "start_date": "2025-02-01T00:00:00Z",
  "end_date": "2025-02-12T23:59:59Z",
  "limit": 100
}
```

**Response**:
```json
{
  "results": [...],
  "total": 156,
  "query": "payment failed",
  "execution_time_ms": 45.2
}
```

---

### Analytics

#### GET /api/v1/analytics/overview

Get overall analytics summary.

**Query Parameters**:
- `time_range` (string): Time range (1h, 6h, 24h, 7d, 30d)

**Response**:
```json
{
  "time_range": "24h",
  "total_logs": 145623,
  "logs_by_level": {
    "INFO": 123456,
    "WARN": 18234,
    "ERROR": 3456,
    "CRITICAL": 477
  },
  "logs_by_service": {
    "api-gateway": 45678,
    "payment-service": 34567,
    "user-service": 23456
  },
  "error_rate": 0.027,
  "anomaly_count": 456,
  "anomaly_rate": 0.0031,
  "avg_response_time": 234.5,
  "active_alerts": 3
}
```

#### GET /api/v1/analytics/error-rate

Get error rate over time.

**Query Parameters**:
- `time_range` (string): Time range
- `interval` (string): Aggregation interval (1m, 5m, 1h, 1d)

**Response**:
```json
{
  "time_range": "24h",
  "interval": "1h",
  "data": [
    {
      "timestamp": "2025-02-12T00:00:00Z",
      "total_logs": 6234,
      "error_logs": 187,
      "error_rate": 0.030
    },
    {
      "timestamp": "2025-02-12T01:00:00Z",
      "total_logs": 5987,
      "error_logs": 145,
      "error_rate": 0.024
    }
  ]
}
```

#### GET /api/v1/analytics/response-time

Get response time statistics.

**Response**:
```json
{
  "time_range": "24h",
  "data": [
    {
      "timestamp": "2025-02-12T00:00:00Z",
      "avg": 234.5,
      "p50": 189.3,
      "p95": 567.8,
      "p99": 1234.5,
      "max": 3456.7
    }
  ]
}
```

#### GET /api/v1/analytics/log-volume

Get log volume over time.

**Response**:
```json
{
  "time_range": "24h",
  "interval": "1h",
  "data": [
    {
      "timestamp": "2025-02-12T00:00:00Z",
      "count": 6234,
      "logs_per_second": 1.73
    }
  ]
}
```

#### GET /api/v1/analytics/top-errors

Get most frequent errors.

**Query Parameters**:
- `limit` (integer): Number of top errors (default: 10)
- `time_range` (string): Time range

**Response**:
```json
{
  "time_range": "24h",
  "top_errors": [
    {
      "message": "Connection timeout to database",
      "count": 234,
      "services": ["payment-service", "order-service"],
      "first_seen": "2025-02-12T08:15:00Z",
      "last_seen": "2025-02-12T14:30:00Z"
    },
    {
      "message": "Invalid authentication token",
      "count": 187,
      "services": ["api-gateway"],
      "first_seen": "2025-02-12T09:00:00Z",
      "last_seen": "2025-02-12T14:25:00Z"
    }
  ]
}
```

#### GET /api/v1/analytics/anomalies

Get detected anomalies.

**Query Parameters**:
- `time_range` (string): Time range
- `severity` (string): Filter by severity (low, medium, high, critical)
- `limit` (integer): Number of results

**Response**:
```json
{
  "time_range": "6h",
  "anomalies": [
    {
      "_id": "65ca789...",
      "log_id": "65ca123...",
      "timestamp": "2025-02-12T14:15:00Z",
      "service": "payment-service",
      "anomaly_score": 0.92,
      "severity": "critical",
      "confidence": 0.88,
      "reasons": [
        "Unusual response time pattern",
        "High error rate spike"
      ],
      "log_entry": {...}
    }
  ],
  "total": 23,
  "severity_distribution": {
    "low": 8,
    "medium": 10,
    "high": 3,
    "critical": 2
  }
}
```

---

### Alerts

#### GET /api/v1/alerts

Get all alerts.

**Query Parameters**:
- `status` (string): Filter by status (open, acknowledged, resolved, closed)
- `severity` (string): Filter by severity
- `service` (string): Filter by service
- `limit` (integer): Number of results

**Response**:
```json
{
  "alerts": [
    {
      "_id": "65ca456...",
      "alert_type": "high_error_rate",
      "severity": "critical",
      "service": "payment-service",
      "message": "Error rate exceeded threshold",
      "threshold": 0.05,
      "current_value": 0.087,
      "status": "open",
      "created_at": "2025-02-12T14:15:00Z",
      "updated_at": "2025-02-12T14:15:00Z",
      "metadata": {
        "time_window": "5m",
        "affected_endpoints": ["/api/v1/payments/process"]
      }
    }
  ],
  "total": 15,
  "summary": {
    "open": 5,
    "acknowledged": 7,
    "resolved": 3
  }
}
```

#### GET /api/v1/alerts/{alert_id}

Get specific alert details.

**Response**:
```json
{
  "_id": "65ca456...",
  "alert_type": "high_error_rate",
  "severity": "critical",
  "service": "payment-service",
  "message": "Error rate exceeded threshold",
  "status": "open",
  "created_at": "2025-02-12T14:15:00Z",
  "related_logs": [...],
  "timeline": [
    {
      "timestamp": "2025-02-12T14:15:00Z",
      "event": "alert_created",
      "user": "system"
    }
  ]
}
```

#### POST /api/v1/alerts

Create a new alert rule.

**Request Body**:
```json
{
  "name": "High Error Rate - Payment Service",
  "description": "Alert when error rate exceeds 5%",
  "condition": {
    "metric": "error_rate",
    "operator": ">",
    "threshold": 0.05,
    "time_window": "5m"
  },
  "filters": {
    "service": ["payment-service"]
  },
  "severity": "critical",
  "channels": ["email", "slack"],
  "enabled": true
}
```

**Response**:
```json
{
  "status": "success",
  "alert_rule_id": "65ca789...",
  "message": "Alert rule created successfully"
}
```

#### PUT /api/v1/alerts/{alert_id}

Update an existing alert rule.

#### DELETE /api/v1/alerts/{alert_id}

Delete an alert rule.

#### POST /api/v1/alerts/{alert_id}/acknowledge

Acknowledge an alert.

**Request Body**:
```json
{
  "acknowledged_by": "user@example.com",
  "notes": "Investigating the issue"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Alert acknowledged"
}
```

#### POST /api/v1/alerts/{alert_id}/resolve

Resolve an alert.

**Request Body**:
```json
{
  "resolved_by": "user@example.com",
  "resolution_notes": "Issue fixed by deploying hotfix v1.2.3"
}
```

---

## WebSocket API

### Real-time Log Stream

Connect to receive real-time log updates.

**Endpoint**: `ws://localhost:8000/ws/logs`

**Message Format**:
```json
{
  "type": "log",
  "data": {
    "timestamp": "2025-02-12T14:30:00Z",
    "level": "ERROR",
    "service": "payment-service",
    "message": "Payment failed"
  }
}
```

**Example (JavaScript)**:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/logs');

ws.onmessage = (event) => {
  const log = JSON.parse(event.data);
  console.log('New log:', log);
};
```

### Real-time Anomaly Stream

**Endpoint**: `ws://localhost:8000/ws/anomalies`

**Message Format**:
```json
{
  "type": "anomaly",
  "data": {
    "log_id": "65ca123...",
    "anomaly_score": 0.92,
    "severity": "critical"
  }
}
```

---

## Rate Limiting

API requests are rate-limited to prevent abuse:

- **Standard endpoints**: 100 requests/minute per IP
- **Log ingestion**: 1000 requests/minute per IP
- **WebSocket connections**: 10 concurrent connections per IP

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1707746400
```

---

## Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid log level specified",
    "details": {
      "field": "level",
      "valid_values": ["INFO", "WARN", "ERROR", "DEBUG", "CRITICAL"]
    }
  }
}
```

**Common Error Codes**:
- `INVALID_REQUEST`: Invalid request parameters
- `UNAUTHORIZED`: Authentication required
- `FORBIDDEN`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `RATE_LIMITED`: Too many requests
- `INTERNAL_ERROR`: Server error

---

## SDKs and Client Libraries

### Python

```python
from log_analytics_client import LogAnalyticsClient

client = LogAnalyticsClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Ingest log
client.logs.ingest({
    "level": "ERROR",
    "service": "my-service",
    "message": "Something went wrong"
})

# Query logs
logs = client.logs.query(
    level="ERROR",
    service="my-service",
    limit=100
)
```

### JavaScript/TypeScript

```typescript
import { LogAnalyticsClient } from '@loganalytics/client';

const client = new LogAnalyticsClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Ingest log
await client.logs.ingest({
  level: 'ERROR',
  service: 'my-service',
  message: 'Something went wrong'
});

// Query logs
const logs = await client.logs.query({
  level: 'ERROR',
  service: 'my-service',
  limit: 100
});
```

---

## API Changelog

### v1.0.0 (2025-02-12)
- Initial release
- Log ingestion and query endpoints
- Analytics endpoints
- Alert management
- WebSocket support