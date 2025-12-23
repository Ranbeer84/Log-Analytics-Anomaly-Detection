# AI-Powered Log Analytics & Anomaly Detection Platform

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-teal.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

**A production-ready, scalable platform for real-time log analytics and AI-powered anomaly detection**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [API Reference](#-api-reference)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [ML Models](#-ml-models)
- [Monitoring](#-monitoring)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The AI-Powered Log Analytics Platform is a comprehensive solution for ingesting, processing, analyzing, and detecting anomalies in application logs at scale. Built with modern technologies and best practices, it provides real-time insights into system behavior and automatically identifies potential issues before they become critical.

### Key Capabilities

- **Real-time Log Ingestion**: High-throughput log ingestion with Redis Streams
- **AI-Powered Anomaly Detection**: Machine learning models for automatic anomaly detection
- **Interactive Dashboard**: Real-time visualization of logs and metrics
- **Intelligent Alerting**: Configurable alerts with multiple notification channels
- **Scalable Architecture**: Microservices design for horizontal scalability
- **Production-Ready**: Comprehensive monitoring, logging, and error handling

---

## ✨ Features

### Core Features

#### 🔍 Log Management
- **Multi-source Log Ingestion**: REST API, bulk uploads, and streaming
- **Structured Logging**: JSON-based log format with flexible metadata
- **Advanced Search**: Full-text search with filters and date ranges
- **Log Aggregation**: Group and aggregate logs by service, level, and time

#### 🤖 AI/ML Capabilities
- **Anomaly Detection**: Isolation Forest, Autoencoders, and LSTM models
- **Pattern Recognition**: Identify recurring patterns and trends
- **Predictive Analytics**: Forecast potential issues before they occur
- **Continuous Learning**: Models retrain automatically on new data

#### 📊 Analytics & Visualization
- **Real-time Dashboards**: Live log streams and metrics
- **Custom Charts**: Error rates, response times, log volumes
- **Heatmaps**: Visualize anomaly patterns over time
- **Historical Analysis**: Query and analyze historical log data

#### 🚨 Alerting System
- **Configurable Rules**: Set custom thresholds and conditions
- **Multi-channel Notifications**: Email, Slack, webhooks
- **Alert Correlation**: Group related alerts to reduce noise
- **Escalation Policies**: Automatic escalation for critical issues

#### 📈 Monitoring & Observability
- **Prometheus Metrics**: Export metrics for monitoring
- **Grafana Dashboards**: Pre-built dashboards for system health
- **Distributed Tracing**: Track requests across services
- **Health Checks**: Automated system health monitoring

---

## 🏗 Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│                    Interactive Dashboard & UI                    │
└────────────────────────────────┬────────────────────────────────┘
                                 │ HTTP/WebSocket
┌────────────────────────────────┴────────────────────────────────┐
│                      Nginx (Reverse Proxy)                       │
│              Load Balancing & SSL Termination                    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────┴────────┐    ┌──────────┴─────────┐    ┌────────┴────────┐
│   FastAPI      │    │   ML Service       │    │   Workers       │
│   Backend      │    │   (Inference)      │    │   (Consumers)   │
│                │    │                    │    │                 │
│ • REST API     │    │ • Anomaly          │    │ • Log Consumer  │
│ • WebSocket    │    │   Detection        │    │ • Anomaly       │
│ • Validation   │    │ • Model Training   │    │   Worker        │
└────────┬───────┘    └─────────┬──────────┘    └────────┬────────┘
         │                      │                         │
         │ ┌────────────────────┴─────────────────────────┤
         │ │                                              │
    ┌────┴─┴──────┐                            ┌─────────┴────────┐
    │   MongoDB    │                            │  Redis Streams   │
    │              │                            │                  │
    │ • Logs       │                            │ • Message Queue  │
    │ • Anomalies  │                            │ • Real-time      │
    │ • Alerts     │                            │   Processing     │
    └──────────────┘                            └──────────────────┘
```

### Data Flow

```
Log Sources → API Gateway → Redis Stream → Workers → MongoDB → Analytics
                                                  ↓
                                            ML Service
                                                  ↓
                                         Anomaly Detection
                                                  ↓
                                          Alert Service
```

### Components

#### Frontend
- **Technology**: React 18+ with Vite
- **UI Library**: Tailwind CSS
- **Charts**: Recharts, Chart.js
- **State Management**: React Hooks
- **Real-time**: WebSocket connections

#### Backend API
- **Framework**: FastAPI (Python 3.11+)
- **Database**: MongoDB (Motor async driver)
- **Cache**: Redis
- **Authentication**: JWT tokens
- **Validation**: Pydantic schemas

#### ML Service
- **Framework**: scikit-learn, PyTorch
- **Models**: Isolation Forest, Autoencoders, LSTM
- **Feature Engineering**: Custom extractors
- **Model Storage**: Pickle, CloudPickle

#### Workers
- **Log Consumer**: Processes incoming logs from Redis Streams
- **Anomaly Worker**: Runs ML inference on logs
- **Alert Worker**: Manages alert generation and delivery

---

## 🛠 Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework
- **Python 3.11+**: Latest Python features
- **Motor**: Async MongoDB driver
- **Redis**: In-memory data store
- **Pydantic**: Data validation

### ML/AI
- **scikit-learn**: Traditional ML algorithms
- **PyTorch**: Deep learning framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Database
- **MongoDB**: Document database for logs
- **Redis**: Stream processing and caching

### DevOps
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Reverse proxy
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization

### Frontend
- **React 18**: UI library
- **Vite**: Build tool
- **Tailwind CSS**: Utility-first CSS
- **Recharts**: Charting library

---

## 📦 Prerequisites

### Required Software
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Python**: 3.11+ (for local development)
- **Node.js**: 18+ (for frontend development)
- **Git**: For version control

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **CPU**: 4+ cores recommended
- **Disk**: 20GB+ free space
- **OS**: Linux, macOS, or Windows with WSL2

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-log-analytics.git
cd ai-log-analytics
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 3. Start the Platform

```bash
# Make deploy script executable
chmod +x scripts/deploy.sh

# Start all services
./scripts/deploy.sh start

# This will:
# - Build Docker images
# - Start all services
# - Initialize databases
# - (Optional) Seed sample data
# - (Optional) Train ML models
```

### 4. Access the Platform

- **Frontend Dashboard**: http://localhost:80
- **API Documentation**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### 5. Ingest Your First Log

```bash
curl -X POST http://localhost:8000/api/v1/logs/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-02-12T14:30:00Z",
    "level": "INFO",
    "service": "api-gateway",
    "message": "User logged in successfully",
    "metadata": {
      "user_id": "user_123",
      "ip_address": "192.168.1.1"
    }
  }'
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Application
APP_NAME="AI Log Analytics"
ENV="production"  # development | staging | production
DEBUG=false
LOG_LEVEL="INFO"

# Database
MONGODB_URL="mongodb://admin:password@mongodb:27017/"
MONGODB_DB_NAME="log_analytics"

# Redis
REDIS_URL="redis://redis:6379"
REDIS_STREAM_NAME="log_stream"

# ML Service
ML_SERVICE_URL="http://ml_service:8001"
ANOMALY_THRESHOLD=0.8

# Alerts
SMTP_HOST="smtp.gmail.com"
SMTP_PORT=587
SMTP_USERNAME="your-email@gmail.com"
SMTP_PASSWORD="your-app-password"
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Security
SECRET_KEY="your-secret-key-min-32-characters"
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:80"]
```

### Docker Compose Profiles

```bash
# Development mode (with hot reload)
docker-compose up

# Production mode (with Nginx)
docker-compose --profile production up

# Scale workers
docker-compose up --scale log_consumer=3 --scale anomaly_worker=2
```

---

## 📚 API Documentation

### Interactive API Docs

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

### Main Endpoints

#### Log Ingestion

```bash
# Single log ingestion
POST /api/v1/logs/ingest
Content-Type: application/json

{
  "timestamp": "2025-02-12T14:30:00Z",
  "level": "ERROR",
  "service": "payment-service",
  "message": "Payment processing failed",
  "metadata": {}
}

# Bulk log ingestion
POST /api/v1/logs/bulk
Content-Type: application/json

{
  "logs": [
    {...},
    {...}
  ]
}
```

#### Log Query

```bash
# Get logs with filters
GET /api/v1/logs?level=ERROR&service=payment-service&limit=100

# Search logs
POST /api/v1/logs/search
{
  "query": "payment failed",
  "start_date": "2025-02-01T00:00:00Z",
  "end_date": "2025-02-12T23:59:59Z"
}
```

#### Analytics

```bash
# Get analytics overview
GET /api/v1/analytics/overview?time_range=24h

# Get error rate
GET /api/v1/analytics/error-rate?time_range=1h

# Get anomalies
GET /api/v1/analytics/anomalies?time_range=6h
```

#### Alerts

```bash
# Create alert rule
POST /api/v1/alerts
{
  "name": "High Error Rate",
  "condition": "error_rate > 0.05",
  "severity": "critical",
  "channels": ["email", "slack"]
}

# Get active alerts
GET /api/v1/alerts?status=open
```

---

## 🤖 ML Models

### Isolation Forest

**Use Case**: General anomaly detection
**Training**: Unsupervised learning on normal log patterns
**Inference**: Real-time anomaly scoring

```bash
# Train model
docker-compose exec ml_service python -m ml.training.train_isolation_forest \
  --contamination 0.1 \
  --n-estimators 100
```

### Features Extracted

- Log level distribution
- Response time statistics
- Error rate patterns
- Service behavior patterns
- Temporal patterns
- Text features (TF-IDF)

---

## 📊 Monitoring

### Prometheus Metrics

```bash
# View metrics
curl http://localhost:8000/metrics

# Key metrics:
# - http_requests_total
# - http_request_duration_seconds
# - logs_ingested_total
# - anomalies_detected_total
# - redis_stream_length
```

### Grafana Dashboards

Pre-configured dashboards available at `http://localhost:3000`:

1. **System Overview**: Overall system health
2. **API Performance**: Request rates, latencies, error rates
3. **Log Analytics**: Ingestion rates, log volumes
4. **ML Metrics**: Anomaly detection performance
5. **Infrastructure**: CPU, memory, disk usage

### Alert Rules

Alerts configured in `monitoring/alerts/alert_rules.yml`:

- High API error rate
- API service down
- High anomaly detection rate
- Database connection issues
- High resource usage

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
docker-compose exec backend pytest

# Run with coverage
docker-compose exec backend pytest --cov=app --cov-report=html

# Run specific test types
pytest -m unit  # Unit tests only
pytest -m integration  # Integration tests only
pytest -m "not slow"  # Skip slow tests
```

### Test Structure

```
backend/tests/
├── conftest.py           # Test fixtures
├── test_api/
│   ├── test_logs.py      # Log API tests
│   ├── test_analytics.py # Analytics tests
│   └── test_alerts.py    # Alert tests
├── test_services/
│   ├── test_log_service.py
│   └── test_alert_service.py
└── test_ml/
    └── test_anomaly_detection.py
```

---

## 🚢 Deployment

### Production Deployment

```bash
# Build production images
./scripts/deploy.sh build

# Start in production mode
./scripts/deploy.sh start prod

# Check status
./scripts/deploy.sh status
```

### Cloud Deployment

#### AWS
- Use ECS/Fargate for containers
- RDS for MongoDB (or MongoDB Atlas)
- ElastiCache for Redis
- ALB for load balancing

#### GCP
- Use GKE for Kubernetes
- Cloud SQL or MongoDB Atlas
- Memorystore for Redis
- Cloud Load Balancing

#### Azure
- AKS for Kubernetes
- Cosmos DB or MongoDB Atlas
- Azure Cache for Redis
- Azure Load Balancer

### SSL/TLS Setup

Update `config/nginx.conf` with your SSL certificates:

```nginx
listen 443 ssl http2;
ssl_certificate /etc/nginx/ssl/cert.pem;
ssl_certificate_key /etc/nginx/ssl/key.pem;
```

---

## 🐛 Troubleshooting

### Common Issues

#### Services won't start

```bash
# Check logs
docker-compose logs backend
docker-compose logs mongodb
docker-compose logs redis

# Restart services
./scripts/deploy.sh restart
```

#### MongoDB connection errors

```bash
# Verify MongoDB is running
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"

# Check credentials in .env
# Ensure MONGODB_URL is correct
```

#### High memory usage

```bash
# Check container stats
docker stats

# Adjust memory limits in docker-compose.yml
# Scale down workers if needed
```

### Logs and Debugging

```bash
# View logs for specific service
./scripts/deploy.sh logs backend
./scripts/deploy.sh logs ml_service

# Follow logs in real-time
docker-compose logs -f --tail=100

# Check system health
curl http://localhost:8000/api/v1/health
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black backend/
isort backend/

# Run linting
flake8 backend/
mypy backend/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

- **Documentation**: [Full Docs](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-log-analytics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-log-analytics/discussions)

---

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- scikit-learn for ML capabilities
- Redis for high-performance streaming
- MongoDB for flexible data storage
- React and Tailwind CSS for the frontend

---

<div align="center">

**Made with ❤️ by the AI Log Analytics Team**

[⬆ Back to Top](#ai-powered-log-analytics--anomaly-detection-platform)

</div>