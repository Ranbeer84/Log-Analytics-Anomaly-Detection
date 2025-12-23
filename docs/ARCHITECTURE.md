# System Architecture

## Overview

The AI Log Analytics Platform is built using a microservices architecture with event-driven processing, enabling horizontal scalability and fault tolerance.

## Architecture Principles

### 1. Separation of Concerns
- **API Layer**: Request handling and validation
- **Processing Layer**: Asynchronous log processing
- **Storage Layer**: Persistent data storage
- **ML Layer**: Anomaly detection and analytics
- **Presentation Layer**: User interface

### 2. Event-Driven Architecture
- Redis Streams for message passing
- Decoupled producers and consumers
- Guaranteed message delivery with consumer groups
- Back-pressure handling

### 3. Scalability
- Stateless services for horizontal scaling
- Database connection pooling
- Efficient caching strategies
- Load balancing at proxy layer

### 4. Reliability
- Health checks and auto-restart
- Graceful degradation
- Circuit breakers
- Comprehensive monitoring

---

## System Components

### Frontend Layer

```
┌─────────────────────────────────────────┐
│           React Frontend                 │
│  ┌────────────┐  ┌──────────────────┐  │
│  │ Dashboard  │  │  Log Viewer      │  │
│  │ Component  │  │  Component       │  │
│  └────────────┘  └──────────────────┘  │
│  ┌────────────┐  ┌──────────────────┐  │
│  │ Analytics  │  │  Alert Manager   │  │
│  │ Charts     │  │  Component       │  │
│  └────────────┘  └──────────────────┘  │
└─────────────────────────────────────────┘
           │
           │ HTTP/WebSocket
           ▼
┌─────────────────────────────────────────┐
│         Nginx Reverse Proxy              │
│   • Load Balancing                       │
│   • SSL Termination                      │
│   • Rate Limiting                        │
│   • Static Asset Serving                 │
└─────────────────────────────────────────┘
```

**Technologies:**
- React 18 with Hooks
- Vite for build tooling
- Tailwind CSS for styling
- Recharts for data visualization
- Axios for API calls
- WebSocket for real-time updates

**Key Features:**
- Real-time log streaming
- Interactive charts and graphs
- Advanced filtering and search
- Alert management interface
- Responsive design

---

### API Layer

```
┌─────────────────────────────────────────┐
│         FastAPI Backend                  │
│                                          │
│  ┌──────────────────────────────────┐  │
│  │      API Routes                   │  │
│  │  • /logs      • /analytics       │  │
│  │  • /alerts    • /health          │  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │      Services Layer               │  │
│  │  • LogService                     │  │
│  │  • AnalyticsService              │  │
│  │  • AlertService                  │  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │      Core Layer                   │  │
│  │  • Database Connection           │  │
│  │  • Redis Client                  │  │
│  │  • Authentication                │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Technologies:**
- FastAPI (Python 3.11+)
- Pydantic for validation
- Motor (Async MongoDB driver)
- Redis client
- JWT for authentication

**Responsibilities:**
1. **Request Handling**: Receive and validate HTTP requests
2. **Business Logic**: Execute core business operations
3. **Data Validation**: Ensure data integrity with Pydantic
4. **Stream Publishing**: Push logs to Redis Streams
5. **Database Operations**: CRUD operations on MongoDB
6. **WebSocket Management**: Handle real-time connections

**API Design:**
- RESTful endpoints
- JSON request/response
- Pagination for large datasets
- Rate limiting
- Comprehensive error handling

---

### Processing Layer

```
┌─────────────────────────────────────────┐
│         Redis Streams                    │
│  ┌────────────────────────────────┐    │
│  │   log_stream (Message Queue)   │    │
│  │   • Persistent                  │    │
│  │   • Consumer Groups             │    │
│  │   • Guaranteed Delivery         │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
           │
           │ XREADGROUP
           ▼
┌─────────────────────────────────────────┐
│         Worker Processes                 │
│  ┌────────────────────────────────┐    │
│  │   Log Consumer Worker          │    │
│  │   • Batch Processing           │    │
│  │   • Bulk Writes to MongoDB     │    │
│  │   • Pending Message Recovery   │    │
│  └────────────────────────────────┘    │
│  ┌────────────────────────────────┐    │
│  │   Anomaly Detection Worker     │    │
│  │   • ML Inference               │    │
│  │   • Feature Extraction         │    │
│  │   • Score Calculation          │    │
│  └────────────────────────────────┘    │
│  ┌────────────────────────────────┐    │
│  │   Alert Worker                 │    │
│  │   • Rule Evaluation            │    │
│  │   • Notification Dispatch      │    │
│  │   • Alert Aggregation          │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**Worker Design:**
- Independent processes
- Consumer groups for load distribution
- Graceful shutdown handling
- Health monitoring
- Automatic restart on failure

**Processing Flow:**
1. **Ingestion**: API receives log → Validates → Publishes to stream
2. **Persistence**: Consumer reads from stream → Batch write to MongoDB → ACK messages
3. **Analysis**: Anomaly worker reads logs → Extracts features → ML inference → Store results
4. **Alerting**: Alert worker evaluates rules → Triggers notifications → Updates alert status

---

### ML Layer

```
┌─────────────────────────────────────────┐
│         ML Service                       │
│                                          │
│  ┌────────────────────────────────┐    │
│  │   Feature Engineering          │    │
│  │   • LogFeatureExtractor        │    │
│  │   • SequenceFeatureExtractor   │    │
│  │   • TF-IDF Vectorizer          │    │
│  └────────────────────────────────┘    │
│  ┌────────────────────────────────┐    │
│  │   ML Models                    │    │
│  │   • Isolation Forest           │    │
│  │   • Autoencoder (Optional)     │    │
│  │   • LSTM (Optional)            │    │
│  └────────────────────────────────┘    │
│  ┌────────────────────────────────┐    │
│  │   Inference Engine             │    │
│  │   • Real-time Scoring          │    │
│  │   • Batch Processing           │    │
│  │   • Model Versioning           │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**ML Pipeline:**

1. **Feature Extraction**:
   - Numerical: response_time, status_code
   - Categorical: level, service (one-hot encoded)
   - Temporal: hour, day_of_week, time_since_last
   - Text: TF-IDF on log messages
   - Statistical: rolling averages, std dev

2. **Model Training**:
   ```python
   # Isolation Forest
   - contamination: 0.1 (expect 10% anomalies)
   - n_estimators: 100 (number of trees)
   - max_samples: auto
   - Training: Unsupervised on normal logs
   ```

3. **Inference**:
   - Input: Raw log entry
   - Feature extraction
   - Model prediction
   - Anomaly score calculation (0-1)
   - Severity classification (low/medium/high/critical)

4. **Model Updates**:
   - Periodic retraining (daily/weekly)
   - Incremental learning
   - A/B testing for model versions
   - Performance monitoring

---

### Storage Layer

```
┌─────────────────────────────────────────┐
│         MongoDB                          │
│                                          │
│  ┌────────────────────────────────┐    │
│  │   logs Collection              │    │
│  │   • Indexed on timestamp       │    │
│  │   • Indexed on level, service  │    │
│  │   • Text index on message      │    │
│  │   • Compound indexes           │    │
│  └────────────────────────────────┘    │
│  ┌────────────────────────────────┐    │
│  │   anomalies Collection         │    │
│  │   • Linked to logs             │    │
│  │   • Anomaly scores & metadata  │    │
│  └────────────────────────────────┘    │
│  ┌────────────────────────────────┐    │
│  │   alerts Collection            │    │
│  │   • Alert rules & instances    │    │
│  │   • Status tracking            │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│         Redis                            │
│                                          │
│  • Streams: Message queuing             │
│  • Cache: Analytics results             │
│  • Pub/Sub: Real-time notifications     │
│  • TTL: Session management              │
└─────────────────────────────────────────┘
```

**Data Models:**

```javascript
// Log Entry
{
  _id: ObjectId,
  timestamp: ISODate,
  level: "ERROR|WARN|INFO|DEBUG|CRITICAL",
  service: String,
  message: String,
  endpoint: String,
  status_code: Number,
  response_time: Number,
  ip_address: String,
  user_id: String,
  metadata: Object,
  processed: Boolean,
  anomaly_checked: Boolean,
  created_at: ISODate
}

// Anomaly
{
  _id: ObjectId,
  log_id: ObjectId,
  anomaly_score: Number (0-1),
  severity: "low|medium|high|critical",
  confidence: Number,
  model_version: String,
  features_used: Array,
  detected_at: ISODate
}

// Alert
{
  _id: ObjectId,
  alert_type: String,
  severity: String,
  service: String,
  condition: Object,
  status: "open|acknowledged|resolved",
  triggered_at: ISODate,
  acknowledged_at: ISODate,
  resolved_at: ISODate
}
```

---

### Monitoring Layer

```
┌─────────────────────────────────────────┐
│         Prometheus                       │
│   • Metrics Collection                   │
│   • Time Series Database                 │
│   • Alert Evaluation                     │
└─────────────────────────────────────────┘
           │
           │ Scrape Metrics
           ▼
┌─────────────────────────────────────────┐
│    Application Metrics                   │
│  • /metrics endpoints                    │
│  • Custom business metrics               │
│  • System metrics (CPU, RAM, Disk)       │
└─────────────────────────────────────────┘
           │
           │ Query
           ▼
┌─────────────────────────────────────────┐
│         Grafana                          │
│   • Dashboard Visualization              │
│   • Alert Notifications                  │
│   • Historical Analysis                  │
└─────────────────────────────────────────┘
```

**Key Metrics:**
- **Application**: Request rate, latency, error rate
- **Business**: Log ingestion rate, anomaly rate
- **Infrastructure**: CPU, memory, disk, network
- **Database**: Connection pool, query time
- **ML**: Inference latency, model accuracy

---

## Data Flow Diagrams

### Log Ingestion Flow

```
┌──────────┐
│  Client  │
│(App/User)│
└────┬─────┘
     │
     │ POST /api/v1/logs/ingest
     ▼
┌────────────────┐
│  API Gateway   │
│  (FastAPI)     │
│ ┌────────────┐ │
│ │ Validate   │ │
│ └────────────┘ │
└────┬───────────┘
     │
     │ Publish
     ▼
┌────────────────┐
│ Redis Stream   │
│  log_stream    │
└────┬───────────┘
     │
     │ XREADGROUP
     ▼
┌────────────────┐
│ Log Consumer   │
│   Worker       │
│ ┌────────────┐ │
│ │Bulk Insert │ │
│ └────────────┘ │
└────┬───────────┘
     │
     │ Write
     ▼
┌────────────────┐
│   MongoDB      │
│ logs collection│
└────────────────┘
```

### Anomaly Detection Flow

```
┌────────────────┐
│   MongoDB      │
│ logs collection│
└────┬───────────┘
     │
     │ Read New Logs
     ▼
┌────────────────┐
│Anomaly Worker  │
│ ┌────────────┐ │
│ │  Feature   │ │
│ │ Extraction │ │
│ └──────┬─────┘ │
│        │       │
│ ┌──────▼─────┐ │
│ │ML Inference│ │
│ │(Isolation  │ │
│ │  Forest)   │ │
│ └──────┬─────┘ │
└────────┼───────┘
         │
         │ if anomaly_score > threshold
         ▼
┌────────────────┐
│   MongoDB      │
│   anomalies    │
│   collection   │
└────┬───────────┘
     │
     │ Trigger
     ▼
┌────────────────┐
│ Alert Worker   │
│ ┌────────────┐ │
│ │  Evaluate  │ │
│ │   Rules    │ │
│ └──────┬─────┘ │
│        │       │
│ ┌──────▼─────┐ │
│ │   Send     │ │
│ │Notification│ │
│ └────────────┘ │
└────────────────┘
     │
     │ Email/Slack
     ▼
┌────────────────┐
│     Users      │
└────────────────┘
```

---

## Scalability Patterns

### Horizontal Scaling

**API Layer:**
```bash
# Scale FastAPI backends
docker-compose up --scale backend=3
```
- Stateless design enables easy scaling
- Nginx load balancer distributes traffic
- Session stored in Redis (not in-memory)

**Worker Layer:**
```bash
# Scale workers independently
docker-compose up --scale log_consumer=5 --scale anomaly_worker=3
```
- Consumer groups distribute messages
- Each worker processes subset of stream
- No coordination needed between workers

### Vertical Scaling

**Database:**
- Increase MongoDB resources
- Enable sharding for very large datasets
- Use replica sets for read scaling

**Cache:**
- Increase Redis memory
- Use Redis Cluster for data partitioning
- Implement cache warming strategies

### Database Sharding Strategy

```
Range-based sharding by timestamp:
┌─────────────────────────────────────┐
│  Shard 1: 2025-01-01 to 2025-01-31 │
│  Shard 2: 2025-02-01 to 2025-02-28 │
│  Shard 3: 2025-03-01 to 2025-03-31 │
└─────────────────────────────────────┘

Benefits:
• Queries typically time-based
• Easy archival of old data
• Predictable growth pattern
```

---

## Performance Optimization

### Caching Strategy

```
┌─────────────────────────────────────┐
│           Cache Layers              │
│                                     │
│  L1: Application (In-memory)        │
│      • Feature vectors              │
│      • Model predictions            │
│                                     │
│  L2: Redis (Distributed)            │
│      • Analytics results (2m TTL)   │
│      • Aggregated metrics (5m TTL)  │
│      • Dashboard data (1m TTL)      │
│                                     │
│  L3: MongoDB (Persistent)           │
│      • All raw data                 │
│      • Historical analytics         │
└─────────────────────────────────────┘
```

### Database Optimization

**Indexes:**
```javascript
// Compound index for common queries
db.logs.createIndex({ timestamp: -1, service: 1 });
db.logs.createIndex({ level: 1, timestamp: -1 });
db.logs.createIndex({ service: 1, timestamp: -1 });

// Text index for search
db.logs.createIndex({ message: "text" });
```

**Query Optimization:**
- Use projections to fetch only needed fields
- Limit result sets with pagination
- Use aggregation pipeline for complex queries
- Implement query result caching

### Connection Pooling

```python
# MongoDB
- Min pool size: 10
- Max pool size: 50
- Connection timeout: 10s

# Redis
- Connection pool: 20
- Socket keepalive: True
- Health check interval: 30s
```

---

## Security Architecture

### Authentication & Authorization

```
┌──────────────────────────────────────┐
│         Security Layers              │
│                                      │
│  1. API Gateway (Nginx)              │
│     • Rate limiting                  │
│     • DDoS protection               │
│     • SSL/TLS termination           │
│                                      │
│  2. Application (FastAPI)            │
│     • JWT token validation          │
│     • Role-based access control     │
│     • Input sanitization            │
│                                      │
│  3. Database                         │
│     • Authentication required        │
│     • Network isolation             │
│     • Encryption at rest            │
└──────────────────────────────────────┘
```

### Data Protection

- **Encryption in Transit**: TLS 1.3
- **Encryption at Rest**: MongoDB encryption
- **Sensitive Data**: PII masking in logs
- **Access Control**: Role-based permissions
- **Audit Logging**: Track all data access

---

## Deployment Architecture

### Development
```
Docker Compose
• All services on single host
• Volume mounts for hot reload
• Debug logging enabled
```

### Staging
```
Docker Compose with profiles
• Production-like environment
• Separate database instances
• Limited resources
```

### Production
```
Kubernetes (Recommended)
• High availability
• Auto-scaling
• Health checks & probes
• Rolling updates
• Resource limits
```

---

## Disaster Recovery

### Backup Strategy

```bash
# Automated Daily Backups
0 2 * * * /scripts/backup_mongodb.sh
0 3 * * * /scripts/backup_models.sh
```

**Backup Components:**
1. MongoDB database dumps
2. ML model artifacts
3. Configuration files
4. Redis snapshots (optional)

**Retention Policy:**
- Daily backups: 7 days
- Weekly backups: 4 weeks
- Monthly backups: 12 months

### Recovery Procedures

**Database Recovery:**
```bash
# Restore MongoDB
mongorestore --uri="$MONGODB_URI" /backups/latest/

# Verify data integrity
mongo --eval "db.logs.count()"
```

**Service Recovery:**
```bash
# Restart services
./scripts/deploy.sh restart

# Health check
curl http://localhost:8000/api/v1/health
```

---

## Monitoring & Observability

### Health Checks

**Endpoint**: `/api/v1/health`

```json
{
  "status": "healthy",
  "services": {
    "api": "healthy",
    "mongodb": "healthy",
    "redis": "healthy",
    "ml_service": "healthy",
    "workers": "healthy"
  },
  "metrics": {
    "uptime": "5d 12h 34m",
    "requests_per_second": 156.4,
    "error_rate": 0.02
  }
}
```

### Alerting Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| API Error Rate | >5% | >10% |
| API Latency (P95) | >1s | >5s |
| Anomaly Rate | >10% | >20% |
| CPU Usage | >70% | >90% |
| Memory Usage | >80% | >95% |
| Disk Usage | >80% | >90% |

---

## Technology Decisions

### Why FastAPI?
- ✅ High performance (async/await)
- ✅ Automatic API documentation
- ✅ Type checking with Pydantic
- ✅ Modern Python features

### Why MongoDB?
- ✅ Flexible schema for diverse logs
- ✅ Excellent query performance
- ✅ Built-in sharding support
- ✅ Time-series collections

### Why Redis Streams?
- ✅ Guaranteed message delivery
- ✅ Consumer groups for scaling
- ✅ Persistent by default
- ✅ High throughput

### Why Isolation Forest?
- ✅ Unsupervised learning
- ✅ Fast training and inference
- ✅ Works well with mixed data types
- ✅ Low false positive rate

---

## Future Enhancements

1. **Distributed Tracing**: Integrate with Jaeger/Zipkin
2. **Auto-scaling**: Kubernetes HPA based on metrics
3. **Advanced ML**: Deep learning models (LSTM, Transformers)
4. **Multi-tenancy**: Support multiple organizations
5. **Log Correlation**: Automatic correlation of related logs
6. **Predictive Alerting**: ML-based alert predictions
7. **Custom Dashboards**: User-defined dashboards
8. **API Gateway**: Kong/Envoy for advanced routing

---

## Conclusion

This architecture provides a solid foundation for a production-grade log analytics platform with:
- ✅ High scalability and performance
- ✅ Fault tolerance and reliability
- ✅ Comprehensive monitoring
- ✅ Security best practices
- ✅ Easy deployment and maintenance

The modular design allows for independent scaling and evolution of components as requirements grow.