# Deployment Guide

Complete guide for deploying the AI Log Analytics Platform in various environments.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring Setup](#monitoring-setup)
- [Backup & Recovery](#backup--recovery)

---

## Local Development

### Prerequisites

```bash
# Required software
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend development)
- Git

# Verify installations
docker --version
docker-compose --version
python --version
node --version
```

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/ai-log-analytics.git
cd ai-log-analytics

# 2. Copy environment file
cp .env.example .env

# 3. Update .env with your settings
nano .env

# 4. Start all services
chmod +x scripts/deploy.sh
./scripts/deploy.sh start

# 5. Access the platform
# Frontend: http://localhost:80
# API: http://localhost:8000/docs
# Grafana: http://localhost:3000
```

### Development Workflow

```bash
# Start with hot reload
docker-compose up

# View logs
docker-compose logs -f backend

# Run tests
docker-compose exec backend pytest

# Access shell
docker-compose exec backend bash

# Stop services
docker-compose down

# Clean everything
./scripts/deploy.sh clean
```

---

## Docker Deployment

### Single Host Production

#### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
```

#### 2. Application Setup

```bash
# Clone repository
git clone https://github.com/yourusername/ai-log-analytics.git
cd ai-log-analytics

# Create production environment file
cp .env.example .env
nano .env
```

**Production .env**:
```bash
# Application
ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Security
SECRET_KEY=<generate-strong-secret-key>  # Use: openssl rand -hex 32

# Database
MONGODB_URL=mongodb://admin:<strong-password>@mongodb:27017/
REDIS_URL=redis://redis:6379

# Monitoring
ENABLE_METRICS=true

# CORS (update with your domain)
CORS_ORIGINS=["https://your-domain.com"]

# Email Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=<app-password>

# Slack (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK
```

#### 3. SSL Certificate Setup

```bash
# Install Certbot
sudo apt install certbot

# Generate certificate
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem config/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem config/ssl/key.pem
```

**Update nginx.conf**:
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # ... rest of configuration
}
```

#### 4. Deploy

```bash
# Start in production mode
./scripts/deploy.sh start prod

# Verify services
docker-compose ps

# Check logs
docker-compose logs -f

# Initialize database
docker-compose exec backend python scripts/init_db.py

# Optional: Seed sample data
docker-compose exec backend python scripts/seed_data.py --duration-hours 24
```

#### 5. Setup Monitoring

```bash
# Access Grafana
# URL: http://your-domain.com:3000
# Default: admin/admin (change immediately)

# Import dashboards
# 1. Login to Grafana
# 2. Import from file: monitoring/grafana/dashboards/system_metrics.json

# Configure alerts
# 1. Settings > Notification channels
# 2. Add Email/Slack channels
```

---

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify cluster access
kubectl cluster-info
```

### 1. Create Kubernetes Manifests

**namespace.yaml**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: log-analytics
```

**mongodb-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mongodb
  namespace: log-analytics
spec:
  serviceName: mongodb
  replicas: 1
  selector:
    matchLabels:
      app: mongodb
  template:
    metadata:
      labels:
        app: mongodb
    spec:
      containers:
      - name: mongodb
        image: mongo:7.0
        ports:
        - containerPort: 27017
        env:
        - name: MONGO_INITDB_ROOT_USERNAME
          valueFrom:
            secretKeyRef:
              name: mongodb-secret
              key: username
        - name: MONGO_INITDB_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mongodb-secret
              key: password
        volumeMounts:
        - name: mongodb-data
          mountPath: /data/db
  volumeClaimTemplates:
  - metadata:
      name: mongodb-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 50Gi
```

**backend-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: log-analytics
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: your-registry/log-analytics-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: MONGODB_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: mongodb-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
  namespace: log-analytics
spec:
  selector:
    app: backend
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
```

**ingress.yaml**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: log-analytics-ingress
  namespace: log-analytics
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - your-domain.com
    secretName: log-analytics-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: backend-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
```

### 2. Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create secrets
kubectl create secret generic mongodb-secret \
  --from-literal=username=admin \
  --from-literal=password=<strong-password> \
  -n log-analytics

kubectl create secret generic app-secrets \
  --from-literal=mongodb-url=mongodb://admin:<password>@mongodb:27017/ \
  --from-literal=secret-key=<generate-key> \
  -n log-analytics

# Create configmap
kubectl create configmap app-config \
  --from-literal=redis-url=redis://redis:6379 \
  -n log-analytics

# Deploy components
kubectl apply -f k8s/mongodb-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/workers-deployment.yaml
kubectl apply -f k8s/ml-service-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n log-analytics
kubectl get svc -n log-analytics
kubectl get ingress -n log-analytics
```

### 3. Auto-scaling

**hpa.yaml**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: log-analytics
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

```bash
kubectl apply -f k8s/hpa.yaml
```

---

## Cloud Deployment

### AWS Deployment

#### Using ECS Fargate

1. **Setup Infrastructure**:

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Create ECS cluster
aws ecs create-cluster --cluster-name log-analytics-cluster

# Create task definitions
aws ecs register-task-definition --cli-input-json file://aws/backend-task-def.json
```

2. **Backend Task Definition** (aws/backend-task-def.json):

```json
{
  "family": "log-analytics-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "your-ecr-repo/backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MONGODB_URL",
          "value": "mongodb://your-documentdb-endpoint:27017"
        },
        {
          "name": "REDIS_URL",
          "value": "redis://your-elasticache-endpoint:6379"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/log-analytics",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "backend"
        }
      }
    }
  ]
}
```

3. **Create Services**:

```bash
# Create backend service
aws ecs create-service \
  --cluster log-analytics-cluster \
  --service-name backend-service \
  --task-definition log-analytics-backend \
  --desired-count 3 \
  --launch-type FARGATE \
  --load-balancers targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=backend,containerPort=8000
```

#### Using DocumentDB (MongoDB) and ElastiCache (Redis)

```bash
# Create DocumentDB cluster
aws docdb create-db-cluster \
  --db-cluster-identifier log-analytics-docdb \
  --engine docdb \
  --master-username admin \
  --master-user-password <password>

# Create ElastiCache cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id log-analytics-redis \
  --engine redis \
  --cache-node-type cache.t3.medium \
  --num-cache-nodes 1
```

### Google Cloud Platform

#### Using GKE

```bash
# Create GKE cluster
gcloud container clusters create log-analytics-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-2 \
  --zone=us-central1-a \
  --enable-autoscaling \
  --min-nodes=3 \
  --max-nodes=10

# Get credentials
gcloud container clusters get-credentials log-analytics-cluster --zone=us-central1-a

# Deploy using kubectl
kubectl apply -f k8s/
```

#### Using Cloud SQL and Memorystore

```bash
# Create Cloud SQL for PostgreSQL (alternative to MongoDB)
gcloud sql instances create log-analytics-db \
  --database-version=POSTGRES_14 \
  --tier=db-n1-standard-2 \
  --region=us-central1

# Create Memorystore (Redis)
gcloud redis instances create log-analytics-redis \
  --size=5 \
  --region=us-central1
```

### Azure Deployment

#### Using AKS

```bash
# Create AKS cluster
az aks create \
  --resource-group log-analytics-rg \
  --name log-analytics-aks \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group log-analytics-rg --name log-analytics-aks

# Deploy
kubectl apply -f k8s/
```

---

## Production Considerations

### Performance Optimization

1. **Database Tuning**:
```javascript
// MongoDB indexes
db.logs.createIndex({ timestamp: -1, service: 1 });
db.logs.createIndex({ level: 1, timestamp: -1 });

// Connection pooling
{
  minPoolSize: 10,
  maxPoolSize: 50,
  maxIdleTimeMS: 30000
}
```

2. **Redis Configuration**:
```conf
# Increase max memory
maxmemory 4gb
maxmemory-policy allkeys-lru

# Enable AOF for durability
appendonly yes
appendfsync everysec

# Optimize for streams
stream-node-max-bytes 4096
stream-node-max-entries 100
```

3. **Application Tuning**:
```python
# Increase worker threads
uvicorn.run(
    app,
    workers=4,  # Match CPU cores
    worker_class="uvicorn.workers.UvicornWorker"
)

# Connection pooling
MONGODB_MAX_POOL_SIZE=50
REDIS_MAX_CONNECTIONS=20
```

### Security Hardening

1. **Network Security**:
```bash
# Firewall rules (UFW)
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw enable

# Fail2ban for SSH protection
sudo apt install fail2ban
sudo systemctl enable fail2ban
```

2. **Application Security**:
```python
# Environment variables
SECRET_KEY = os.getenv("SECRET_KEY")  # 32+ characters
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Rate limiting
RATE_LIMIT_PER_MINUTE = 100

# CORS
CORS_ORIGINS = [
    "https://your-domain.com"  # No wildcards in production
]
```

3. **Database Security**:
```javascript
// MongoDB authentication
use admin
db.createUser({
  user: "admin",
  pwd: "<strong-password>",
  roles: ["root"]
})

// Enable authentication in mongod.conf
security:
  authorization: enabled
```

### Monitoring Setup

1. **Prometheus**:
```bash
# Start Prometheus
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

2. **Grafana**:
```bash
# Start Grafana
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -e "GF_SECURITY_ADMIN_PASSWORD=<password>" \
  grafana/grafana
```

3. **Log Aggregation**:
```yaml
# Fluentd configuration for centralized logging
<source>
  @type forward
  port 24224
</source>

<match **>
  @type elasticsearch
  host elasticsearch
  port 9200
  logstash_format true
</match>
```

---

## Backup & Recovery

### Automated Backups

**backup.sh**:
```bash
#!/bin/bash

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup MongoDB
docker exec mongodb mongodump --out $BACKUP_DIR/mongodb

# Backup ML models
cp -r ml/saved_models $BACKUP_DIR/

# Compress
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR.tar.gz s3://your-bucket/backups/

# Retention: Keep last 7 days
find /backups -name "*.tar.gz" -mtime +7 -delete
```

**Cron Job**:
```bash
# Add to crontab
0 2 * * * /path/to/backup.sh
```

### Recovery Procedures

```bash
# 1. Stop services
docker-compose down

# 2. Restore MongoDB
tar -xzf backup_20250212.tar.gz
mongorestore --uri="mongodb://localhost:27017" backup_20250212/mongodb

# 3. Restore ML models
cp -r backup_20250212/saved_models ml/

# 4. Start services
docker-compose up -d

# 5. Verify
curl http://localhost:8000/api/v1/health
```

---

## Troubleshooting

### Common Issues

1. **Services won't start**:
```bash
# Check Docker daemon
sudo systemctl status docker

# Check logs
docker-compose logs backend

# Verify environment variables
docker-compose config
```

2. **High memory usage**:
```bash
# Check container stats
docker stats

# Adjust limits in docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 1G
```

3. **Database connection errors**:
```bash
# Test MongoDB connection
docker exec -it mongodb mongosh

# Check network
docker network inspect ai-log-analytics_network
```

### Health Check Commands

```bash
# API health
curl http://localhost:8000/api/v1/health

# MongoDB
docker exec mongodb mongosh --eval "db.adminCommand('ping')"

# Redis
docker exec redis redis-cli ping

# View all services
docker-compose ps
```

---

## Maintenance

### Updates

```bash
# Pull latest images
docker-compose pull

# Rebuild
docker-compose build

# Rolling update (zero downtime)
docker-compose up -d --no-deps --build backend
```

### Log Rotation

```bash
# Configure Docker log rotation
# /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}

# Restart Docker
sudo systemctl restart docker
```

---

## Conclusion

This deployment guide covers various deployment scenarios from local development to production cloud deployments. Choose the approach that best fits your infrastructure and scale requirements.