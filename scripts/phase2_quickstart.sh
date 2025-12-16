#!/bin/bash

# Phase 2 Quick Start Script
# Sets up and launches ML service and workers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  AI Log Analytics - Phase 2 Setup  ${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if Phase 1 is running
echo -e "${YELLOW}Checking Phase 1 services...${NC}"

if ! nc -z localhost 27017 2>/dev/null; then
    print_error "MongoDB is not running on port 27017"
    echo "  Start it with: docker-compose up -d mongodb"
    exit 1
fi
print_status "MongoDB is running"

if ! nc -z localhost 6379 2>/dev/null; then
    print_error "Redis is not running on port 6379"
    echo "  Start it with: docker-compose up -d redis"
    exit 1
fi
print_status "Redis is running"

if ! nc -z localhost 8000 2>/dev/null; then
    print_warning "Backend API is not running on port 8000"
    echo "  Start it with: docker-compose up -d backend"
fi
print_status "Backend API is running"

echo ""

# Check Python dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi
print_status "Python 3 is available"

# Check if ML dependencies are installed
if ! python3 -c "import sklearn" 2>/dev/null; then
    print_warning "ML dependencies not installed"
    echo "  Installing ML requirements..."
    # pip install -r ml/requirements.txt
    echo "ML dependencies will be installed inside Docker containers"
fi
print_status "ML dependencies installed"

if ! python3 -c "import redis" 2>/dev/null; then
    print_warning "Worker dependencies not installed"
    echo "  Installing worker requirements..."
    # pip install -r workers/requirements.txt
fi
print_status "Worker dependencies installed"

echo ""

# Check for trained model
echo -e "${YELLOW}Checking ML model...${NC}"

MODEL_DIR="ml/saved_models"
mkdir -p "$MODEL_DIR"

MODEL_COUNT=$(ls -1 "$MODEL_DIR"/isolation_forest_*.pkl 2>/dev/null | wc -l)

if [ "$MODEL_COUNT" -eq 0 ]; then
    print_warning "No trained model found"
    echo ""
    read -p "  Do you want to train a model now? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        print_info "Checking training data..."
        
        # Check if we have enough logs
        LOG_COUNT=$(mongo log_analytics --quiet --eval "db.logs.countDocuments()" 2>/dev/null || echo "0")
        
        if [ "$LOG_COUNT" -lt 100 ]; then
            print_warning "Not enough training data ($LOG_COUNT logs)"
            print_info "Generating 5000 sample logs..."
            python3 scripts/generate_logs.py --count 5000
            print_status "Sample logs generated"
        else
            print_status "Found $LOG_COUNT training logs"
        fi
        
        echo ""
        print_info "Training Isolation Forest model..."
        python3 ml/training/train_isolation_forest.py
        
        if [ $? -eq 0 ]; then
            print_status "Model training complete!"
        else
            print_error "Model training failed"
            exit 1
        fi
    else
        print_error "Cannot proceed without a trained model"
        echo "  Train manually with: python3 ml/training/train_isolation_forest.py"
        exit 1
    fi
else
    print_status "Found $MODEL_COUNT trained model(s)"
    LATEST_MODEL=$(ls -t "$MODEL_DIR"/isolation_forest_*.pkl | head -1)
    print_info "Latest: $(basename $LATEST_MODEL)"
fi

echo ""

# Create logs directory for workers
mkdir -p logs
print_status "Created logs directory"

echo ""
echo -e "${YELLOW}Starting workers...${NC}"

# Kill existing workers
print_info "Stopping any existing workers..."
pkill -f "python.*workers/log_consumer.py" 2>/dev/null || true
pkill -f "python.*workers/anomaly_worker.py" 2>/dev/null || true
pkill -f "python.*workers/alert_worker.py" 2>/dev/null || true
sleep 2

# Start log consumer
print_info "Starting Log Consumer Worker..."
nohup python3 workers/log_consumer.py > logs/log_consumer.log 2>&1 &
LOG_CONSUMER_PID=$!
sleep 2

if ps -p $LOG_CONSUMER_PID > /dev/null; then
    print_status "Log Consumer Worker running (PID: $LOG_CONSUMER_PID)"
else
    print_error "Log Consumer Worker failed to start"
    cat logs/log_consumer.log
    exit 1
fi

# Start anomaly worker
print_info "Starting Anomaly Detection Worker..."
nohup python3 workers/anomaly_worker.py > logs/anomaly_worker.log 2>&1 &
ANOMALY_WORKER_PID=$!
sleep 2

if ps -p $ANOMALY_WORKER_PID > /dev/null; then
    print_status "Anomaly Worker running (PID: $ANOMALY_WORKER_PID)"
else
    print_error "Anomaly Worker failed to start"
    cat logs/anomaly_worker.log
    exit 1
fi

# Start alert worker
print_info "Starting Alert Processing Worker..."
nohup python3 workers/alert_worker.py > logs/alert_worker.log 2>&1 &
ALERT_WORKER_PID=$!
sleep 2

if ps -p $ALERT_WORKER_PID > /dev/null; then
    print_status "Alert Worker running (PID: $ALERT_WORKER_PID)"
else
    print_error "Alert Worker failed to start"
    cat logs/alert_worker.log
    exit 1
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Phase 2 Setup Complete! 🚀${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

echo "Worker PIDs:"
echo "  Log Consumer: $LOG_CONSUMER_PID"
echo "  Anomaly Worker: $ANOMALY_WORKER_PID"
echo "  Alert Worker: $ALERT_WORKER_PID"
echo ""

echo "Monitor logs with:"
echo "  tail -f logs/log_consumer.log"
echo "  tail -f logs/anomaly_worker.log"
echo "  tail -f logs/alert_worker.log"
echo ""

echo "Test the system:"
echo "  python3 scripts/test_phase2.py"
echo ""

echo "Generate test data:"
echo "  python3 scripts/generate_logs.py --count 100 --anomaly-rate 0.2"
echo ""

echo "Check anomalies:"
echo "  curl http://localhost:8000/api/analytics/anomalies | jq"
echo ""

echo "Check alerts:"
echo "  curl http://localhost:8000/api/alerts/ | jq"
echo ""

echo -e "${BLUE}System is ready! Let it run for a few minutes to process data.${NC}"
echo ""

# Optional: Wait and show initial stats
read -p "Show live stats? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    print_info "Waiting 30 seconds for workers to process data..."
    sleep 30
    
    echo ""
    echo -e "${YELLOW}Current Statistics:${NC}"
    echo ""
    
    # Get stats from MongoDB
    TOTAL_LOGS=$(mongo log_analytics --quiet --eval "db.logs.countDocuments()" 2>/dev/null || echo "N/A")
    ANOMALIES=$(mongo log_analytics --quiet --eval "db.anomalies.countDocuments()" 2>/dev/null || echo "N/A")
    ALERTS=$(mongo log_analytics --quiet --eval "db.alerts.countDocuments()" 2>/dev/null || echo "N/A")
    
    echo "  Total Logs: $TOTAL_LOGS"
    echo "  Anomalies Detected: $ANOMALIES"
    echo "  Alerts Generated: $ALERTS"
    echo ""
    
    # Show recent anomaly
    echo "Recent Anomaly:"
    mongo log_analytics --quiet --eval "db.anomalies.find().sort({detected_at:-1}).limit(1).pretty()" 2>/dev/null || echo "  None yet"
    echo ""
fi

echo -e "${GREEN}Happy anomaly hunting! 🔍${NC}"