#!/bin/bash

# ===================================
# Deployment Script
# AI Log Analytics Platform
# ===================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ai-log-analytics"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Functions
print_header() {
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_requirements() {
    print_header "Checking Requirements"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker is installed: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_success "Docker Compose is installed"
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    print_success "Docker daemon is running"
}

setup_environment() {
    print_header "Setting Up Environment"
    
    # Create .env file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f ".env.example" ]; then
            print_info "Creating .env file from .env.example"
            cp .env.example .env
            print_warning "Please update .env with your configuration"
        else
            print_warning "No .env file found. Using defaults."
        fi
    else
        print_success ".env file exists"
    fi
    
    # Create necessary directories
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p data/sample_logs
    mkdir -p ml/saved_models
    mkdir -p logs
    
    print_success "Directories created"
}

build_images() {
    print_header "Building Docker Images"
    
    print_info "Building images... This may take a while on first run."
    
    if docker-compose build; then
        print_success "Docker images built successfully"
    else
        print_error "Failed to build Docker images"
        exit 1
    fi
}

start_services() {
    print_header "Starting Services"
    
    local mode=${1:-"dev"}
    
    if [ "$mode" == "prod" ]; then
        print_info "Starting in PRODUCTION mode"
        docker-compose --profile production up -d
    else
        print_info "Starting in DEVELOPMENT mode"
        docker-compose up -d
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Services started successfully"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

wait_for_services() {
    print_header "Waiting for Services to be Ready"
    
    local max_attempts=30
    local attempt=0
    
    # Wait for MongoDB
    print_info "Waiting for MongoDB..."
    while ! docker-compose exec -T mongodb mongosh --eval "db.adminCommand('ping')" &> /dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            print_error "MongoDB failed to start"
            exit 1
        fi
        sleep 2
    done
    print_success "MongoDB is ready"
    
    # Wait for Redis
    print_info "Waiting for Redis..."
    attempt=0
    while ! docker-compose exec -T redis redis-cli ping &> /dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            print_error "Redis failed to start"
            exit 1
        fi
        sleep 2
    done
    print_success "Redis is ready"
    
    # Wait for Backend API
    print_info "Waiting for Backend API..."
    attempt=0
    while ! curl -f http://localhost:8000/api/v1/health &> /dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            print_error "Backend API failed to start"
            exit 1
        fi
        sleep 3
    done
    print_success "Backend API is ready"
}

initialize_database() {
    print_header "Initializing Database"
    
    print_info "Running database initialization..."
    
    if docker-compose exec -T backend python scripts/init_db.py; then
        print_success "Database initialized"
    else
        print_warning "Database initialization failed (may already be initialized)"
    fi
}

seed_sample_data() {
    print_header "Seeding Sample Data"
    
    read -p "Do you want to seed sample data? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Seeding sample data..."
        
        docker-compose exec -T backend python scripts/seed_data.py \
            --duration-hours 24 \
            --logs-per-minute 50 \
            --anomaly-rate 0.05 \
            --add-burst
        
        print_success "Sample data seeded"
    else
        print_info "Skipping sample data seeding"
    fi
}

train_ml_models() {
    print_header "Training ML Models"
    
    read -p "Do you want to train ML models? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Training Isolation Forest model..."
        
        docker-compose exec -T ml_service python -m ml.training.train_isolation_forest \
            --data-source mongodb \
            --limit 10000 \
            --contamination 0.1 \
            --n-estimators 100
        
        print_success "ML models trained"
    else
        print_info "Skipping ML model training"
    fi
}

show_status() {
    print_header "Service Status"
    
    docker-compose ps
    
    echo ""
    print_header "Access Information"
    print_info "Backend API:        http://localhost:8000"
    print_info "API Documentation:  http://localhost:8000/docs"
    print_info "Frontend:           http://localhost:80"
    print_info "MongoDB:            localhost:27017"
    print_info "Redis:              localhost:6379"
    echo ""
    print_info "View logs: docker-compose logs -f [service]"
    print_info "Stop services: ./scripts/deploy.sh stop"
}

stop_services() {
    print_header "Stopping Services"
    
    docker-compose down
    
    print_success "Services stopped"
}

clean_deployment() {
    print_header "Cleaning Deployment"
    
    print_warning "This will remove all containers, volumes, and data!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v --remove-orphans
        
        # Remove generated data
        rm -rf data/raw/*
        rm -rf data/processed/*
        rm -rf ml/saved_models/*
        rm -rf logs/*
        
        print_success "Deployment cleaned"
    else
        print_info "Clean cancelled"
    fi
}

show_logs() {
    local service=${1:-""}
    
    if [ -z "$service" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$service"
    fi
}

run_tests() {
    print_header "Running Tests"
    
    print_info "Running backend tests..."
    docker-compose exec -T backend pytest tests/ -v
    
    print_success "Tests completed"
}

backup_data() {
    print_header "Backing Up Data"
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    print_info "Backing up MongoDB..."
    docker-compose exec -T mongodb mongodump --out=/data/backup
    docker cp $(docker-compose ps -q mongodb):/data/backup "$backup_dir/mongodb"
    
    print_info "Backing up ML models..."
    cp -r ml/saved_models "$backup_dir/"
    
    print_success "Backup completed: $backup_dir"
}

# Main script logic
case "${1:-}" in
    start)
        check_requirements
        setup_environment
        build_images
        start_services "${2:-dev}"
        wait_for_services
        initialize_database
        seed_sample_data
        train_ml_models
        show_status
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services "${2:-dev}"
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "${2:-}"
        ;;
    clean)
        clean_deployment
        ;;
    test)
        run_tests
        ;;
    backup)
        backup_data
        ;;
    build)
        build_images
        ;;
    init-db)
        initialize_database
        ;;
    seed-data)
        seed_sample_data
        ;;
    train-models)
        train_ml_models
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|clean|test|backup|build|init-db|seed-data|train-models}"
        echo ""
        echo "Commands:"
        echo "  start [dev|prod]  - Start all services (default: dev)"
        echo "  stop              - Stop all services"
        echo "  restart [dev|prod]- Restart all services"
        echo "  status            - Show service status"
        echo "  logs [service]    - Show logs (optionally for specific service)"
        echo "  clean             - Remove all containers and data"
        echo "  test              - Run tests"
        echo "  backup            - Backup data and models"
        echo "  build             - Build Docker images"
        echo "  init-db           - Initialize database"
        echo "  seed-data         - Seed sample data"
        echo "  train-models      - Train ML models"
        exit 1
        ;;
esac

exit 0