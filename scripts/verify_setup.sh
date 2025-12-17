#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Header
echo -e "${BLUE}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║       AI LOG ANALYTICS - SETUP VERIFICATION              ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

# Function to check command
check_command() {
    if command -v $1 &> /dev/null; then
        print_status 0 "$1 is installed"
        return 0
    else
        print_status 1 "$1 is NOT installed"
        return 1
    fi
}

# Function to check docker container
check_container() {
    if docker ps | grep -q $1; then
        print_status 0 "$1 container is running"
        return 0
    else
        print_status 1 "$1 container is NOT running"
        return 1
    fi
}

# Function to check port
check_port() {
    if nc -z localhost $1 2>/dev/null; then
        print_status 0 "Port $1 ($2) is accessible"
        return 0
    else
        print_status 1 "Port $1 ($2) is NOT accessible"
        return 1
    fi
}

# Function to check API endpoint
check_endpoint() {
    response=$(curl -s -o /dev/null -w "%{http_code}" $1)
    if [ "$response" = "200" ]; then
        print_status 0 "API endpoint $1 is working (HTTP $response)"
        return 0
    else
        print_status 1 "API endpoint $1 failed (HTTP $response)"
        return 1
    fi
}

echo -e "${CYAN}${BOLD}[1/6] Checking Prerequisites${NC}\n"

PREREQ_OK=true

check_command "docker" || PREREQ_OK=false
check_command "docker-compose" || PREREQ_OK=false
check_command "python3" || PREREQ_OK=false
check_command "make" || PREREQ_OK=false

if [ "$PREREQ_OK" = false ]; then
    echo -e "\n${RED}${BOLD}❌ Missing prerequisites. Please install missing tools.${NC}\n"
    exit 1
fi

echo -e "\n${CYAN}${BOLD}[2/6] Checking Project Structure${NC}\n"

STRUCTURE_OK=true

# Check critical directories
for dir in backend frontend ml workers scripts config; do
    if [ -d "$dir" ]; then
        print_status 0 "Directory $dir/ exists"
    else
        print_status 1 "Directory $dir/ is missing"
        STRUCTURE_OK=false
    fi
done

# Check critical files
for file in docker-compose.yml Makefile backend/requirements.txt; do
    if [ -f "$file" ]; then
        print_status 0 "File $file exists"
    else
        print_status 1 "File $file is missing"
        STRUCTURE_OK=false
    fi
done

if [ "$STRUCTURE_OK" = false ]; then
    echo -e "\n${RED}${BOLD}❌ Project structure incomplete.${NC}\n"
    exit 1
fi

echo -e "\n${CYAN}${BOLD}[3/6] Checking Docker Containers${NC}\n"

CONTAINERS_OK=true

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker daemon is not running. Please start Docker.${NC}\n"
    exit 1
fi

# Check containers
check_container "mongodb" || CONTAINERS_OK=false
check_container "redis" || CONTAINERS_OK=false
check_container "backend" || CONTAINERS_OK=false

if [ "$CONTAINERS_OK" = false ]; then
    echo -e "\n${YELLOW}${BOLD}⚠️  Some containers are not running.${NC}"
    echo -e "${YELLOW}Run: ${BOLD}make up${NC}${YELLOW} or ${BOLD}docker-compose up -d${NC}\n"
    
    read -p "Would you like to start the containers now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "\n${CYAN}Starting containers...${NC}\n"
        docker-compose up -d
        echo -e "\n${CYAN}Waiting 10 seconds for services to initialize...${NC}\n"
        sleep 10
    else
        exit 1
    fi
fi

echo -e "\n${CYAN}${BOLD}[4/6] Checking Service Ports${NC}\n"

PORTS_OK=true

check_port 27017 "MongoDB" || PORTS_OK=false
check_port 6379 "Redis" || PORTS_OK=false
check_port 8000 "Backend API" || PORTS_OK=false

if [ "$PORTS_OK" = false ]; then
    echo -e "\n${YELLOW}${BOLD}⚠️  Some ports are not accessible.${NC}"
    echo -e "${YELLOW}Check if services are running properly.${NC}\n"
fi

echo -e "\n${CYAN}${BOLD}[5/6] Checking API Endpoints${NC}\n"

API_OK=true

# Wait a bit for API to be ready
sleep 2

check_endpoint "http://localhost:8000/api/health" || API_OK=false
check_endpoint "http://localhost:8000/api/health/detailed" || API_OK=false

# Test analytics endpoints (may fail if no data yet)
curl -s http://localhost:8000/api/analytics/dashboard > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_status 0 "Analytics endpoints are accessible"
else
    print_status 1 "Analytics endpoints returned error (may be normal if no data)"
fi

curl -s http://localhost:8000/api/alerts/rules > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_status 0 "Alert endpoints are accessible"
else
    print_status 1 "Alert endpoints returned error"
    API_OK=false
fi

echo -e "\n${CYAN}${BOLD}[6/6] Checking Phase Files${NC}\n"

PHASE_FILES_OK=true

# Phase 1 files
phase1_files=(
    "backend/app/__init__.py"
    "backend/app/main.py"
    "backend/app/config.py"
    "backend/app/core/database.py"
    "backend/app/core/redis_client.py"
    "docker-compose.yml"
)

# Phase 2 files
phase2_files=(
    "backend/app/models/log_entry.py"
    "backend/app/models/anomaly.py"
    "backend/app/models/alert.py"
    "backend/app/api/routes/logs.py"
    "backend/app/services/log_service.py"
)

# Phase 3 files
phase3_files=(
    "ml/Dockerfile"
    "frontend/Dockerfile"
    "frontend/nginx.conf"
    "backend/app/services/stream_service.py"
    "backend/app/services/analytics_service.py"
    "backend/app/services/alert_service.py"
    "backend/app/api/routes/analytics.py"
    "backend/app/api/routes/alerts.py"
    "backend/app/utils/log_parser.py"
    "backend/app/utils/validators.py"
    "backend/app/utils/helpers.py"
    "backend/app/core/security.py"
)

echo -e "${BLUE}Phase 1 Files:${NC}"
for file in "${phase1_files[@]}"; do
    if [ -f "$file" ]; then
        print_status 0 "$file"
    else
        print_status 1 "$file"
        PHASE_FILES_OK=false
    fi
done

echo -e "\n${BLUE}Phase 2 Files:${NC}"
for file in "${phase2_files[@]}"; do
    if [ -f "$file" ]; then
        print_status 0 "$file"
    else
        print_status 1 "$file"
        PHASE_FILES_OK=false
    fi
done

echo -e "\n${BLUE}Phase 3 Files:${NC}"
for file in "${phase3_files[@]}"; do
    if [ -f "$file" ]; then
        print_status 0 "$file"
    else
        print_status 1 "$file"
        PHASE_FILES_OK=false
    fi
done

# Final Summary
echo -e "\n${CYAN}${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}                      SUMMARY                              ${NC}"
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════${NC}\n"

ALL_OK=true

if [ "$PREREQ_OK" = true ]; then
    echo -e "${GREEN}✓${NC} Prerequisites: OK"
else
    echo -e "${RED}✗${NC} Prerequisites: FAILED"
    ALL_OK=false
fi

if [ "$STRUCTURE_OK" = true ]; then
    echo -e "${GREEN}✓${NC} Project Structure: OK"
else
    echo -e "${RED}✗${NC} Project Structure: FAILED"
    ALL_OK=false
fi

if [ "$CONTAINERS_OK" = true ]; then
    echo -e "${GREEN}✓${NC} Docker Containers: OK"
else
    echo -e "${YELLOW}⚠${NC}  Docker Containers: SOME NOT RUNNING"
fi

if [ "$PORTS_OK" = true ]; then
    echo -e "${GREEN}✓${NC} Service Ports: OK"
else
    echo -e "${YELLOW}⚠${NC}  Service Ports: SOME NOT ACCESSIBLE"
fi

if [ "$API_OK" = true ]; then
    echo -e "${GREEN}✓${NC} API Endpoints: OK"
else
    echo -e "${YELLOW}⚠${NC}  API Endpoints: SOME ISSUES"
fi

if [ "$PHASE_FILES_OK" = true ]; then
    echo -e "${GREEN}✓${NC} Phase Files: OK"
else
    echo -e "${RED}✗${NC} Phase Files: SOME MISSING"
    ALL_OK=false
fi

echo ""

if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}${BOLD}🎉 ALL CHECKS PASSED!${NC}"
    echo -e "${GREEN}Your AI Log Analytics system is ready!${NC}\n"
    
    echo -e "${CYAN}${BOLD}Next Steps:${NC}"
    echo -e "  1. Run comprehensive tests: ${BOLD}python scripts/test_all_phases.py${NC}"
    echo -e "  2. Generate sample logs: ${BOLD}python scripts/generate_logs.py${NC}"
    echo -e "  3. Access API docs: ${BOLD}http://localhost:8000/docs${NC}"
    echo -e "  4. View health status: ${BOLD}curl http://localhost:8000/api/health/detailed${NC}\n"
    
    exit 0
else
    echo -e "${RED}${BOLD}❌ SOME CHECKS FAILED${NC}"
    echo -e "${RED}Please fix the issues above before proceeding.${NC}\n"
    exit 1
fi