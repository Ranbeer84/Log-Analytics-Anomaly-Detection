#!/bin/bash

# Quick interactive test script for Phases 1-3
# Run: ./scripts/quick_test.sh

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

API="http://localhost:8000"

echo -e "${BLUE}${BOLD}"
echo "╔════════════════════════════════════════════════════╗"
echo "║    AI LOG ANALYTICS - QUICK TEST SUITE            ║"
echo "╚════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Test function
test_endpoint() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}
    
    echo -n "Testing $name... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    
    if [ "$response" = "$expected_code" ]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP $response)"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (HTTP $response, expected $expected_code)"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Phase 1 Tests
echo -e "${BLUE}${BOLD}Phase 1: Core Infrastructure${NC}\n"

test_endpoint "Health Check" "$API/api/health"
test_endpoint "Detailed Health" "$API/api/health/detailed"

# Phase 2 Tests
echo -e "\n${BLUE}${BOLD}Phase 2: Log Processing${NC}\n"

# Test log ingestion
echo -n "Testing Log Ingestion... "
RESPONSE=$(curl -s -X POST "$API/api/logs/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Quick test log",
    "level": "INFO",
    "service": "test-service"
  }')

if echo "$RESPONSE" | grep -q "success"; then
    LOG_ID=$(echo "$RESPONSE" | jq -r '.log_id')
    echo -e "${GREEN}✓ PASS${NC} (Log ID: ${LOG_ID:0:8}...)"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

# Test batch ingestion
echo -n "Testing Batch Ingestion... "
RESPONSE=$(curl -s -X POST "$API/api/logs/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      {"message": "Batch 1", "level": "INFO", "service": "test"},
      {"message": "Batch 2", "level": "ERROR", "service": "test"},
      {"message": "Batch 3", "level": "WARNING", "service": "test"}
    ]
  }')

if echo "$RESPONSE" | grep -q "inserted_count"; then
    COUNT=$(echo "$RESPONSE" | jq -r '.inserted_count')
    echo -e "${GREEN}✓ PASS${NC} (Inserted $COUNT logs)"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

test_endpoint "Query Logs" "$API/api/logs?limit=10"
test_endpoint "Search Logs" "$API/api/logs/search?query=test&limit=5"

# Phase 3 Tests
echo -e "\n${BLUE}${BOLD}Phase 3: Analytics & Alerts${NC}\n"

test_endpoint "Log Volume" "$API/api/analytics/volume?interval=1h"
test_endpoint "Error Rate" "$API/api/analytics/error-rate"
test_endpoint "Top Errors" "$API/api/analytics/top-errors?limit=5"
test_endpoint "Service Metrics" "$API/api/analytics/services"
test_endpoint "Response Time" "$API/api/analytics/response-time"
test_endpoint "Anomaly Summary" "$API/api/analytics/anomalies"
test_endpoint "Log Levels" "$API/api/analytics/log-levels"
test_endpoint "Dashboard" "$API/api/analytics/dashboard"

# Test alert rule creation
echo -n "Testing Alert Rule Creation... "
RESPONSE=$(curl -s -X POST "$API/api/alerts/rules" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Quick Test Rule",
    "description": "Test rule",
    "condition": {"metric": "error_rate", "operator": "gt", "threshold": 10},
    "severity": "WARNING",
    "enabled": true
  }')

if echo "$RESPONSE" | grep -q "rule_id"; then
    RULE_ID=$(echo "$RESPONSE" | jq -r '.rule_id')
    echo -e "${GREEN}✓ PASS${NC} (Rule ID: ${RULE_ID:0:8}...)"
    ((TESTS_PASSED++))
    
    # Clean up - delete the rule
    curl -s -X DELETE "$API/api/alerts/rules/$RULE_ID" > /dev/null
    echo "  └─ Cleaned up test rule"
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

test_endpoint "List Alert Rules" "$API/api/alerts/rules"
test_endpoint "List Alerts" "$API/api/alerts"
test_endpoint "Alert Statistics" "$API/api/alerts/statistics/summary"

# Database verification
echo -e "\n${BLUE}${BOLD}Database Verification${NC}\n"

echo -n "Checking MongoDB... "
if docker exec mongodb mongosh log_analytics --quiet --eval "db.logs.countDocuments()" > /dev/null 2>&1; then
    LOG_COUNT=$(docker exec mongodb mongosh log_analytics --quiet --eval "db.logs.countDocuments()")
    echo -e "${GREEN}✓ PASS${NC} (Logs in DB: $LOG_COUNT)"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

echo -n "Checking Redis... "
if docker exec redis redis-cli PING > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

# Summary
TOTAL=$((TESTS_PASSED + TESTS_FAILED))
echo -e "\n${BLUE}${BOLD}════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}${BOLD}                   TEST SUMMARY                     ${NC}"
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════${NC}\n"

echo -e "Total Tests:  ${BOLD}$TOTAL${NC}"
echo -e "Passed:       ${GREEN}${BOLD}$TESTS_PASSED${NC}"
echo -e "Failed:       ${RED}${BOLD}$TESTS_FAILED${NC}\n"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}${BOLD}🎉 ALL TESTS PASSED! 🎉${NC}\n"
    echo -e "${BLUE}Your AI Log Analytics system is fully operational!${NC}"
    echo -e "${BLUE}All Phase 1-3 components are working correctly.${NC}\n"
    
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  • View API docs: http://localhost:8000/docs"
    echo "  • Generate more data: python scripts/generate_logs.py"
    echo "  • Run full test suite: python scripts/test_all_phases.py"
    echo ""
    
    exit 0
else
    echo -e "${RED}${BOLD}❌ SOME TESTS FAILED${NC}\n"
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  • Check logs: docker-compose logs backend"
    echo "  • Verify services: docker-compose ps"
    echo "  • Restart: docker-compose restart"
    echo ""
    
    exit 1
fi