.PHONY: help setup start stop restart logs clean test

# Colors for output
BLUE=\033[0;34m
GREEN=\033[0;32m
RED=\033[0;31m
NC=\033[0m # No Color

help: ## Show this help message
	@echo '$(BLUE)AI Log Analytics - Available Commands$(NC)'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Initial setup - create env files and directories
	@echo "$(BLUE)Setting up project...$(NC)"
	@mkdir -p data/raw data/processed data/sample_logs
	@mkdir -p ml/saved_models
	@mkdir -p logs
	@touch data/raw/.gitkeep data/processed/.gitkeep ml/saved_models/.gitkeep
	@if [ ! -f backend/.env ]; then cp backend/.env.example backend/.env; fi
	@if [ ! -f frontend/.env ]; then cp frontend/.env.example frontend/.env; fi
	@echo "$(GREEN)✓ Setup complete!$(NC)"

build: ## Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Build complete!$(NC)"

start: ## Start all services
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started!$(NC)"
	@echo "Dashboard: http://localhost:3000"
	@echo "API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

stop: ## Stop all services
	@echo "$(BLUE)Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped!$(NC)"

restart: ## Restart all services
	@echo "$(BLUE)Restarting services...$(NC)"
	docker-compose restart
	@echo "$(GREEN)✓ Services restarted!$(NC)"

logs: ## Show logs from all services
	docker-compose logs -f

logs-backend: ## Show backend logs
	docker-compose logs -f backend

logs-ml: ## Show ML service logs
	docker-compose logs -f ml_service

logs-frontend: ## Show frontend logs
	docker-compose logs -f frontend

status: ## Show status of all services
	docker-compose ps

shell-backend: ## Open shell in backend container
	docker-compose exec backend /bin/bash

shell-ml: ## Open shell in ML container
	docker-compose exec ml_service /bin/bash

init-db: ## Initialize database
	@echo "$(BLUE)Initializing database...$(NC)"
	docker-compose exec backend python scripts/init_db.py
	@echo "$(GREEN)✓ Database initialized!$(NC)"

seed-data: ## Seed sample data
	@echo "$(BLUE)Seeding sample data...$(NC)"
	docker-compose exec backend python scripts/seed_data.py
	@echo "$(GREEN)✓ Data seeded!$(NC)"

generate-logs: ## Generate synthetic logs
	@echo "$(BLUE)Generating logs...$(NC)"
	docker-compose exec backend python scripts/generate_logs.py
	@echo "$(GREEN)✓ Logs generated!$(NC)"

train-model: ## Train ML model
	@echo "$(BLUE)Training model...$(NC)"
	docker-compose exec ml_service python training/train_isolation_forest.py
	@echo "$(GREEN)✓ Model trained!$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	docker-compose exec backend pytest
	@echo "$(GREEN)✓ Tests complete!$(NC)"

test-backend: ## Run backend tests
	docker-compose exec backend pytest

test-ml: ## Run ML tests
	docker-compose exec ml_service pytest

clean: ## Clean up containers and volumes
	@echo "$(RED)Warning: This will remove all containers and volumes!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		echo "$(GREEN)✓ Cleanup complete!$(NC)"; \
	fi

clean-data: ## Clean generated data
	@echo "$(RED)Cleaning data directories...$(NC)"
	rm -rf data/raw/* data/processed/*
	@echo "$(GREEN)✓ Data cleaned!$(NC)"

install-backend: ## Install backend dependencies locally
	cd backend && pip install -r requirements.txt

install-ml: ## Install ML dependencies locally
	cd ml && pip install -r requirements.txt

install-frontend: ## Install frontend dependencies locally
	cd frontend && npm install

dev-backend: ## Run backend in development mode (local)
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend: ## Run frontend in development mode (local)
	cd frontend && npm run dev

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	docker-compose exec backend black app/
	cd frontend && npm run format
	@echo "$(GREEN)✓ Code formatted!$(NC)"

lint: ## Lint code
	@echo "$(BLUE)Linting code...$(NC)"
	docker-compose exec backend flake8 app/
	cd frontend && npm run lint
	@echo "$(GREEN)✓ Linting complete!$(NC)"