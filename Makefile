# KTCD_Aug Makefile
# Convenient commands for development and deployment

.PHONY: help install setup start stop restart logs clean test lint format docker-build docker-up docker-down docker-logs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

help: ## Display this help message
	@echo "$(BLUE)KTCD_Aug - Available Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(YELLOW)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install Python dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

setup: ## Initial project setup
	@echo "$(BLUE)Setting up KTCD_Aug...$(NC)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✓ Created .env file$(NC)"; \
	else \
		echo "$(YELLOW)⚠ .env already exists$(NC)"; \
	fi
	@mkdir -p logs data/generated_blogs models/checkpoints
	@echo "$(GREEN)✓ Created directories$(NC)"
	@echo "$(GREEN)✓ Setup complete!$(NC)"
	@echo "$(YELLOW)⚠ Don't forget to edit .env with your configuration$(NC)"

##@ Development

start: ## Start the application (without Docker)
	@echo "$(BLUE)Starting KTCD_Aug...$(NC)"
	python nexus_app.py

start-ai: ## Start AI models server
	@echo "$(BLUE)Starting AI models server...$(NC)"
	python ai_models_server.py

dev: ## Start in development mode with auto-reload
	@echo "$(BLUE)Starting in development mode...$(NC)"
	FLASK_ENV=development FLASK_DEBUG=1 python nexus_app.py

##@ Docker

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Images built$(NC)"

docker-up: ## Start all services with Docker
	@echo "$(BLUE)Starting Docker services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "$(YELLOW)Access the application at: http://localhost:8080$(NC)"
	@echo "$(YELLOW)Neo4j Browser at: http://localhost:7474$(NC)"
	@echo "$(YELLOW)AI Models API at: http://localhost:5000$(NC)"

docker-down: ## Stop all Docker services
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

docker-restart: docker-down docker-up ## Restart all Docker services

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-ps: ## Show running containers
	docker-compose ps

docker-clean: ## Remove all containers, volumes, and images
	@echo "$(RED)⚠ This will remove all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v --rmi all; \
		echo "$(GREEN)✓ Cleaned up$(NC)"; \
	fi

##@ Database

db-start: ## Start only Neo4j database
	@echo "$(BLUE)Starting Neo4j...$(NC)"
	docker-compose up -d neo4j
	@echo "$(GREEN)✓ Neo4j started$(NC)"

db-stop: ## Stop Neo4j database
	@echo "$(BLUE)Stopping Neo4j...$(NC)"
	docker-compose stop neo4j
	@echo "$(GREEN)✓ Neo4j stopped$(NC)"

db-shell: ## Open Neo4j Cypher shell
	docker exec -it ktcd_neo4j cypher-shell -u neo4j -p ktcd_password123

db-backup: ## Backup Neo4j database
	@echo "$(BLUE)Backing up database...$(NC)"
	@mkdir -p backups
	docker exec ktcd_neo4j neo4j-admin database dump neo4j --to-path=/backups
	docker cp ktcd_neo4j:/backups ./backups/
	@echo "$(GREEN)✓ Backup created in ./backups/$(NC)"

##@ Testing

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=services --cov-report=html --cov-report=term

benchmark: ## Run AI model benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	python utilities/benchmark_algorithms.py

verify: ## Verify system pipelines
	@echo "$(BLUE)Verifying pipelines...$(NC)"
	python utilities/verify_and_test_pipelines.py

##@ Code Quality

lint: ## Run linting
	@echo "$(BLUE)Running linters...$(NC)"
	flake8 services/ routes/ --max-line-length=100
	pylint services/ routes/

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	black services/ routes/ utilities/
	@echo "$(GREEN)✓ Code formatted$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checker...$(NC)"
	mypy services/ routes/

##@ Utilities

visualize: ## Visualize knowledge graph
	@echo "$(BLUE)Visualizing knowledge graph...$(NC)"
	python utilities/visualize_knowledge_graph.py

cleanup: ## Clean up knowledge graph
	@echo "$(BLUE)Cleaning up knowledge graph...$(NC)"
	python utilities/cleanup_knowledge_graph.py

demo: ## Set up demo system
	@echo "$(BLUE)Setting up demo system...$(NC)"
	python utilities/setup_demo_system.py

##@ Maintenance

clean: ## Clean temporary files
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache .coverage htmlcov/
	@echo "$(GREEN)✓ Cleaned$(NC)"

logs-clean: ## Clean log files
	@echo "$(BLUE)Cleaning logs...$(NC)"
	rm -rf logs/*.log
	@echo "$(GREEN)✓ Logs cleaned$(NC)"

##@ Information

status: ## Show system status
	@echo "$(BLUE)System Status:$(NC)"
	@echo ""
	@echo "$(YELLOW)Docker Services:$(NC)"
	@docker-compose ps 2>/dev/null || echo "  Not running"
	@echo ""
	@echo "$(YELLOW)Python Version:$(NC)"
	@python --version
	@echo ""
	@echo "$(YELLOW)Virtual Environment:$(NC)"
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo "  Active: $$VIRTUAL_ENV"; \
	else \
		echo "  Not activated"; \
	fi

version: ## Show version information
	@echo "$(BLUE)KTCD_Aug Version Information$(NC)"
	@echo "Version: 4.0.0"
	@echo "Python: $$(python --version)"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not installed')"

##@ Documentation

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "$(YELLOW)Documentation is in docs/ folder$(NC)"
	@ls -la docs/

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	@echo "$(YELLOW)Opening docs in browser...$(NC)"
	@open docs/ULTIMATE_PROJECT_SUMMARY.md || xdg-open docs/ULTIMATE_PROJECT_SUMMARY.md

##@ Git

git-status: ## Show git status
	@git status

git-clean: ## Clean untracked files (dry run)
	@echo "$(YELLOW)Files that would be removed:$(NC)"
	@git clean -n -d

git-clean-force: ## Clean untracked files (DANGEROUS!)
	@echo "$(RED)⚠ This will delete untracked files!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		git clean -f -d; \
		echo "$(GREEN)✓ Cleaned$(NC)"; \
	fi

##@ Quick Commands

all: setup install docker-build ## Complete setup (setup + install + build)
	@echo "$(GREEN)✓ Complete setup finished!$(NC)"

run: docker-up ## Quick start with Docker
	@echo "$(GREEN)✓ Application running!$(NC)"

stop: docker-down ## Quick stop

restart: docker-restart ## Quick restart

update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	pip install --upgrade -r requirements.txt
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

