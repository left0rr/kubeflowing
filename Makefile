.PHONY: install test lint docker-up docker-down docker-logs clean help

PYTHON      ?= python3
PIP         ?= pip3
VENV_DIR    ?= .venv
COMPOSE_FILE = infrastructure/docker-compose.yml

##@ General

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

install: ## Create virtualenv and install all dependencies
	@echo "--- Creating virtual environment in $(VENV_DIR) ---"
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "--- Installing dependencies ---"
	$(VENV_DIR)/bin/$(PIP) install --upgrade pip
	$(VENV_DIR)/bin/$(PIP) install -r requirements.txt
	@echo "--- Installation complete. Activate with: source $(VENV_DIR)/bin/activate ---"

lint: ## Run ruff linter and formatter checks
	@echo "--- Running ruff linter ---"
	$(VENV_DIR)/bin/ruff check . --fix
	@echo "--- Running ruff formatter ---"
	$(VENV_DIR)/bin/ruff format .
	@echo "--- Lint complete ---"

test: ## Run pytest test suite
	@echo "--- Running tests ---"
	$(VENV_DIR)/bin/pytest tests/ \
		--junitxml=reports/junit.xml \
		-v \
		--tb=short
	@echo "--- Tests complete ---"

##@ Infrastructure
docker-up: ## Start MLflow, PostgreSQL, and MinIO via Docker Compose
	@echo "--- Starting infrastructure stack ---"
	docker compose -f $(COMPOSE_FILE) up -d --build --remove-orphans
	@echo ""
	@echo "  Stack is up!"
	@echo "  MLflow UI  -> http://localhost:5000"
	@echo "  MinIO UI   -> http://localhost:9001  (minioadmin / minioadmin)"
	@echo ""

docker-down: ## Tear down the Docker Compose stack (keeps volumes)
	@echo "--- Stopping infrastructure stack ---"
	docker compose -f $(COMPOSE_FILE) down

docker-logs: ## Tail logs from all Docker Compose services
docker compose -f $(COMPOSE_FILE) logs -f

##@ Cleanup

clean: ## Remove virtualenv, caches, and build artifacts
	@echo "--- Cleaning project ---"
	rm -rf $(VENV_DIR)
	rm -rf .pytest_cache
	rm -rf reports/
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "--- Clean complete ---"