# GPU-Accelerated Sentiment Analysis Service Makefile

.PHONY: help setup build start stop restart logs test clean convert-model

# Default target
help:
	@echo "=== 🤖 GPU-Accelerated Sentiment Analysis Service ==="
	@echo ""
	@echo "Available commands:"
	@echo "  setup         - Setup the development environment"
	@echo "  convert-model - Convert a model to TensorRT format"
	@echo "  build         - Build Docker images"
	@echo "  start         - Start all services"
	@echo "  stop          - Stop all services"
	@echo "  restart       - Restart all services"
	@echo "  logs          - View service logs"
	@echo "  test          - Run API tests"
	@echo "  clean         - Clean up containers and images"
	@echo "  status        - Show service status"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make convert-model MODEL=bert-base-uncased"
	@echo "  make start"
	@echo "  make test"

# Setup development environment
setup:
	@echo "🔧 Setting up development environment..."
	@mkdir -p logs
	@mkdir -p triton-model-repository/sentiment_analysis_tensorrt/1
	@echo "✅ Setup complete!"

# Convert model to TensorRT
convert-model:
	@echo "🔄 Converting model to TensorRT..."
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ Please specify MODEL parameter:"; \
		echo "   make convert-model MODEL=bert-base-uncased"; \
		exit 1; \
	fi
	cd model-conversion && \
	pip install -r requirements.txt && \
	python convert_to_tensorrt.py \
		--input "$(MODEL)" \
		--output "../triton-model-repository/sentiment_analysis_tensorrt/1/model.plan" \
		--type pytorch \
		--precision fp16 \
		--max-batch-size 32 \
		--validate
	@echo "✅ Model conversion complete!"

# Build Docker images
build:
	@echo "🏗️ Building Docker images..."
	docker compose build --no-cache
	@echo "✅ Build complete!"

# Start all services
start:
	@echo "🚀 Starting services..."
	@bash scripts/start_services.sh
	@echo "✅ Services started!"

# Stop all services
stop:
	@echo "🛑 Stopping services..."
	docker compose down
	@echo "✅ Services stopped!"

# Restart all services
restart: stop start

# View logs
logs:
	@echo "📋 Viewing service logs..."
	docker compose logs -f

# View logs for specific service
logs-triton:
	@echo "📋 Viewing Triton server logs..."
	docker compose logs -f triton-server

logs-api:
	@echo "📋 Viewing FastAPI service logs..."
	docker compose logs -f fastapi-service

# Run API tests
test:
	@echo "🧪 Running API tests..."
	@bash scripts/test_api.sh

# Show service status
status:
	@echo "📊 Service Status:"
	@docker compose ps
	@echo ""
	@echo "🔗 Service Endpoints:"
	@echo "  • FastAPI: http://localhost:8080"
	@echo "  • API Docs: http://localhost:8080/docs"
	@echo "  • Triton: http://localhost:8000"
	@echo "  • Health: http://localhost:8080/health"

# Clean up containers and images
clean:
	@echo "🧹 Cleaning up..."
	docker compose down --remove-orphans --volumes
	docker system prune -f
	@echo "✅ Cleanup complete!"

# Development helpers
dev-setup:
	@echo "🔧 Setting up development environment..."
	@make setup
	cd model-conversion && pip install -r requirements.txt
	cd fastapi-service && pip install -r requirements.txt
	@echo "✅ Development setup complete!"

# Quick health check
health:
	@echo "🏥 Checking service health..."
	@curl -s http://localhost:8080/health 2>/dev/null | jq . || echo "❌ FastAPI service not responding"
	@curl -s http://localhost:8000/v2/health/ready 2>/dev/null && echo "✅ Triton server healthy" || echo "❌ Triton server not responding"

# Performance test
perf:
	@echo "⚡ Running performance test..."
	@bash scripts/test_api.sh | grep -E "(benchmark|req/s|texts/s)"

# Docker image sizes
image-sizes:
	@echo "📦 Docker image sizes:"
	@docker images | grep -E "(tritonserver|sentiment|fastapi)" | awk '{print $$1":"$$2" - "$$7}'

# Model info
model-info:
	@echo "📋 Model information:"
	@curl -s http://localhost:8080/models 2>/dev/null | jq . || echo "❌ Could not retrieve model info"

# Metrics
metrics:
	@echo "📊 Service metrics:"
	@curl -s http://localhost:8080/metrics 2>/dev/null | grep -E "sentiment_(requests|errors|duration)" || echo "❌ Could not retrieve metrics"

# Debug commands
debug-triton:
	@echo "🔍 Triton server debug info:"
	@docker compose exec triton-server ls -la /models
	@docker compose exec triton-server curl -s localhost:8000/v2/models

debug-api:
	@echo "🔍 FastAPI debug info:"
	@docker compose exec fastapi-service ps aux
	@docker compose exec fastapi-service curl -s localhost:8080/health 