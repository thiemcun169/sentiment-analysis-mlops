#!/bin/bash

# Start GPU-accelerated sentiment analysis services
# This script handles the complete deployment process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="sentiment-analysis"
COMPOSE_FILE="docker-compose.yml"
DOCKER_COMPOSE="docker compose"
LOG_FILE="deployment.log"

# Functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker runtime
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        error "NVIDIA Docker runtime is not properly configured"
        error "Please install nvidia-container-runtime and restart Docker"
        exit 1
    fi
    
    # Check if model file exists
    MODEL_FILE="triton-model-repository/sentiment_analysis_tensorrt/1/model.plan"
    if [ ! -f "$MODEL_FILE" ]; then
        warn "TensorRT model file not found at $MODEL_FILE"
        warn "Please run model conversion first:"
        warn "  cd model-conversion && bash convert_example.sh"
    fi
    
    success "Prerequisites check completed"
}

cleanup_containers() {
    log "Cleaning up existing containers..."
    
    # Stop and remove existing containers
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    
    # Remove orphaned volumes if any
    docker volume prune -f 2>/dev/null || true
    
    success "Cleanup completed"
}

build_services() {
    log "Building services..."
    
    # Build FastAPI service
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" build --no-cache fastapi-service
    
    success "Services built successfully"
}

start_services() {
    log "Starting services..."
    
    # Start Triton server first
    log "Starting Triton Inference Server..."
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d triton-server
    
    # Wait for Triton to be ready
    log "Waiting for Triton server to be ready..."
    timeout=120
    counter=0
    
    while [ $counter -lt $timeout ]; do
        if curl -s -f http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
            success "Triton server is ready"
            break
        fi
        
        if [ $((counter % 10)) -eq 0 ]; then
            log "Still waiting for Triton server... (${counter}s elapsed)"
        fi
        
        sleep 2
        counter=$((counter + 2))
    done
    
    if [ $counter -ge $timeout ]; then
        error "Triton server failed to start within ${timeout} seconds"
        $DOCKER_COMPOSE -f "$COMPOSE_FILE" logs triton-server
        exit 1
    fi
    
    # Start FastAPI service
    log "Starting FastAPI service..."
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d fastapi-service
    
    # Wait for FastAPI to be ready
    log "Waiting for FastAPI service to be ready..."
    timeout=60
    counter=0
    
    while [ $counter -lt $timeout ]; do
        if curl -s -f http://localhost:8080/health > /dev/null 2>&1; then
            success "FastAPI service is ready"
            break
        fi
        
        if [ $((counter % 10)) -eq 0 ]; then
            log "Still waiting for FastAPI service... (${counter}s elapsed)"
        fi
        
        sleep 2
        counter=$((counter + 2))
    done
    
    if [ $counter -ge $timeout ]; then
        error "FastAPI service failed to start within ${timeout} seconds"
        $DOCKER_COMPOSE -f "$COMPOSE_FILE" logs fastapi-service
        exit 1
    fi
    
    success "All services started successfully"
}

verify_deployment() {
    log "Verifying deployment..."
    
    # Test Triton server
    log "Testing Triton server..."
    if curl -s -f http://localhost:8000/v2/health/ready > /dev/null; then
        success "Triton server health check passed"
    else
        error "Triton server health check failed"
        return 1
    fi
    
    # Test FastAPI service
    log "Testing FastAPI service..."
    if curl -s -f http://localhost:8080/health > /dev/null; then
        success "FastAPI service health check passed"
    else
        error "FastAPI service health check failed"
        return 1
    fi
    
    # Test sentiment analysis endpoint
    log "Testing sentiment analysis endpoint..."
    response=$(curl -s -X POST "http://localhost:8080/analyze/single" \
        -H "Content-Type: application/json" \
        -d '"This is a test message"' || echo "failed")
    
    if [[ "$response" != "failed" && "$response" != *"error"* ]]; then
        success "Sentiment analysis endpoint test passed"
    else
        warn "Sentiment analysis endpoint test failed - model might not be ready"
        warn "Response: $response"
    fi
    
    success "Deployment verification completed"
}

show_status() {
    echo ""
    echo "=== 🚀 Sentiment Analysis Service Status ==="
    echo ""
    
    # Service URLs
    echo "📡 Service Endpoints:"
    echo "  • FastAPI Service: http://localhost:8080"
    echo "  • API Documentation: http://localhost:8080/docs"
    echo "  • Triton Server: http://localhost:8000"
    echo "  • Health Check: http://localhost:8080/health"
    echo "  • Metrics: http://localhost:8080/metrics"
    echo ""
    
    # Container status
    echo "🐳 Container Status:"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" ps
    echo ""
    
    # Quick test
    echo "🧪 Quick Test:"
    echo 'curl -X POST "http://localhost:8080/analyze/single" \'
    echo '  -H "Content-Type: application/json" \'
    echo '  -d '"'"'"This is amazing!"'"'"''
    echo ""
    
    # Logs
    echo "📋 View Logs:"
    echo "  docker-compose logs -f                # All services"
    echo "  docker-compose logs -f triton-server  # Triton only"
    echo "  docker-compose logs -f fastapi-service # FastAPI only"
    echo ""
    
    # Stop command
    echo "🛑 Stop Services:"
    echo "  docker compose down"
    echo ""
}

# Main execution
main() {
    echo "=== 🤖 GPU-Accelerated Sentiment Analysis Deployment ==="
    echo ""
    
    # Parse command line arguments
    SKIP_CHECKS=false
    SKIP_BUILD=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-checks)
                SKIP_CHECKS=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-checks    Skip prerequisite checks"
                echo "  --skip-build     Skip service building"
                echo "  -h, --help       Show this help message"
                echo ""
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Initialize log file
    echo "Deployment started at $(date)" > "$LOG_FILE"
    
    # Execute deployment steps
    if [ "$SKIP_CHECKS" = false ]; then
        check_prerequisites
    fi
    
    cleanup_containers
    
    if [ "$SKIP_BUILD" = false ]; then
        build_services
    fi
    
    start_services
    verify_deployment
    show_status
    
    success "🎉 Deployment completed successfully!"
    log "Check the deployment log at: $LOG_FILE"
}

# Handle script interruption
trap 'error "Deployment interrupted"; exit 1' INT TERM

# Execute main function
main "$@" 