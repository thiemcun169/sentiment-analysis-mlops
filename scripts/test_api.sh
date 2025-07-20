#!/bin/bash

# Test script for the sentiment analysis API
# Tests all endpoints and provides performance benchmarks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
API_BASE_URL="http://localhost:8080"
TRITON_BASE_URL="http://localhost:8000"

# Test data
POSITIVE_TEXTS=(
    "I love this product!"
    "This is amazing and wonderful!"
    "Excellent service and great quality!"
    "Best purchase I've ever made!"
    "Highly recommended!"
)

NEGATIVE_TEXTS=(
    "This is terrible and disappointing."
    "Worst product ever!"
    "Complete waste of money."
    "I hate this so much."
    "Absolutely horrible experience."
)

NEUTRAL_TEXTS=(
    "This is a product."
    "It works as expected."
    "Average quality."
    "It's okay, nothing special."
    "Could be better."
)

# Functions
log() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test health endpoints
test_health() {
    log "Testing health endpoints..."
    
    # FastAPI health
    if curl -s -f "$API_BASE_URL/health" > /dev/null; then
        success "FastAPI health check passed"
    else
        fail "FastAPI health check failed"
        return 1
    fi
    
    # Triton health
    if curl -s -f "$TRITON_BASE_URL/v2/health/ready" > /dev/null; then
        success "Triton health check passed"
    else
        fail "Triton health check failed"
        return 1
    fi
    
    # Model readiness
    if curl -s -f "$TRITON_BASE_URL/v2/models/sentiment_analysis_tensorrt/ready" > /dev/null; then
        success "Model readiness check passed"
    else
        warn "Model readiness check failed - model may not be loaded"
    fi
}

# Test single text analysis
test_single_analysis() {
    log "Testing single text analysis..."
    
    local test_text="This is an amazing product!"
    local response
    
    response=$(curl -s -X POST "$API_BASE_URL/analyze/single" \
        -H "Content-Type: application/json" \
        -d "\"$test_text\"" 2>/dev/null)
    
    if [[ $? -eq 0 && "$response" != *"error"* && "$response" != *"Internal Server Error"* ]]; then
        success "Single text analysis passed"
        echo "  Response: $response"
    else
        fail "Single text analysis failed"
        echo "  Response: $response"
        return 1
    fi
}

# Test batch analysis
test_batch_analysis() {
    log "Testing batch analysis..."
    
    local batch_texts='["I love this!", "This is terrible.", "It works fine."]'
    local response
    
    response=$(curl -s -X POST "$API_BASE_URL/analyze" \
        -H "Content-Type: application/json" \
        -d "{\"texts\": $batch_texts}" 2>/dev/null)
    
    if [[ $? -eq 0 && "$response" != *"error"* && "$response" != *"Internal Server Error"* ]]; then
        success "Batch analysis passed"
        echo "  Processed $(echo "$response" | jq -r '.results | length' 2>/dev/null || echo "N/A") texts"
    else
        fail "Batch analysis failed"
        echo "  Response: $response"
        return 1
    fi
}

# Test model information
test_model_info() {
    log "Testing model information endpoint..."
    
    local response
    response=$(curl -s "$API_BASE_URL/models" 2>/dev/null)
    
    if [[ $? -eq 0 && "$response" != *"error"* ]]; then
        success "Model information endpoint passed"
        echo "  Model name: $(echo "$response" | jq -r '.name' 2>/dev/null || echo "N/A")"
        echo "  Platform: $(echo "$response" | jq -r '.platform' 2>/dev/null || echo "N/A")"
    else
        fail "Model information endpoint failed"
        echo "  Response: $response"
        return 1
    fi
}

# Test sentiment accuracy
test_sentiment_accuracy() {
    log "Testing sentiment accuracy with known examples..."
    
    local correct=0
    local total=0
    local response
    
    # Test positive examples
    for text in "${POSITIVE_TEXTS[@]}"; do
        response=$(curl -s -X POST "$API_BASE_URL/analyze/single" \
            -H "Content-Type: application/json" \
            -d "\"$text\"" 2>/dev/null)
        
        if [[ "$response" == *"positive"* ]]; then
            ((correct++))
        fi
        ((total++))
    done
    
    # Test negative examples
    for text in "${NEGATIVE_TEXTS[@]}"; do
        response=$(curl -s -X POST "$API_BASE_URL/analyze/single" \
            -H "Content-Type: application/json" \
            -d "\"$text\"" 2>/dev/null)
        
        if [[ "$response" == *"negative"* ]]; then
            ((correct++))
        fi
        ((total++))
    done
    
    local accuracy=$((correct * 100 / total))
    
    if [ $accuracy -ge 70 ]; then
        success "Sentiment accuracy test passed: $accuracy% ($correct/$total)"
    else
        warn "Sentiment accuracy below threshold: $accuracy% ($correct/$total)"
    fi
}

# Performance benchmark
benchmark_performance() {
    log "Running performance benchmark..."
    
    local num_requests=10
    local batch_size=5
    local start_time
    local end_time
    local duration
    
    # Single request benchmark
    log "Benchmarking single requests (n=$num_requests)..."
    start_time=$(date +%s.%N)
    
    for i in $(seq 1 $num_requests); do
        curl -s -X POST "$API_BASE_URL/analyze/single" \
            -H "Content-Type: application/json" \
            -d '"Test message for benchmarking"' > /dev/null 2>&1
    done
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "N/A")
    
    if [[ "$duration" != "N/A" ]]; then
        local rps=$(echo "scale=2; $num_requests / $duration" | bc -l 2>/dev/null || echo "N/A")
        success "Single request benchmark: ${duration}s total, ${rps} req/s"
    else
        warn "Could not calculate benchmark metrics"
    fi
    
    # Batch request benchmark
    log "Benchmarking batch requests (n=$num_requests, batch_size=$batch_size)..."
    
    local batch_json
    batch_json=$(printf '"%s",' $(printf "Test message %d " $(seq 1 $batch_size)) | sed 's/,$//')
    
    start_time=$(date +%s.%N)
    
    for i in $(seq 1 $num_requests); do
        curl -s -X POST "$API_BASE_URL/analyze" \
            -H "Content-Type: application/json" \
            -d "{\"texts\": [$batch_json]}" > /dev/null 2>&1
    done
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "N/A")
    
    if [[ "$duration" != "N/A" ]]; then
        local total_texts=$((num_requests * batch_size))
        local tps=$(echo "scale=2; $total_texts / $duration" | bc -l 2>/dev/null || echo "N/A")
        success "Batch request benchmark: ${duration}s total, ${tps} texts/s"
    else
        warn "Could not calculate batch benchmark metrics"
    fi
}

# Test error handling
test_error_handling() {
    log "Testing error handling..."
    
    # Test empty input
    local response
    response=$(curl -s -w "%{http_code}" -X POST "$API_BASE_URL/analyze" \
        -H "Content-Type: application/json" \
        -d '{"texts": []}' 2>/dev/null)
    
    if [[ "$response" == *"422"* || "$response" == *"400"* ]]; then
        success "Empty input validation passed"
    else
        warn "Empty input validation may not be working"
    fi
    
    # Test oversized batch
    local large_batch
    large_batch=$(printf '"%s",' $(printf "text " $(seq 1 50)) | sed 's/,$//')
    
    response=$(curl -s -w "%{http_code}" -X POST "$API_BASE_URL/analyze" \
        -H "Content-Type: application/json" \
        -d "{\"texts\": [$large_batch]}" 2>/dev/null)
    
    if [[ "$response" == *"422"* || "$response" == *"400"* ]]; then
        success "Oversized batch validation passed"
    else
        warn "Oversized batch validation may not be working"
    fi
}

# Test metrics endpoint
test_metrics() {
    log "Testing metrics endpoint..."
    
    local response
    response=$(curl -s "$API_BASE_URL/metrics" 2>/dev/null)
    
    if [[ $? -eq 0 && "$response" == *"sentiment_requests_total"* ]]; then
        success "Metrics endpoint passed"
        
        # Extract some metrics
        local total_requests
        total_requests=$(echo "$response" | grep "sentiment_requests_total" | head -1 | awk '{print $2}' 2>/dev/null || echo "N/A")
        echo "  Total requests processed: $total_requests"
    else
        fail "Metrics endpoint failed"
        return 1
    fi
}

# Show test summary
show_summary() {
    echo ""
    echo "=== 📊 Test Summary ==="
    echo ""
    echo "🔗 Service Endpoints Tested:"
    echo "  • Health: $API_BASE_URL/health"
    echo "  • Single Analysis: $API_BASE_URL/analyze/single"
    echo "  • Batch Analysis: $API_BASE_URL/analyze"
    echo "  • Model Info: $API_BASE_URL/models"
    echo "  • Metrics: $API_BASE_URL/metrics"
    echo ""
    echo "📈 Additional Testing:"
    echo "  • Sentiment accuracy validation"
    echo "  • Performance benchmarking"
    echo "  • Error handling validation"
    echo ""
    echo "🎯 For detailed API documentation, visit:"
    echo "  $API_BASE_URL/docs"
    echo ""
}

# Main test execution
main() {
    echo "=== 🧪 Sentiment Analysis API Test Suite ==="
    echo ""
    
    local failed_tests=0
    
    # Check if jq is available for JSON parsing
    if ! command -v jq &> /dev/null; then
        warn "jq not found - JSON parsing will be limited"
    fi
    
    # Check if bc is available for calculations
    if ! command -v bc &> /dev/null; then
        warn "bc not found - performance calculations will be limited"
    fi
    
    # Run all tests
    test_health || ((failed_tests++))
    echo ""
    
    test_single_analysis || ((failed_tests++))
    echo ""
    
    test_batch_analysis || ((failed_tests++))
    echo ""
    
    test_model_info || ((failed_tests++))
    echo ""
    
    test_sentiment_accuracy
    echo ""
    
    benchmark_performance
    echo ""
    
    test_error_handling
    echo ""
    
    test_metrics || ((failed_tests++))
    echo ""
    
    show_summary
    
    if [ $failed_tests -eq 0 ]; then
        success "🎉 All critical tests passed!"
        return 0
    else
        fail "❌ $failed_tests critical tests failed"
        return 1
    fi
}

# Execute main function
main "$@" 