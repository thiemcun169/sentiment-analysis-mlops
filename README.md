# Sentiment Analysis with TensorRT and Triton

High-performance multilingual sentiment analysis service using NVIDIA TensorRT and Triton Inference Server, wrapped in a FastAPI service and deployed via Docker Compose.

## 🚀 Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- Python 3.10+ (for model conversion)
- At least 8GB GPU memory

### 1. Clone and Setup

```bash
git clone <repository>
cd mlops_docker
```

### 2. Convert Model to TensorRT

```bash
# Install conversion dependencies
pip install torch transformers tensorrt onnx

# Convert the model (takes 5-10 minutes)
python convert_to_tensorrt.py
```

### 3. Deploy with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Test the Service

```bash
# Quick stress test
python docker_stress_test.py --quick

# Full stress test
python docker_stress_test.py
```

## 📊 Performance Results

### TensorRT Direct Performance (localhost:8000)
- **Peak Throughput**: 1,929 req/s
- **Latency**: 0.9ms - 69ms (P95)
- **Success Rate**: 100%

### FastAPI Service Performance (localhost:8080)
- **Throughput**: ~500-800 req/s (depending on batch size)
- **Features**: Batch processing, validation, error handling
- **Endpoints**: `/predict`, `/predict/single`, `/health`

## 🛠️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Triton        │    │   TensorRT      │
│   Service       │───▶│   Server        │───▶│   Engine        │
│   (Port 8080)   │gRPC│   (Port 8001)   │    │   (.plan)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Components

1. **TensorRT Engine**: Optimized sentiment analysis model
   - Model: `tabularisai/multilingual-sentiment-analysis`
   - Format: TensorRT engine (.plan file)
   - Optimization: FP16 precision
   - Classes: 5 (very_negative, negative, neutral, positive, very_positive)

2. **Triton Inference Server**: Model serving platform
   - Backend: TensorRT
   - Batch size: 1-4
   - Dynamic batching enabled
   - GPU acceleration

3. **FastAPI Service**: REST API wrapper
   - Endpoints: `/predict`, `/predict/single`, `/health`, `/model/info`
   - Validation: Input validation and error handling
   - Communication: gRPC with Triton
   - Features: Batch processing, detailed responses

## 🔧 Configuration

### Model Configuration (`triton-model-repository/sentiment_analysis_tensorrt/config.pbtxt`)

```protobuf
name: "sentiment_analysis_tensorrt"
backend: "tensorrt"
max_batch_size: 4
platform: "tensorrt_plan"

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ 128 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ 128 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 5 ]
  }
]
```

### Docker Compose Services

- **triton-server**: NVIDIA Triton container with GPU access
- **fastapi-service**: Custom FastAPI container
- **Networks**: Custom bridge network for service communication

## 📡 API Usage

### Predict Multiple Texts

```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "I love this product!",
         "This is terrible quality."
       ],
       "return_scores": true
     }'
```

### Predict Single Text

```bash
curl -X POST "http://localhost:8080/predict/single?text=Great product!&return_scores=true"
```

### Health Check

```bash
curl http://localhost:8080/health
```

### Model Information

```bash
curl http://localhost:8080/model/info
```

## 📈 Performance Testing

### Available Test Scripts

1. **Direct Triton Testing**: `simple_stress_test.py`
   - Tests Triton server directly
   - Maximum performance measurement
   - Use for baseline performance

2. **Docker Compose Testing**: `docker_stress_test.py`
   - Tests complete FastAPI + Triton stack
   - Real-world performance measurement
   - Use for production validation

### Test Options

```bash
# Quick test (reduced load)
python docker_stress_test.py --quick

# Custom test parameters
python docker_stress_test.py \
  --light-requests 50 \
  --medium-requests 200 \
  --heavy-requests 500 \
  --heavy-concurrency 25

# Direct Triton test
python simple_stress_test.py --quick
```

## 🛡️ Production Considerations

### Security
- Add authentication/authorization
- Enable HTTPS/TLS
- Implement rate limiting
- Add input sanitization

### Monitoring
- Add metrics collection (Prometheus)
- Implement logging aggregation
- Set up health checks
- Monitor GPU utilization

### Scaling
- Use multiple Triton instances
- Implement load balancing
- Add horizontal pod autoscaling (Kubernetes)
- Consider model versioning

## 🔍 Troubleshooting

### Common Issues

1. **TensorRT Version Mismatch**
   ```bash
   # Check TensorRT version
   python -c "import tensorrt; print(tensorrt.__version__)"
   
   # Must match Triton container (10.0.1)
   pip install tensorrt==10.0.1
   ```

2. **Model Not Loading**
   ```bash
   # Check model files
   ls -la triton-model-repository/sentiment_analysis_tensorrt/
   
   # Check Triton logs
   docker-compose logs triton-server
   ```

3. **FastAPI Connection Issues**
   ```bash
   # Check network connectivity
   docker-compose exec fastapi-service ping triton
   
   # Check service logs
   docker-compose logs fastapi-service
   ```

### Performance Optimization

1. **GPU Memory**: Ensure sufficient GPU memory (8GB+ recommended)
2. **Batch Size**: Adjust max_batch_size in model config
3. **Concurrency**: Tune FastAPI worker count
4. **Network**: Use dedicated network for high throughput

## 📁 Project Structure

```
mlops_docker/
├── convert_to_tensorrt.py         # Model conversion script
├── docker-compose.yml             # Docker Compose configuration
├── simple_stress_test.py          # Direct Triton stress test
├── docker_stress_test.py          # Docker Compose stress test
├── fastapi-service/               # FastAPI service
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       └── main.py               # FastAPI application
├── triton-model-repository/       # Triton model repository
│   └── sentiment_analysis_tensorrt/
│       ├── config.pbtxt          # Model configuration
│       ├── tokenizer_info.json   # Tokenizer metadata
│       └── 1/
│           └── model.plan        # TensorRT engine
└── logs/                         # Application logs
```

## 🎯 Use Cases

### Real-time Sentiment Analysis
- Social media monitoring
- Customer feedback analysis
- Product review processing
- Chat/comment moderation

### Batch Processing
- Large-scale text analysis
- Historical data processing
- Report generation
- Data pipeline integration

## 📊 Benchmarks

### Hardware Used
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CPU: Intel i9-13900K
- RAM: 64GB DDR5

### Results Summary
- **Best Throughput**: 1,929 req/s (Direct Triton)
- **Production Throughput**: ~600 req/s (FastAPI + Triton)
- **Latency P95**: <70ms
- **GPU Utilization**: ~30-50% under load
- **Memory Usage**: ~2-3GB VRAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- NVIDIA Triton Inference Server team
- HuggingFace Transformers team
- FastAPI development team
- `tabularisai/multilingual-sentiment-analysis` model authors