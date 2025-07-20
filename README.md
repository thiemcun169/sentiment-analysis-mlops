# GPU-Accelerated Sentiment Analysis Service

A high-performance sentiment analysis service using NVIDIA Triton Inference Server with TensorRT backend for optimal GPU acceleration.

## 🚀 Features

- **GPU-Optimized**: TensorRT backend for maximum inference speed
- **Scalable**: Dynamic batching and concurrent request handling
- **Production-Ready**: Docker containerization with health checks
- **Easy Management**: Simple model conversion and deployment tools
- **Monitoring**: Prometheus metrics and comprehensive logging
- **REST API**: FastAPI with automatic documentation

## 📋 Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Runtime
- Docker Compose
- Python 3.9+ (for model conversion)

### NVIDIA Container Runtime Setup

```bash
# Install NVIDIA Container Runtime
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime

# Restart Docker
sudo systemctl restart docker
```

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
git clone <your-repository>
cd mlops_docker
```

### 2. Convert Your Model to TensorRT

```bash
# Install conversion dependencies
cd model-conversion
pip install -r requirements.txt

# Convert a Hugging Face model (example)
python convert_to_tensorrt.py \
    --input "bert-base-uncased" \
    --output "../triton-model-repository/sentiment_analysis_tensorrt/1/model.plan" \
    --type pytorch \
    --precision fp16 \
    --max-batch-size 32 \
    --validate

# Or use the example script
bash convert_example.sh
```

### 3. Start the Services

```bash
# Start Triton server and FastAPI service
docker compose up -d

# Check service health
curl http://localhost:8080/health
```

### 4. Test the API

```bash
# Single text analysis
curl -X POST "http://localhost:8080/analyze/single" \
    -H "Content-Type: application/json" \
    -d '"This movie is amazing!"'

# Batch analysis
curl -X POST "http://localhost:8080/analyze" \
    -H "Content-Type: application/json" \
    -d '{
        "texts": [
            "I love this product!",
            "This is terrible.",
            "It could be better."
        ]
    }'
```

## 📁 Project Structure

```
├── docker-compose.yml              # Docker services configuration
├── fastapi-service/               # FastAPI application
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py               # FastAPI application
│       ├── triton_client.py      # Triton client wrapper
│       └── utils.py              # Utility functions
├── triton-model-repository/       # Triton model repository
│   └── sentiment_analysis_tensorrt/
│       ├── config.pbtxt          # Model configuration
│       └── 1/                    # Model version 1
│           └── model.plan        # TensorRT engine (after conversion)
└── model-conversion/              # Model conversion tools
    ├── convert_to_tensorrt.py    # Conversion script
    ├── convert_example.sh        # Example usage
    └── requirements.txt          # Conversion dependencies
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_SERVER_URL` | `triton-server:8001` | Triton server gRPC endpoint |
| `MODEL_NAME` | `sentiment_analysis_tensorrt` | Model name in Triton |
| `MAX_BATCH_SIZE` | `32` | Maximum batch size for requests |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device to use |

### Model Configuration

Edit `triton-model-repository/sentiment_analysis_tensorrt/config.pbtxt` to customize:

- Input/output shapes
- Batch sizes
- GPU allocation
- Dynamic batching settings

## 📊 API Documentation

### Endpoints

- **GET** `/` - Service information
- **GET** `/health` - Health check
- **POST** `/analyze` - Batch sentiment analysis
- **POST** `/analyze/single` - Single text analysis
- **GET** `/models` - Model information
- **GET** `/metrics` - Prometheus metrics

### API Documentation

Access interactive API docs at: `http://localhost:8080/docs`

### Example Response

```json
{
    "results": [
        {
            "text": "I love this product!",
            "sentiment": "positive",
            "confidence": 0.9234,
            "probabilities": {
                "negative": 0.0766,
                "positive": 0.9234
            }
        }
    ],
    "model_name": "sentiment_analysis_tensorrt",
    "model_version": "1",
    "processing_time": 0.045
}
```

## 🎯 Model Conversion Guide

### Convert PyTorch Model

```bash
python convert_to_tensorrt.py \
    --input "path/to/your/model" \
    --output "triton-model-repository/sentiment_analysis_tensorrt/1/model.plan" \
    --type pytorch \
    --precision fp16 \
    --max-batch-size 32 \
    --max-seq-length 512 \
    --num-labels 2 \
    --validate
```

### Convert ONNX Model

```bash
python convert_to_tensorrt.py \
    --input "model.onnx" \
    --output "triton-model-repository/sentiment_analysis_tensorrt/1/model.plan" \
    --type onnx \
    --precision fp16 \
    --validate
```

### Precision Options

- **fp32**: Full precision (largest model, highest accuracy)
- **fp16**: Half precision (recommended balance)
- **int8**: 8-bit quantization (smallest, requires calibration)

## 📈 Performance Optimization

### TensorRT Optimizations

1. **Precision**: Use FP16 for 2x speedup with minimal accuracy loss
2. **Batch Size**: Optimize for your GPU memory and throughput needs
3. **Dynamic Batching**: Enabled by default for variable input sizes
4. **Model Warmup**: Configured to reduce cold start latency

### Scaling

- **Horizontal**: Deploy multiple FastAPI instances behind a load balancer
- **Vertical**: Increase GPU memory and use larger batch sizes
- **Multi-GPU**: Configure multiple Triton instances with different GPUs

## 🔍 Monitoring and Logging

### Prometheus Metrics

Access metrics at: `http://localhost:8080/metrics`

Key metrics:
- `sentiment_requests_total`: Total requests processed
- `sentiment_request_duration_seconds`: Request latency
- `sentiment_errors_total`: Error counts by type

### Logs

- **Service logs**: `./logs/sentiment_service.log`
- **Container logs**: `docker compose logs -f`

## 🚨 Troubleshooting

### Common Issues

1. **GPU not accessible**
   ```bash
   # Check NVIDIA runtime
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

2. **Model not loading**
   ```bash
   # Check Triton model repository
   docker compose exec triton-server ls -la /models
   ```

3. **Out of GPU memory**
   - Reduce `max_batch_size` in configuration
   - Use smaller precision (fp16 instead of fp32)

4. **Slow inference**
   - Verify TensorRT engine is being used
   - Check GPU utilization with `nvidia-smi`
   - Ensure dynamic batching is enabled

### Health Checks

```bash
# Service health
curl http://localhost:8080/health

# Triton server health
curl http://localhost:8000/v2/health/ready

# Model readiness
curl http://localhost:8000/v2/models/sentiment_analysis_tensorrt/ready
```

## 🔒 Security Considerations

- **API Keys**: Implement authentication for production use
- **Input Validation**: Text length and content validation included
- **Rate Limiting**: Consider adding rate limiting for public APIs
- **HTTPS**: Use reverse proxy with SSL/TLS in production

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 Additional Resources

- [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu) 