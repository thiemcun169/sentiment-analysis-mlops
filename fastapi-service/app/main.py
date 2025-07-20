import os
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

from .triton_client import TritonSentimentClient
from .utils import setup_logging, health_check

# Setup logging
setup_logging()

# Metrics
REQUEST_COUNT = Counter('sentiment_requests_total', 'Total sentiment analysis requests')
REQUEST_DURATION = Histogram('sentiment_request_duration_seconds', 'Request duration')
ERROR_COUNT = Counter('sentiment_errors_total', 'Total errors', ['error_type'])


class SentimentRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    texts: List[str] = Field(..., min_items=1, max_items=32, description="List of texts to analyze")
    model_version: Optional[str] = Field(default="1", description="Model version to use")


class SentimentResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    results: List[Dict[str, Any]] = Field(..., description="Sentiment analysis results")
    model_name: str = Field(..., description="Model used for inference")
    model_version: str = Field(..., description="Model version used")
    processing_time: float = Field(..., description="Processing time in seconds")


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: str
    triton_server: str
    model_ready: bool
    uptime: float


# Global client instance
triton_client: Optional[TritonSentimentClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global triton_client
    
    # Startup
    logger.info("Starting Sentiment Analysis Service...")
    
    triton_url = os.getenv("TRITON_SERVER_URL", "localhost:8001")
    model_name = os.getenv("MODEL_NAME", "sentiment_analysis_tensorrt")
    max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "32"))
    
    triton_client = TritonSentimentClient(
        triton_url=triton_url,
        model_name=model_name,
        max_batch_size=max_batch_size
    )
    
    # Initialize Triton connection (non-blocking)
    try:
        await triton_client.connect()
        logger.info("Triton client initialized - server connection will be verified on first request")
    except Exception as e:
        logger.warning(f"Could not initialize Triton client: {e}")
        logger.info("Service will start without Triton connection")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if triton_client:
        await triton_client.close()


app = FastAPI(
    title="Sentiment Analysis Service",
    description="High-performance sentiment analysis using NVIDIA Triton Inference Server with TensorRT",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    global triton_client
    
    if not triton_client:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        server_ready = await triton_client.is_server_ready()
        model_ready = await triton_client.is_model_ready()
        
        return HealthResponse(
            status="healthy" if server_ready and model_ready else "unhealthy",
            triton_server="ready" if server_ready else "not_ready",
            model_ready=model_ready,
            uptime=triton_client.uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")


@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of input texts"""
    global triton_client
    
    if not triton_client:
        ERROR_COUNT.labels(error_type="service_not_ready").inc()
        raise HTTPException(status_code=503, detail="Service not ready")
    
    REQUEST_COUNT.inc()
    
    try:
        with REQUEST_DURATION.time():
            results, processing_time = await triton_client.predict(
                texts=request.texts,
                model_version=request.model_version
            )
        
        return SentimentResponse(
            results=results,
            model_name=triton_client.model_name,
            model_version=request.model_version,
            processing_time=processing_time
        )
        
    except Exception as e:
        ERROR_COUNT.labels(error_type="prediction_error").inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/analyze/single")
async def analyze_single_text(text: str, model_version: Optional[str] = "1"):
    """Analyze sentiment of a single text (convenience endpoint)"""
    request = SentimentRequest(texts=[text], model_version=model_version)
    response = await analyze_sentiment(request)
    return response.results[0] if response.results else None


@app.get("/models")
async def get_model_info():
    """Get information about available models"""
    global triton_client
    
    if not triton_client:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        model_info = await triton_client.get_model_metadata()
        return model_info
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Sentiment Analysis API",
        "description": "High-performance sentiment analysis using NVIDIA Triton with TensorRT",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze",
            "analyze_single": "/analyze/single",
            "health": "/health",
            "models": "/models",
            "metrics": "/metrics",
            "simple": "/simple"
        }
    }


@app.get("/simple")
async def simple_health():
    """Simple health check without Triton dependencies"""
    return {"status": "ok", "message": "FastAPI service is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 