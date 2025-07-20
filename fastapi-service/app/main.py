#!/usr/bin/env python3
"""
FastAPI service for sentiment analysis using Triton Inference Server
Uses gRPC client for optimal performance with TensorRT model
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import asyncio
import logging
import time
import json
import numpy as np
from pathlib import Path

# Triton and ML imports
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="High-performance sentiment analysis using TensorRT and Triton",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SentimentRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze", example=["I love this product!", "This is terrible"])
    return_scores: bool = Field(True, description="Whether to return confidence scores")

class SentimentResult(BaseModel):
    text: str
    label: str
    score: float
    scores: Optional[Dict[str, float]] = None

class SentimentResponse(BaseModel):
    results: List[SentimentResult]
    processing_time_ms: float
    model_info: Dict[str, Union[str, int]]

class HealthResponse(BaseModel):
    status: str
    triton_server: str
    model_status: str
    uptime_seconds: float

# Global variables for the service
class SentimentService:
    def __init__(self):
        self.triton_client = None
        self.tokenizer = None
        self.model_name = "sentiment_analysis_tensorrt"
        self.triton_url = "triton-server:8001"  # Docker service name
        self.max_seq_length = 128
        self.max_batch_size = 4
        self.label_mapping = {
            0: "very_negative",
            1: "negative", 
            2: "neutral",
            3: "positive",
            4: "very_positive"
        }
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize the service components"""
        logger.info("🚀 Initializing Sentiment Analysis Service...")
        
        # Initialize Triton client
        await self._init_triton_client()
        
        # Initialize tokenizer
        await self._init_tokenizer()
        
        logger.info("✅ Service initialization completed")
        
    async def _init_triton_client(self):
        """Initialize Triton gRPC client"""
        try:
            logger.info(f"🔗 Connecting to Triton server at {self.triton_url}...")
            self.triton_client = grpcclient.InferenceServerClient(
                url=self.triton_url,
                verbose=False
            )
            
            # Check if server is ready
            if not self.triton_client.is_server_ready():
                raise Exception("Triton server is not ready")
            
            # Check if model is available
            if not self.triton_client.is_model_ready(self.model_name):
                raise Exception(f"Model {self.model_name} is not ready")
            
            logger.info("✅ Triton client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Triton client: {e}")
            raise
            
    async def _init_tokenizer(self):
        """Initialize the tokenizer"""
        try:
            logger.info("🔤 Loading tokenizer...")
            
            # Try to load tokenizer info from model repository
            tokenizer_info_path = Path("triton-model-repository/sentiment_analysis_tensorrt/tokenizer_info.json")
            if tokenizer_info_path.exists():
                with open(tokenizer_info_path) as f:
                    info = json.load(f)
                    model_name = info["model_name"]
                    self.max_seq_length = info["max_length"]
                    self.label_mapping = {int(k): v for k, v in info["label_mapping"].items()}
                    logger.info(f"📋 Loaded tokenizer info: {model_name}")
            else:
                model_name = "tabularisai/multilingual-sentiment-analysis"
                logger.info("📋 Using default tokenizer configuration")
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=True
                )
                logger.info("✅ Tokenizer loaded from cache")
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info("✅ Tokenizer downloaded and loaded")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize tokenizer: {e}")
            raise
    
    async def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for a batch of texts"""
        if not texts:
            return []
            
        start_time = time.time()
        
        try:
            # Tokenize all texts
            batch_size = len(texts)
            if batch_size > self.max_batch_size:
                # Process in chunks if batch is too large
                results = []
                for i in range(0, batch_size, self.max_batch_size):
                    chunk = texts[i:i + self.max_batch_size]
                    chunk_results = await self._predict_chunk(chunk)
                    results.extend(chunk_results)
                return results
            else:
                return await self._predict_chunk(texts)
                
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def _predict_chunk(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for a chunk of texts"""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np"
        )
        
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)
        
        # Prepare Triton inputs
        inputs = [
            grpcclient.InferInput("input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)),
            grpcclient.InferInput("attention_mask", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype))
        ]
        
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        
        # Prepare outputs
        outputs = [
            grpcclient.InferRequestedOutput("output")
        ]
        
        # Run inference
        response = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        # Get results
        logits = response.as_numpy("output")
        
        # Convert to probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Format results
        results = []
        for i, text in enumerate(texts):
            probs = probabilities[i]
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
            
            # Create score dictionary
            scores = {
                self.label_mapping[j]: float(probs[j]) 
                for j in range(len(self.label_mapping))
            }
            
            results.append({
                "text": text,
                "label": self.label_mapping[predicted_class],
                "score": confidence,
                "scores": scores
            })
        
        return results

# Initialize service
sentiment_service = SentimentService()

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup"""
    await sentiment_service.initialize()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Sentiment Analysis API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check Triton server
        triton_status = "healthy" if sentiment_service.triton_client and sentiment_service.triton_client.is_server_ready() else "unhealthy"
        
        # Check model
        model_status = "ready" if sentiment_service.triton_client and sentiment_service.triton_client.is_model_ready(sentiment_service.model_name) else "not_ready"
        
        return HealthResponse(
            status="healthy" if triton_status == "healthy" and model_status == "ready" else "unhealthy",
            triton_server=triton_status,
            model_status=model_status,
            uptime_seconds=time.time() - sentiment_service.start_time
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            triton_server="error",
            model_status="error",
            uptime_seconds=time.time() - sentiment_service.start_time
        )

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """Predict sentiment for given texts"""
    start_time = time.time()
    
    try:
        # Validate input
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(request.texts) > 50:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Too many texts (max 50)")
        
        # Get predictions
        predictions = await sentiment_service.predict_batch(request.texts)
        
        # Format response
        results = []
        for pred in predictions:
            result = SentimentResult(
                text=pred["text"],
                label=pred["label"], 
                score=pred["score"],
                scores=pred["scores"] if request.return_scores else None
            )
            results.append(result)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SentimentResponse(
            results=results,
            processing_time_ms=processing_time,
            model_info={
                "model_name": sentiment_service.model_name,
                "max_sequence_length": sentiment_service.max_seq_length,
                "num_classes": len(sentiment_service.label_mapping),
                "backend": "tensorrt"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/single")
async def predict_single(text: str, return_scores: bool = True):
    """Predict sentiment for a single text (simple endpoint)"""
    request = SentimentRequest(texts=[text], return_scores=return_scores)
    response = await predict_sentiment(request)
    return response.results[0] if response.results else None

@app.get("/model/info")
async def model_info():
    """Get model information"""
    try:
        # Get model metadata from Triton
        metadata = sentiment_service.triton_client.get_model_metadata(sentiment_service.model_name)
        
        return {
            "model_name": sentiment_service.model_name,
            "platform": metadata.platform,
            "backend": "tensorrt",
            "max_batch_size": sentiment_service.max_batch_size,
            "max_sequence_length": sentiment_service.max_seq_length,
            "input_names": [inp.name for inp in metadata.inputs],
            "output_names": [out.name for out in metadata.outputs],
            "label_mapping": sentiment_service.label_mapping,
            "tokenizer": "tabularisai/multilingual-sentiment-analysis"
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    ) 