import os
import sys
import re
from typing import List, Dict, Any
from loguru import logger


def setup_logging():
    """Setup logging configuration"""
    # Remove default logger
    logger.remove()
    
    # Console logging
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # File logging
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger.add(
        f"{log_dir}/sentiment_service.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    logger.info("Logging setup completed")


def clean_text(text: str) -> str:
    """Clean and normalize text for sentiment analysis"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove or replace special characters that might cause issues
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\'\"]', ' ', text)
    
    # Limit length (will be further truncated by tokenizer)
    max_length = 1000
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Text truncated to {max_length} characters")
    
    return text


def preprocess_texts(texts: List[str]) -> List[str]:
    """Preprocess a list of texts for sentiment analysis"""
    cleaned_texts = []
    
    for i, text in enumerate(texts):
        try:
            cleaned_text = clean_text(text)
            if not cleaned_text or len(cleaned_text.strip()) == 0:
                cleaned_text = "empty text"
                logger.warning(f"Empty text at index {i}, using fallback")
            
            cleaned_texts.append(cleaned_text)
            
        except Exception as e:
            logger.error(f"Error preprocessing text at index {i}: {e}")
            cleaned_texts.append("error processing text")
    
    return cleaned_texts


def postprocess_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Postprocess sentiment analysis results"""
    processed = []
    
    for result in results:
        try:
            # Ensure required fields exist
            processed_result = {
                "text": result.get("text", ""),
                "sentiment": result.get("sentiment", "neutral"),
                "confidence": float(result.get("confidence", 0.5)),
                "probabilities": result.get("probabilities", {"negative": 0.5, "positive": 0.5})
            }
            
            # Validate confidence score
            if not 0 <= processed_result["confidence"] <= 1:
                processed_result["confidence"] = 0.5
                logger.warning("Invalid confidence score, using 0.5")
            
            # Validate sentiment
            if processed_result["sentiment"] not in ["positive", "negative", "neutral"]:
                processed_result["sentiment"] = "neutral"
                logger.warning("Invalid sentiment, using neutral")
            
            processed.append(processed_result)
            
        except Exception as e:
            logger.error(f"Error postprocessing result: {e}")
            processed.append({
                "text": result.get("text", ""),
                "sentiment": "neutral",
                "confidence": 0.5,
                "probabilities": {"negative": 0.5, "positive": 0.5}
            })
    
    return processed


def validate_text_input(text: str) -> bool:
    """Validate text input"""
    if not isinstance(text, str):
        return False
    
    # Check length
    if len(text.strip()) == 0 or len(text) > 10000:
        return False
    
    return True


def health_check() -> Dict[str, Any]:
    """Basic health check information"""
    return {
        "status": "healthy",
        "timestamp": __import__("time").time(),
        "python_version": sys.version,
        "process_id": os.getpid()
    }


def format_error_response(error: Exception, request_id: str = None) -> Dict[str, Any]:
    """Format error response"""
    return {
        "error": True,
        "message": str(error),
        "type": type(error).__name__,
        "request_id": request_id,
        "timestamp": __import__("time").time()
    }


def calculate_batch_size(texts: List[str], max_batch_size: int = 32) -> int:
    """Calculate optimal batch size based on input"""
    num_texts = len(texts)
    
    # Consider text length for dynamic batching
    avg_length = sum(len(text) for text in texts) / num_texts if texts else 0
    
    if avg_length > 500:
        # Reduce batch size for longer texts
        optimal_batch = min(max_batch_size // 2, num_texts)
    elif avg_length < 100:
        # Can handle more short texts
        optimal_batch = min(max_batch_size, num_texts)
    else:
        optimal_batch = min(max_batch_size, num_texts)
    
    return max(1, optimal_batch) 