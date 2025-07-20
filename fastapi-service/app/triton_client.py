import time
import asyncio
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException
from loguru import logger
from transformers import AutoTokenizer

from .utils import preprocess_texts, postprocess_results


class TritonSentimentClient:
    """Asynchronous Triton client for sentiment analysis with TensorRT backend"""
    
    def __init__(
        self,
        triton_url: str,
        model_name: str = "sentiment_analysis_tensorrt",
        max_batch_size: int = 32,
        timeout: float = 60.0
    ):
        self.triton_url = triton_url
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.client: Optional[grpcclient.InferenceServerClient] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.start_time = time.time()
        
        # Model configuration
        self.model_config = None
        self.input_names = []
        self.output_names = []
        self.max_sequence_length = 512
        
    @property
    def uptime(self) -> float:
        """Get service uptime in seconds"""
        return time.time() - self.start_time
    
    async def connect(self):
        """Connect to Triton server"""
        if self.client is None:
            try:
                self.client = grpcclient.InferenceServerClient(
                    url=self.triton_url,
                    verbose=False
                )
                logger.info(f"Connected to Triton server at {self.triton_url}")
                # Initialize tokenizer immediately
                await self._initialize_tokenizer()
            except Exception as e:
                logger.error(f"Failed to connect to Triton server: {e}")
                # Don't raise - allow service to start anyway
                self.client = None
    
    async def disconnect(self):
        """Disconnect from Triton server"""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Disconnected from Triton server")
    
    async def close(self):
        """Close the client"""
        await self.disconnect()
    
    async def wait_for_server_ready(self, timeout: float = 60.0):
        """Wait for Triton server to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                await self.connect()
                if await self.is_server_ready():
                    try:
                        await self._load_model_config()
                        await self._initialize_tokenizer()
                        logger.info("Triton server is ready and model loaded")
                    except Exception as e:
                        logger.warning(f"Model not available yet, but server is ready: {e}")
                        await self._initialize_tokenizer()  # Initialize tokenizer anyway
                        logger.info("Triton server is ready (model will be loaded later)")
                    return
            except Exception as e:
                logger.warning(f"Waiting for server: {e}")
                await asyncio.sleep(2)
        
        raise TimeoutError(f"Triton server not ready after {timeout} seconds")
    
    async def is_server_ready(self) -> bool:
        """Check if Triton server is ready"""
        try:
            if not self.client:
                await self.connect()
            return await self.client.is_server_ready()
        except Exception:
            return False
    
    async def is_model_ready(self, model_version: str = "1") -> bool:
        """Check if model is ready"""
        try:
            if not self.client:
                await self.connect()
            return await self.client.is_model_ready(
                model_name=self.model_name,
                model_version=model_version
            )
        except Exception:
            return False
    
    async def _load_model_config(self):
        """Load model configuration"""
        try:
            if not self.client:
                await self.connect()
            
            self.model_config = await self.client.get_model_config(
                model_name=self.model_name
            )
            
            # Extract input/output names
            self.input_names = [input_config.name for input_config in self.model_config.input]
            self.output_names = [output_config.name for output_config in self.model_config.output]
            
            logger.info(f"Model config loaded - Inputs: {self.input_names}, Outputs: {self.output_names}")
            
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            raise
    
    async def _initialize_tokenizer(self):
        """Initialize tokenizer based on model type"""
        try:
            # For sentiment analysis, using a common tokenizer
            # In practice, this should match the tokenizer used during model training
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased",
                use_fast=True
            )
            logger.info("Tokenizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            # Fallback to basic tokenizer if needed
            self.tokenizer = None
    
    async def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata"""
        try:
            if not self.client:
                await self.connect()
            
            metadata = await self.client.get_model_metadata(
                model_name=self.model_name
            )
            
            return {
                "name": metadata.name,
                "platform": metadata.platform,
                "backend": metadata.backend,
                "versions": metadata.versions,
                "inputs": [
                    {
                        "name": inp.name,
                        "datatype": inp.datatype,
                        "shape": list(inp.shape)
                    }
                    for inp in metadata.inputs
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "datatype": out.datatype,
                        "shape": list(out.shape)
                    }
                    for out in metadata.outputs
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            raise
    
    def _prepare_inputs(self, texts: List[str]) -> List[grpcclient.InferInput]:
        """Prepare inputs for Triton inference"""
        # Tokenize texts
        if self.tokenizer:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="np"
            )
            
            inputs = []
            
            # Assuming BERT-style inputs: input_ids, attention_mask, token_type_ids
            if "input_ids" in self.input_names:
                input_ids = grpcclient.InferInput("input_ids", encoded["input_ids"].shape, "INT64")
                input_ids.set_data_from_numpy(encoded["input_ids"].astype(np.int64))
                inputs.append(input_ids)
            
            if "attention_mask" in self.input_names:
                attention_mask = grpcclient.InferInput("attention_mask", encoded["attention_mask"].shape, "INT64")
                attention_mask.set_data_from_numpy(encoded["attention_mask"].astype(np.int64))
                inputs.append(attention_mask)
            
            if "token_type_ids" in self.input_names and "token_type_ids" in encoded:
                token_type_ids = grpcclient.InferInput("token_type_ids", encoded["token_type_ids"].shape, "INT64")
                token_type_ids.set_data_from_numpy(encoded["token_type_ids"].astype(np.int64))
                inputs.append(token_type_ids)
        
        else:
            # Fallback: create dummy inputs (this should be properly implemented based on your model)
            batch_size = len(texts)
            seq_length = self.max_sequence_length
            
            input_ids = grpcclient.InferInput("input_ids", [batch_size, seq_length], "INT64")
            input_ids.set_data_from_numpy(np.ones((batch_size, seq_length), dtype=np.int64))
            inputs = [input_ids]
        
        return inputs
    
    def _prepare_outputs(self) -> List[grpcclient.InferRequestedOutput]:
        """Prepare output requests"""
        outputs = []
        for output_name in self.output_names:
            outputs.append(grpcclient.InferRequestedOutput(output_name))
        return outputs
    
    def _process_outputs(self, results, texts: List[str]) -> List[Dict[str, Any]]:
        """Process inference results"""
        try:
            # Extract logits from the first output (assuming sentiment classification)
            logits = results.as_numpy(self.output_names[0])
            
            # Apply softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            processed_results = []
            for i, (text, probs) in enumerate(zip(texts, probabilities)):
                # Assuming binary sentiment: [negative, positive]
                if len(probs) >= 2:
                    negative_prob = float(probs[0])
                    positive_prob = float(probs[1])
                    
                    sentiment = "positive" if positive_prob > negative_prob else "negative"
                    confidence = max(positive_prob, negative_prob)
                else:
                    # Fallback for single output
                    sentiment = "positive" if float(probs[0]) > 0.5 else "negative"
                    confidence = float(probs[0])
                
                processed_results.append({
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "probabilities": {
                        "negative": negative_prob if len(probs) >= 2 else 1 - confidence,
                        "positive": positive_prob if len(probs) >= 2 else confidence
                    }
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to process outputs: {e}")
            # Return fallback results
            return [
                {
                    "text": text,
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "probabilities": {"negative": 0.5, "positive": 0.5}
                }
                for text in texts
            ]
    
    async def predict(
        self,
        texts: List[str],
        model_version: str = "1"
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Run sentiment analysis prediction"""
        start_time = time.time()
        
        try:
            if not self.client:
                await self.connect()
            
            # Validate batch size
            if len(texts) > self.max_batch_size:
                raise ValueError(f"Batch size {len(texts)} exceeds maximum {self.max_batch_size}")
            
            # Prepare inputs and outputs
            inputs = self._prepare_inputs(texts)
            outputs = self._prepare_outputs()
            
            # Run inference
            results = await self.client.infer(
                model_name=self.model_name,
                model_version=model_version,
                inputs=inputs,
                outputs=outputs,
                timeout=self.timeout
            )
            
            # Process results
            processed_results = self._process_outputs(results, texts)
            processing_time = time.time() - start_time
            
            logger.info(f"Processed {len(texts)} texts in {processing_time:.3f}s")
            
            return processed_results, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Prediction failed after {processing_time:.3f}s: {e}")
            raise 