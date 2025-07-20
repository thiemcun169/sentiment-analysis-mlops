#!/usr/bin/env python3
"""
Convert sentiment analysis models to TensorRT .plan format for optimal GPU inference
Supports PyTorch and ONNX model conversion to TensorRT
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import onnx
import tensorrt as trt
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class SentimentModelConverter:
    """Convert sentiment analysis models to TensorRT format"""
    
    def __init__(
        self,
        precision: str = "fp16",
        max_batch_size: int = 32,
        max_sequence_length: int = 512,
        workspace_size: int = 1 << 30  # 1GB
    ):
        self.precision = precision
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.workspace_size = workspace_size
        
        # TensorRT precision settings
        self.precision_flags = {
            "fp32": [],
            "fp16": [trt.BuilderFlag.FP16],
            "int8": [trt.BuilderFlag.INT8, trt.BuilderFlag.FP16]
        }
    
    def convert_pytorch_to_onnx(
        self,
        model_name_or_path: str,
        output_path: str,
        num_labels: int = 2
    ) -> str:
        """Convert PyTorch sentiment model to ONNX format"""
        logger.info(f"Converting PyTorch model {model_name_or_path} to ONNX...")
        
        # Load model and tokenizer
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = num_labels
        
        model = AutoModel.from_pretrained(model_name_or_path, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Add classification head if not present
        if not hasattr(model, 'classifier'):
            model.classifier = nn.Linear(config.hidden_size, num_labels)
        
        model.eval()
        
        # Create dummy inputs
        dummy_input_ids = torch.randint(
            0, tokenizer.vocab_size, 
            (self.max_batch_size, self.max_sequence_length),
            dtype=torch.long
        )
        dummy_attention_mask = torch.ones(
            (self.max_batch_size, self.max_sequence_length),
            dtype=torch.long
        )
        
        # Dynamic axes for variable batch size and sequence length
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size'}
        }
        
        # Export to ONNX
        onnx_path = output_path.replace('.plan', '.onnx')
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        logger.info(f"ONNX model saved to {onnx_path}")
        return onnx_path
    
    def create_network_from_onnx(self, builder: trt.Builder, onnx_path: str) -> trt.INetworkDefinition:
        """Create TensorRT network from ONNX model"""
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return None
        
        return network
    
    def create_optimization_profile(self, builder: trt.Builder, network: trt.INetworkDefinition) -> trt.IOptimizationProfile:
        """Create optimization profile for dynamic shapes"""
        profile = builder.create_optimization_profile()
        
        # Input IDs profile
        profile.set_shape(
            "input_ids",
            (1, 1),  # min
            (self.max_batch_size // 2, self.max_sequence_length // 2),  # opt
            (self.max_batch_size, self.max_sequence_length)  # max
        )
        
        # Attention mask profile
        profile.set_shape(
            "attention_mask",
            (1, 1),  # min
            (self.max_batch_size // 2, self.max_sequence_length // 2),  # opt
            (self.max_batch_size, self.max_sequence_length)  # max
        )
        
        return profile
    
    def build_tensorrt_engine(self, onnx_path: str, engine_path: str) -> bool:
        """Build TensorRT engine from ONNX model"""
        logger.info(f"Building TensorRT engine from {onnx_path}...")
        
        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()
        
        # Set workspace size
        config.max_workspace_size = self.workspace_size
        
        # Set precision flags
        for flag in self.precision_flags.get(self.precision, []):
            config.set_flag(flag)
        
        # Create network from ONNX
        network = self.create_network_from_onnx(builder, onnx_path)
        if network is None:
            return False
        
        # Create optimization profile
        profile = self.create_optimization_profile(builder, network)
        config.add_optimization_profile(profile)
        
        # Build engine
        engine = builder.build_engine(network, config)
        if engine is None:
            logger.error("Failed to build TensorRT engine")
            return False
        
        # Serialize and save engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"TensorRT engine saved to {engine_path}")
        return True
    
    def convert_model(
        self,
        input_path: str,
        output_path: str,
        model_type: str = "pytorch",
        num_labels: int = 2
    ) -> bool:
        """Convert model to TensorRT .plan format"""
        try:
            if model_type == "pytorch":
                # Convert PyTorch to ONNX first
                onnx_path = self.convert_pytorch_to_onnx(input_path, output_path, num_labels)
            elif model_type == "onnx":
                onnx_path = input_path
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Convert ONNX to TensorRT
            success = self.build_tensorrt_engine(onnx_path, output_path)
            
            # Clean up intermediate ONNX file if we created it
            if model_type == "pytorch" and os.path.exists(onnx_path):
                os.remove(onnx_path)
                logger.info(f"Cleaned up intermediate ONNX file: {onnx_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False
    
    def validate_engine(self, engine_path: str, test_data: Optional[Dict] = None) -> bool:
        """Validate the TensorRT engine"""
        logger.info(f"Validating TensorRT engine: {engine_path}")
        
        try:
            # Load engine
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                engine = runtime.deserialize_cuda_engine(f.read())
            
            if engine is None:
                logger.error("Failed to load engine")
                return False
            
            # Check input/output bindings
            logger.info(f"Engine has {engine.num_bindings} bindings:")
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                shape = engine.get_binding_shape(i)
                dtype = engine.get_binding_dtype(i)
                is_input = engine.binding_is_input(i)
                logger.info(f"  Binding {i}: {name} - Shape: {shape}, Type: {dtype}, Input: {is_input}")
            
            logger.info("Engine validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Engine validation failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Convert sentiment analysis models to TensorRT")
    parser.add_argument("--input", "-i", required=True, help="Input model path")
    parser.add_argument("--output", "-o", required=True, help="Output .plan file path")
    parser.add_argument("--type", "-t", choices=["pytorch", "onnx"], default="pytorch", help="Input model type")
    parser.add_argument("--precision", "-p", choices=["fp32", "fp16", "int8"], default="fp16", help="TensorRT precision")
    parser.add_argument("--max-batch-size", type=int, default=32, help="Maximum batch size")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num-labels", type=int, default=2, help="Number of sentiment labels")
    parser.add_argument("--workspace-size", type=int, default=1<<30, help="TensorRT workspace size in bytes")
    parser.add_argument("--validate", action="store_true", help="Validate the generated engine")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize converter
    converter = SentimentModelConverter(
        precision=args.precision,
        max_batch_size=args.max_batch_size,
        max_sequence_length=args.max_seq_length,
        workspace_size=args.workspace_size
    )
    
    # Convert model
    logger.info(f"Starting conversion: {args.input} -> {args.output}")
    success = converter.convert_model(
        input_path=args.input,
        output_path=args.output,
        model_type=args.type,
        num_labels=args.num_labels
    )
    
    if success:
        logger.info("Conversion completed successfully!")
        
        # Validate if requested
        if args.validate:
            converter.validate_engine(args.output)
    else:
        logger.error("Conversion failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 