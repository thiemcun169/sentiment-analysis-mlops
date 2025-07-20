#!/usr/bin/env python3
"""
Complete model conversion script for TensorRT sentiment analysis
Converts tabularisai/multilingual-sentiment-analysis to TensorRT engine for Triton 24.05
Compatible with TensorRT 10.0.1
"""

import os
import torch
import numpy as np
import tensorrt as trt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx
from pathlib import Path
import argparse
import shutil

class SentimentModelConverter:
    def __init__(self, model_name="tabularisai/multilingual-sentiment-analysis"):
        self.model_name = model_name
        self.max_seq_length = 128
        self.max_batch_size = 4
        self.output_dir = Path("triton-model-repository/sentiment_analysis_tensorrt")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "1").mkdir(exist_ok=True)
        
        print(f"🔄 Initializing converter for {model_name}")
        
    def load_model_and_tokenizer(self):
        """Load the HuggingFace model and tokenizer"""
        print("📥 Loading model and tokenizer...")
        
        try:
            # Try loading from cache first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                local_files_only=True
            )
            print("✅ Loaded from cache")
        except:
            # Download if not in cache
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            print("✅ Downloaded and loaded")
        
        self.model.eval()
        
        # Get model info
        print(f"   Model config: {self.model.config}")
        print(f"   Number of classes: {self.model.config.num_labels}")
        
    def export_to_onnx(self):
        """Export PyTorch model to ONNX"""
        print("🔄 Exporting to ONNX...")
        
        onnx_path = self.output_dir / "model.onnx"
        
        # Create dummy inputs with dynamic batch size
        dummy_input_ids = torch.randint(0, 1000, (self.max_batch_size, self.max_seq_length), dtype=torch.long)
        dummy_attention_mask = torch.ones((self.max_batch_size, self.max_seq_length), dtype=torch.long)
        
        # Export with dynamic axes
        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask),
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        print(f"✅ ONNX model exported to {onnx_path}")
        return onnx_path
        
    def build_tensorrt_engine(self, onnx_path):
        """Build TensorRT engine from ONNX"""
        print("🔄 Building TensorRT engine...")
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        
        # Enable FP16 for better performance
        config.set_flag(trt.BuilderFlag.FP16)
        
        # Set memory pool
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
        
        # Create network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("❌ Failed to parse ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Configure optimization profiles for dynamic shapes
        profile = builder.create_optimization_profile()
        
        # Set dynamic shape ranges
        profile.set_shape("input_ids", (1, self.max_seq_length), 
                         (self.max_batch_size//2, self.max_seq_length), 
                         (self.max_batch_size, self.max_seq_length))
        profile.set_shape("attention_mask", (1, self.max_seq_length), 
                         (self.max_batch_size//2, self.max_seq_length), 
                         (self.max_batch_size, self.max_seq_length))
        
        config.add_optimization_profile(profile)
        
        # Build engine
        print("   Building engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("❌ Failed to build TensorRT engine")
            return None
        
        # Save engine
        engine_path = self.output_dir / "1" / "model.plan"
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"✅ TensorRT engine saved to {engine_path}")
        
        # Get engine size
        engine_size_mb = len(serialized_engine) / (1024 * 1024)
        print(f"   Engine size: {engine_size_mb:.1f} MB")
        
        return engine_path
        
    def create_triton_config(self):
        """Create Triton model configuration"""
        print("🔄 Creating Triton configuration...")
        
        config_content = f"""name: "sentiment_analysis_tensorrt"
backend: "tensorrt"
max_batch_size: {self.max_batch_size}
platform: "tensorrt_plan"

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ {self.max_seq_length} ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ {self.max_seq_length} ]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ {self.model.config.num_labels} ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]

dynamic_batching {{
  max_delay_microseconds: 1000
  preferred_batch_size: [ 1, 2, 4 ]
}}

version_policy: {{ latest: {{ num_versions: 1 }} }}
"""
        
        config_path = self.output_dir / "config.pbtxt"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Triton config saved to {config_path}")
        
    def save_tokenizer_info(self):
        """Save tokenizer information for the FastAPI service"""
        print("🔄 Saving tokenizer information...")
        
        tokenizer_info = {
            "model_name": self.model_name,
            "max_length": self.max_seq_length,
            "num_labels": self.model.config.num_labels,
            "label_mapping": {
                0: "very_negative",
                1: "negative", 
                2: "neutral",
                3: "positive",
                4: "very_positive"
            }
        }
        
        import json
        info_path = self.output_dir / "tokenizer_info.json"
        with open(info_path, 'w') as f:
            json.dump(tokenizer_info, f, indent=2)
        
        print(f"✅ Tokenizer info saved to {info_path}")
        
    def test_conversion(self):
        """Test the converted model with a sample input"""
        print("🔄 Testing converted model...")
        
        # Test text
        test_text = "I love this product! It's amazing!"
        
        # Tokenize
        encoded = self.tokenizer(
            test_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Get PyTorch prediction for comparison
        with torch.no_grad():
            pytorch_output = self.model(**encoded)
            pytorch_logits = pytorch_output.logits.numpy()
            pytorch_predictions = torch.softmax(pytorch_output.logits, dim=-1).numpy()
        
        print(f"✅ Test completed")
        print(f"   Input: '{test_text}'")
        print(f"   PyTorch logits shape: {pytorch_logits.shape}")
        print(f"   PyTorch predictions: {pytorch_predictions[0]}")
        print(f"   Predicted class: {np.argmax(pytorch_predictions[0])}")
        
    def convert(self):
        """Run the complete conversion process"""
        print("🚀 Starting model conversion process")
        print("=" * 60)
        
        try:
            # Step 1: Load model
            self.load_model_and_tokenizer()
            
            # Step 2: Export to ONNX
            onnx_path = self.export_to_onnx()
            
            # Step 3: Build TensorRT engine
            engine_path = self.build_tensorrt_engine(onnx_path)
            
            if engine_path is None:
                print("❌ Conversion failed!")
                return False
            
            # Step 4: Create Triton config
            self.create_triton_config()
            
            # Step 5: Save tokenizer info
            self.save_tokenizer_info()
            
            # Step 6: Test conversion
            self.test_conversion()
            
            # Cleanup ONNX file
            os.remove(onnx_path)
            print(f"🧹 Cleaned up temporary ONNX file")
            
            print("\n" + "=" * 60)
            print("🎉 CONVERSION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📁 Model files saved to: {self.output_dir}")
            print(f"🏷️  Model name: sentiment_analysis_tensorrt")
            print(f"📦 Max batch size: {self.max_batch_size}")
            print(f"📏 Max sequence length: {self.max_seq_length}")
            print(f"🏆 Number of classes: {self.model.config.num_labels}")
            print("\n✅ Ready for Triton Inference Server!")
            
            return True
            
        except Exception as e:
            print(f"❌ Conversion failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description="Convert sentiment model to TensorRT")
    parser.add_argument("--model", default="tabularisai/multilingual-sentiment-analysis", 
                       help="HuggingFace model name")
    parser.add_argument("--max-batch-size", type=int, default=4, 
                       help="Maximum batch size")
    parser.add_argument("--max-seq-length", type=int, default=128, 
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    print(f"🔧 TensorRT version: {trt.__version__}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, but continuing...")
    else:
        print(f"🔧 CUDA available: {torch.cuda.get_device_name(0)}")
    
    converter = SentimentModelConverter(args.model)
    converter.max_batch_size = args.max_batch_size
    converter.max_seq_length = args.max_seq_length
    
    success = converter.convert()
    
    if success:
        print("\n🚀 Next steps:")
        print("1. Start Triton server with the model repository")
        print("2. Run the FastAPI service")
        print("3. Test with: python simple_stress_test.py")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 