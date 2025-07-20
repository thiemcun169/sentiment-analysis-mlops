#!/bin/bash

# Example script to convert sentiment analysis models to TensorRT

set -e

echo "=== Sentiment Analysis Model Conversion to TensorRT ==="

# Configuration
MODEL_NAME="bert-base-uncased"  # Replace with your trained model
OUTPUT_DIR="../triton-model-repository/sentiment_analysis_tensorrt/1"
PRECISION="fp16"
MAX_BATCH_SIZE=32
MAX_SEQ_LENGTH=512
NUM_LABELS=2

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Converting $MODEL_NAME to TensorRT..."

# Option 1: Convert from Hugging Face model
python convert_to_tensorrt.py \
    --input "$MODEL_NAME" \
    --output "$OUTPUT_DIR/model.plan" \
    --type pytorch \
    --precision "$PRECISION" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --num-labels "$NUM_LABELS" \
    --validate

# Option 2: Convert from local PyTorch model
# python convert_to_tensorrt.py \
#     --input "./path/to/your/model" \
#     --output "$OUTPUT_DIR/model.plan" \
#     --type pytorch \
#     --precision "$PRECISION" \
#     --max-batch-size "$MAX_BATCH_SIZE" \
#     --max-seq-length "$MAX_SEQ_LENGTH" \
#     --num-labels "$NUM_LABELS" \
#     --validate

# Option 3: Convert from ONNX model
# python convert_to_tensorrt.py \
#     --input "./path/to/your/model.onnx" \
#     --output "$OUTPUT_DIR/model.plan" \
#     --type onnx \
#     --precision "$PRECISION" \
#     --max-batch-size "$MAX_BATCH_SIZE" \
#     --max-seq-length "$MAX_SEQ_LENGTH" \
#     --validate

echo "Conversion completed! TensorRT model saved to: $OUTPUT_DIR/model.plan"
echo ""
echo "Next steps:"
echo "1. Verify the model is in the correct Triton repository structure"
echo "2. Start the Triton server with: docker-compose up triton-server"
echo "3. Test the model with: docker-compose up fastapi-service"

# Optional: Show model info
if [ -f "$OUTPUT_DIR/model.plan" ]; then
    echo ""
    echo "Model file size: $(du -h "$OUTPUT_DIR/model.plan" | cut -f1)"
    echo "Model file location: $OUTPUT_DIR/model.plan"
fi 