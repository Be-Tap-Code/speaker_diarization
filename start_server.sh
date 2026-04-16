#!/bin/bash

# Speaker Diarization Web UI - Startup Script

echo "Starting Speaker Diarization Web UI..."

# Copy UI file to static directory
mkdir -p static
cp ui.html static/index.html

# Limit PyTorch GPU memory to ~10GB (A100 has 80GB, so 10GB = ~12.5%)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start the FastAPI server
python app_api.py
