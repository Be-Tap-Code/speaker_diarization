#!/bin/bash

# Speaker Diarization Web UI - Startup Script

echo "Starting Speaker Diarization Web UI..."

# Copy UI file to static directory
mkdir -p static
cp ui.html static/index.html

# Start the FastAPI server
python app_api.py
