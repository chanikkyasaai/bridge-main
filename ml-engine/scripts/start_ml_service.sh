#!/bin/bash

# BRIDGE ML-Engine Standalone Service Startup Script
# This script starts the ML-Engine as a separate service

set -e

echo "🚀 Starting BRIDGE ML-Engine Service..."

# Configuration
ML_ENGINE_HOST=${ML_ENGINE_HOST:-"0.0.0.0"}
ML_ENGINE_PORT=${ML_ENGINE_PORT:-8001}
ML_ENGINE_WORKERS=${ML_ENGINE_WORKERS:-1}
BRIDGE_ENV=${BRIDGE_ENV:-"production"}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
export BRIDGE_ENV=$BRIDGE_ENV
export PYTHONPATH=$(pwd):$PYTHONPATH

# Initialize models if needed
if [ ! -d "models" ] || [ ! -d "faiss/index" ]; then
    echo "🏗️ Initializing ML models..."
    python scripts/initialize_models.py
fi

# Start ML-Engine service
echo "🚀 Starting ML-Engine API service on $ML_ENGINE_HOST:$ML_ENGINE_PORT..."
echo "📊 Environment: $BRIDGE_ENV"
echo "👥 Workers: $ML_ENGINE_WORKERS"

uvicorn ml_engine_api_service:app \
    --host $ML_ENGINE_HOST \
    --port $ML_ENGINE_PORT \
    --workers $ML_ENGINE_WORKERS \
    --access-log \
    --log-level info
