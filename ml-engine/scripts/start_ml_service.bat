@echo off
REM BRIDGE ML-Engine Standalone Service Startup Script for Windows
REM This script starts the ML-Engine as a separate service

echo 🚀 Starting BRIDGE ML-Engine Service...

REM Configuration
if "%ML_ENGINE_HOST%"=="" set ML_ENGINE_HOST=0.0.0.0
if "%ML_ENGINE_PORT%"=="" set ML_ENGINE_PORT=8001
if "%ML_ENGINE_WORKERS%"=="" set ML_ENGINE_WORKERS=1
if "%BRIDGE_ENV%"=="" set BRIDGE_ENV=production

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Set environment variables
set BRIDGE_ENV=%BRIDGE_ENV%
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Initialize models if needed
if not exist "models" (
    echo 🏗️ Initializing ML models...
    python scripts\initialize_models.py
) else if not exist "faiss\index" (
    echo 🏗️ Initializing ML models...
    python scripts\initialize_models.py
)

REM Start ML-Engine service
echo 🚀 Starting ML-Engine API service on %ML_ENGINE_HOST%:%ML_ENGINE_PORT%...
echo 📊 Environment: %BRIDGE_ENV%
echo 👥 Workers: %ML_ENGINE_WORKERS%

uvicorn ml_engine_api_service:app --host %ML_ENGINE_HOST% --port %ML_ENGINE_PORT% --workers %ML_ENGINE_WORKERS% --access-log --log-level info

pause
