@echo off
echo ===================================
echo BRIDGE ML-Engine Service Startup
echo ===================================

cd /d "%~dp0"

echo Current directory: %CD%

echo.
echo Setting up Python environment...
if exist "venv\" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
)

echo.
echo Checking Python and dependencies...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not found! Please install Python 3.8+ and add to PATH.
    pause
    exit /b 1
)

echo.
echo Installing/updating required packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Warning: Failed to install some packages. Continuing anyway...
)

echo.
echo Setting environment variables for ML-Engine...
set ML_ENGINE_HOST=0.0.0.0
set ML_ENGINE_PORT=8001
set ML_ENGINE_WORKERS=1
set ML_ENGINE_LOG_LEVEL=info

echo.
echo Starting BRIDGE ML-Engine API Service...
echo Host: %ML_ENGINE_HOST%
echo Port: %ML_ENGINE_PORT%
echo Workers: %ML_ENGINE_WORKERS%
echo.

python ml_engine_api_service.py

echo.
echo ML-Engine service has stopped.
pause
