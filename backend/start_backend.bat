@echo off
echo ===================================
echo BRIDGE Backend Service Startup
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
echo Setting environment variables for Backend...
set BACKEND_HOST=0.0.0.0
set BACKEND_PORT=8000
set ML_ENGINE_URL=http://localhost:8001
set ML_ENGINE_ENABLED=true

echo.
echo Starting BRIDGE Backend API Service...
echo Host: %BACKEND_HOST%
echo Port: %BACKEND_PORT%
echo ML-Engine URL: %ML_ENGINE_URL%
echo ML-Engine Enabled: %ML_ENGINE_ENABLED%
echo.

python -m uvicorn main:app --host %BACKEND_HOST% --port %BACKEND_PORT% --reload

echo.
echo Backend service has stopped.
pause
