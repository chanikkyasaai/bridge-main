@echo off
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                                                                              ║
echo ║  ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗                              ║
echo ║  ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝                              ║
echo ║  ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗                                ║
echo ║  ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝                                ║
echo ║  ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗                              ║
echo ║  ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝                              ║
echo ║                                                                              ║
echo ║  Behavioral Risk Intelligence for Dynamic Guarded Entry                     ║
echo ║  ML-Engine Startup Script                                                   ║
echo ║                                                                              ║
echo ║  🏆 SuRaksha Cyber Hackathon - Team "Five"                                  ║
echo ║                                                                              ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

echo ✓ Python found

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ⚠️  Some dependencies may have failed to install
    echo Continuing anyway...
)

echo ✓ Dependencies installed

REM Set environment variables
set BRIDGE_ENV=development
set PYTHONPATH=%CD%

echo 🚀 Starting BRIDGE ML-Engine...
echo.
echo Available commands:
echo   1. Start ML-Engine Service
echo   2. Run Demo
echo   3. Run Tests
echo   4. Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Starting ML-Engine Service...
    python start_ml_engine.py --host 0.0.0.0 --port 8001 --dev
) else if "%choice%"=="2" (
    echo Running ML-Engine Demo...
    python demo_ml_engine.py
) else if "%choice%"=="3" (
    echo Running Tests...
    pytest tests/ -v
) else if "%choice%"=="4" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice. Starting ML-Engine Service by default...
    python start_ml_engine.py --host 0.0.0.0 --port 8001 --dev
)

pause
