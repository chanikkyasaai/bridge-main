@echo off
echo Starting Canara AI Backend Server...

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if .env file exists
if not exist ".env" (
    echo Warning: .env file not found, using default configuration
)

REM Start the FastAPI server
echo Starting FastAPI server on http://localhost:8000
echo.
echo Available endpoints:
echo - API Documentation: http://localhost:8000/docs
echo - Alternative Docs: http://localhost:8000/redoc
echo - Health Check: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

python main.py
