@echo off
echo Starting Behavioral Authentication ML Engine...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the ML Engine API
echo Starting ML Engine API on port 8001...
python ml_engine_api_service.py

pause
