@echo off
echo ========================================
echo  Behavioral Authentication System
echo  Complete Startup Script
echo ========================================

echo.
echo Starting both Backend and ML Engine...

REM Start ML Engine in a new window
echo [1/2] Starting ML Engine API (Port 8001)...
start "ML Engine API" cmd /k "cd /d behavioral-auth-engine && start_ml_engine.bat"

REM Wait a bit for ML Engine to start
timeout /t 5 /nobreak > nul

REM Start Backend in a new window  
echo [2/2] Starting Backend API (Port 8000)...
start "Backend API" cmd /k "cd /d backend && start.bat"

echo.
echo ========================================
echo  System Started Successfully!
echo ========================================
echo  Backend API: http://localhost:8000
echo  ML Engine API: http://localhost:8001
echo  Docs: http://localhost:8000/docs
echo  ML Docs: http://localhost:8001/docs
echo ========================================
echo.
echo Both services are running in separate windows.
echo Close this window when done.

pause
