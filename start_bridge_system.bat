@echo off
echo ===================================
echo BRIDGE Complete System Startup
echo ===================================
echo.
echo This script will start both the ML-Engine and Backend services
echo in separate command windows for parallel operation.
echo.

set PROJECT_ROOT=%~dp0

echo ML-Engine will start on: http://localhost:8001
echo Backend will start on: http://localhost:8000
echo.

echo Press any key to start both services...
pause > nul

echo.
echo Starting ML-Engine service...
start "BRIDGE ML-Engine" cmd /k "cd /d "%PROJECT_ROOT%ml-engine" && start_ml_engine.bat"

echo Waiting 10 seconds for ML-Engine to initialize...
timeout /t 10 /nobreak > nul

echo.
echo Starting Backend service...
start "BRIDGE Backend" cmd /k "cd /d "%PROJECT_ROOT%backend" && start_backend.bat"

echo.
echo ===================================
echo Both services are starting up!
echo ===================================
echo.
echo ML-Engine: http://localhost:8001
echo Backend: http://localhost:8000
echo.
echo Check the opened command windows for service status.
echo.
echo To stop services: Close the respective command windows
echo or press Ctrl+C in each window.
echo.
pause
