@echo off
echo Setting up Canara AI Backend with Supabase Integration...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ===== Setup Complete! =====
echo.
echo Next steps:
echo 1. Create .env file with your Supabase credentials:
echo    SUPABASE_URL=https://your-project.supabase.co
echo    SUPABASE_SERVICE_KEY=your_service_key_here
echo.
echo 2. Run Supabase setup:
echo    python setup_supabase.py
echo.
echo 3. Start the server:
echo    python main.py
echo.
echo 4. Test with demo client:
echo    python supabase_demo_client.py
echo.
pause

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Copy environment file if it doesn't exist
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env file from .env.example...
        copy .env.example .env
        echo Please edit .env file with your configuration
    )
)

echo.
echo Setup complete!
echo.
echo To start the server:
echo 1. Activate virtual environment: venv\Scripts\activate.bat
echo 2. Run the server: python main.py
echo.
echo API Documentation will be available at:
echo - Swagger UI: http://localhost:8000/docs
echo - ReDoc: http://localhost:8000/redoc
echo.
pause
