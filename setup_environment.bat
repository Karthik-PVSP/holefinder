@echo off
:: filepath: d:\Projects\python_finder\setup_environment.bat
echo Setting up the Hole Detector environment...

:: Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or later.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Make sure you have Python venv module installed.
        pause
        exit /b 1
    )
)

:: Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo Failed to install requirements. Check the requirements.txt file.
    pause
    exit /b 1
)

echo.
echo Environment setup complete!
echo.
echo To run the Hole Detector application:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run the application: streamlit run hole_detector.py
echo.
echo Or simply run the "run_app.bat" script.
echo.
pause