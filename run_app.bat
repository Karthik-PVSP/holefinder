@echo off
:: filepath: d:\Projects\python_finder\run_app.bat
echo Starting Hole Detector application...

:: Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found. Please run setup_environment.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment and run the application
call venv\Scripts\activate.bat
streamlit run hole_detector.py

:: If the application exits, deactivate the virtual environment
call venv\Scripts\deactivate.bat
echo Application closed.
pause