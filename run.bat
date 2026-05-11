@echo off
REM ============================================================
REM  Smart Pothole Detection System - Start Flask Server
REM ============================================================

title Smart Pothole Detection System

echo.
echo  =====================================================
echo   Starting Smart Pothole Detection System...
echo   http://127.0.0.1:5000
echo  =====================================================
echo.

REM Activate venv if present
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual environment not found. Using system Python.
)

REM Ensure ML model exists before starting
if not exist "severity_model.pkl" (
    echo [INFO] severity_model.pkl not found. Training now...
    python train_ml.py
)

REM Start Flask
python app.py

pause
