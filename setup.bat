@echo off
REM ============================================================
REM  Smart Pothole Detection System - Windows Setup Script
REM  Run this script once to set up the entire project.
REM ============================================================

title Smart Pothole Detection System - Setup

echo.
echo  =====================================================
echo   Smart Pothole Detection System - Setup
echo  =====================================================
echo.

REM ── Step 1: Create virtual environment ───────────────────────
echo [1/5] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment. Is Python 3.8+ installed?
    pause
    exit /b 1
)
echo       Done. (venv\)

REM ── Step 2: Activate venv ────────────────────────────────────
echo.
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

REM ── Step 3: Upgrade pip and install dependencies ─────────────
echo.
echo [3/5] Installing dependencies (this may take a few minutes)...
pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check your internet connection.
    pause
    exit /b 1
)
echo       All packages installed successfully.

REM ── Step 4: Generate synthetic dataset & train YOLO ──────────
echo.
echo [4/5] Generating synthetic dataset for YOLO smoke-test...
python generate_sample_data.py
echo.
echo       NOTE: For real pothole detection accuracy you should:
echo         1. Download a real pothole dataset (see README.md)
echo         2. Place images+labels in dataset\train\valid\test
echo         3. Run:  python train_yolo.py
echo.

REM ── Step 5: Train the ML severity model ──────────────────────
echo [5/5] Training RandomForest severity classifier...
python train_ml.py
if errorlevel 1 (
    echo [ERROR] ML training failed. Check pothole_dataset.csv
    pause
    exit /b 1
)
echo       severity_model.pkl saved.

echo.
echo  =====================================================
echo   SETUP COMPLETE!
echo  =====================================================
echo.
echo   To start the Flask server, run:
echo       run.bat
echo   OR manually:
echo       venv\Scripts\activate
echo       python app.py
echo.
pause
