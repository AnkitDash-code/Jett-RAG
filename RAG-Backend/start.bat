@echo off
REM Quick start script for RAG Backend
REM Runs in OFFLINE MODE using cached models

echo ================================
echo  RAG Backend - Quick Start
echo ================================
echo.

REM Check if virtual environment exists
if not exist "myenv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv myenv
    echo Then run: myenv\Scripts\activate ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)

echo Activating virtual environment...
call myenv\Scripts\activate.bat

REM ========================================
REM OFFLINE MODE - Use cached models only
REM ========================================
echo Setting OFFLINE MODE environment variables...
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

echo.
echo Starting RAG Backend on port 8081 (OFFLINE MODE)...
echo Using locally cached models only.
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8081

pause
