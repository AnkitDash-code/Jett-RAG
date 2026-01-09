@echo off
REM Quick start script for RAG Backend
REM Automatically caches models on first run

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

echo.
echo Starting RAG Backend on port 8001...
echo First run will download models (~175 MB, one-time only)
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8001

pause
