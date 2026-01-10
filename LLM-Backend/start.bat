@echo off
echo ================================
echo  LLM-Backend - Quick Start
echo ================================
echo.

cd /d "%~dp0"

if exist "myenv\Scripts\activate.bat" (
    call myenv\Scripts\activate.bat
)

echo [STARTING] Starting KoboldCpp llama.cpp server...
start /B python init.py

echo Waiting 20 seconds for KoboldCpp...
timeout /t 20 /nobreak > nul

echo [STARTING] Starting LLM-Backend on port 8080...
echo Press Ctrl+C to stop...
python main.py

echo.
echo [STOPPING] Cleaning up...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /PID %%a /F >nul 2>&1
echo [OK] Stopped.
