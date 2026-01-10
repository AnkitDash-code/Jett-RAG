@echo off
echo ================================
echo  LLM-Backend - Stop All
echo ================================
echo.

echo Stopping processes on port 8000 (KoboldCpp)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo Killing PID %%a
    taskkill /PID %%a /F 2>nul
)

echo Stopping processes on port 8080 (FastAPI)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080 ^| findstr LISTENING') do (
    echo Killing PID %%a
    taskkill /PID %%a /F 2>nul
)

echo.
echo [OK] All LLM-Backend processes stopped.
pause
