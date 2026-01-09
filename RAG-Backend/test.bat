@echo off
REM Test offline readiness of RAG Backend

echo ================================
echo  Testing Offline Readiness
echo ================================
echo.

call myenv\Scripts\activate.bat
python test_offline.py

pause
