@echo off
REM Manually cache all models

echo ================================
echo  Caching Models for Offline Use
echo ================================
echo.

call myenv\Scripts\activate.bat
python cache_models.py

pause
