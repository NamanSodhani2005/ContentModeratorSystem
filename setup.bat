@echo off
echo ========================================
echo Content Moderation RL System Setup
echo ========================================
echo.

echo [1/4] Installing Python dependencies...
pip install -r backend\requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)
echo.

echo [2/4] Installing Node.js dependencies...
cd frontend
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install Node.js dependencies
    pause
    exit /b 1
)
cd ..
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Set up Kaggle API credentials in %USERPROFILE%\.kaggle\kaggle.json
echo   2. Run: python backend\data\download.py
echo   3. Run: python backend\data\preprocess.py
echo   4. Run: python backend\rl_training\train.py
echo   5. Run: python backend\api\app.py
echo   6. In another terminal: cd frontend ^&^& npm run dev
echo.
pause
