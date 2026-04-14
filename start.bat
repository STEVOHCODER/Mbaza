@echo off
echo Starting AutoDev AI Dashboard...

echo Installing backend requirements...
cd /d "%~dp0backend"
pip install -r requirements.txt
start cmd /k "title AutoDev Backend & uvicorn main:app --reload --port 8000"

echo Starting frontend dev server...
cd /d "%~dp0frontend"
start cmd /k "title AutoDev Frontend & npm run dev"

echo Both services have been started. You can view the UI in your browser!
pause
