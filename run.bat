@echo off
REM Starts the application. Open http://127.0.0.1:8000 in your browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.bat first.
  exit /b 1
)
echo Starting Sentiment Analysis with Facebook at http://127.0.0.1:8000  (press Ctrl+C to stop)
".venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
