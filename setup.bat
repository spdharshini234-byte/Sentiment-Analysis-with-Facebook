@echo off
REM One-time setup for Windows: creates a virtual environment, installs packages, prepares the model and database.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"
if errorlevel 1 (
  echo This project needs Python 3.11 or newer. Your version is:
  python --version
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 exit /b 1
)
echo Installing packages (this can take a few minutes)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

if not exist "models\sentiment_pipeline.pkl" (
  echo Training the model...
  ".venv\Scripts\python.exe" scripts\train_model.py
  if errorlevel 1 exit /b 1
)
if not exist "data\processed\sentiment.db" (
  echo Building the analytics database...
  ".venv\Scripts\python.exe" scripts\build_database.py
  if errorlevel 1 exit /b 1
)
echo.
echo Setup finished. Now double-click run.bat (or type run.bat) to start the application.
endlocal
