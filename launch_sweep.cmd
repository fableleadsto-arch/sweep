@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_sweep.cmd first.
    pause
    exit /b 1
)
start "Sweep" ".venv\Scripts\pythonw.exe" -m sweep.desktop %*
if errorlevel 1 pause
