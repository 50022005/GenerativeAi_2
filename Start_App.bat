@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
 .venv\Scripts\python.exe scripts\start_app.py
) else (
 python scripts\start_app.py
)
if errorlevel 1 (
 echo The app could not start. Run Setup_CPU.bat once and check that port 7860 is available.
 pause
)
