@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
 echo Install Python 3.11 or newer from python.org, then reopen this file.
 pause
 exit /b 1
)
if not exist .venv\Scripts\python.exe python -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe scripts\download.py
if errorlevel 1 goto fail
.venv\Scripts\python.exe scripts\prepare.py
if errorlevel 1 goto fail
echo Setup complete. Double-click Start_App.bat.
pause
exit /b 0
:fail
echo Setup failed. Keep this window open and check the message above.
pause
exit /b 1
