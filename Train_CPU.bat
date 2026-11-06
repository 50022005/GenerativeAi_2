@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (set "PYTHON_EXE=.venv\Scripts\python.exe") else (set "PYTHON_EXE=python")
%PYTHON_EXE% scripts\train.py --epochs 3
if errorlevel 1 goto fail
echo Training completed. Run evaluation separately as described in README.md.
pause
exit /b 0
:fail
echo Training failed. Read the message above.
pause
exit /b 1
