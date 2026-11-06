@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (set "PYTHON_EXE=.venv\Scripts\python.exe") else (set "PYTHON_EXE=python")
%PYTHON_EXE% scripts\analyze.py
if errorlevel 1 goto fail
if exist tools\tectonic\tectonic.exe (
 set "TECTONIC_CACHE_DIR=%CD%\tools\tectonic-cache"
 tools\tectonic\tectonic.exe --keep-logs report\main.tex
 if errorlevel 1 goto fail
) else (
 where tectonic >nul 2>nul
 if errorlevel 1 (
  echo Install Tectonic from its official release or compile report/main.tex in your LaTeX editor.
  pause
  exit /b 1
 )
 tectonic --keep-logs report\main.tex
 if errorlevel 1 goto fail
)
copy /y report\main.pdf output\pdf\Criterion_Report.pdf >nul
echo Report updated in output/pdf/Criterion_Report.pdf.
pause
exit /b 0
:fail
echo Report build failed. Read the message above.
pause
exit /b 1
