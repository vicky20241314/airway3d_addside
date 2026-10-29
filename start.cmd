@echo off
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
if defined AIRWAY3D_PYTHON ("%AIRWAY3D_PYTHON%" main.py & exit /b)
if exist ".venv\Scripts\python.exe" (".venv\Scripts\python.exe" main.py & exit /b)
if exist "D:\softwares\miniconda\envs\airway3d\python.exe" ("D:\softwares\miniconda\envs\airway3d\python.exe" main.py & exit /b)
where python >nul 2>nul
if not errorlevel 1 (python main.py & exit /b)
echo Please activate the airway3d environment or set AIRWAY3D_PYTHON.
pause
