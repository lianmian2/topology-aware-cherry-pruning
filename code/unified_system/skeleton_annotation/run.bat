@echo off
title Skeleton Annotation Tool
cd /d "%~dp0"

echo ================================
echo   Skeleton Annotation Tool
echo ================================
echo.

set PYEXE=

REM Try conda activate cherry first
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Activating conda environment: cherry
    call conda activate cherry >nul 2>&1
    if %errorlevel% equ 0 (
        where python >nul 2>&1 && set "PYEXE=python"
        echo [INFO] Conda cherry activated
    )
)

REM Fallback: scan for python
if not defined PYEXE where python >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE where python3 >nul 2>&1 && set "PYEXE=python3"
if not defined PYEXE where py >nul 2>&1 && set "PYEXE=py"

REM Fallback: scan common conda paths
if not defined PYEXE (
    for %%d in (
        "%USERPROFILE%\miniconda3\envs\cherry"
        "%USERPROFILE%\anaconda3\envs\cherry"
        "D:\app\anna\envs\cherry"
        "C:\miniconda3\envs\cherry"
        "%USERPROFILE%\miniconda3"
        "%USERPROFILE%\anaconda3"
    ) do (
        if exist "%%~d\python.exe" (
            set "PYEXE=%%~d\python.exe"
            goto :found
        )
    )
)

:found
if not defined PYEXE (
    echo [ERROR] Python not found.
    echo Try one of:
    echo   1. Run in conda terminal: conda activate cherry
    echo      then: python -c "import sys,os;os.chdir(r'%~dp0');sys.path.insert(0,r'%~dp0..\..');from skeleton_annotation.main_gui import main;main()"
    echo   2. Install Python 3.8+ from https://www.python.org/downloads/
    echo      then double-click this bat again
    pause
    exit /b 1
)

echo [INFO] Python: %PYEXE%
echo.

if exist "requirements.txt" (
    echo [INFO] Installing dependencies...
    "%PYEXE%" -m pip install -r requirements.txt -q
    echo.
)

echo [INFO] Starting annotation tool...
echo.
"%PYEXE%" "%~dp0\_launcher.py" 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start.
    echo Run this in terminal to see full error:
    echo   "%PYEXE%" "%~dp0\_launcher.py"
    pause
)
