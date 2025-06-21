REM SPDX-FileCopyrightText: 2025 The Despair Authors
REM SPDX-License-Identifier: MIT
@echo off
echo Starting Edge Detection Application with Rye...
echo.

REM Check if .venv exists
if not exist ".venv" (
    echo Virtual environment not found!
    echo Please run setup_rye_windows.bat first.
    pause
    exit /b 1
)

REM Activate Rye virtual environment
call .venv\Scripts\activate.bat

REM Set PyQt6 environment variables
set "QT_PLUGIN_PATH=%CD%\.venv\Lib\site-packages\PyQt6\Qt6\plugins"
set "PATH=%CD%\.venv\Lib\site-packages\PyQt6\Qt6\bin;%PATH%"

REM Launch the application
echo Launching application...
python main.py

REM Keep window open if there was an error
if %errorlevel% neq 0 (
    echo.
    echo Application exited with an error!
    echo.
    echo Trying to fix PyQt6 DLL issues...
    python fix_pyqt6_dll.py
    echo.
    echo Please try running the application again.
    pause
)
