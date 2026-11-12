@echo off
REM Enhanced Document Converter - Windows Quick Start Script
REM This script sets up the virtual environment, installs dependencies,
REM and launches the document converter.

setlocal enabledelayedexpansion

REM Always run from the project root (parent of scripts/)
cd /d "%~dp0\.."

echo ============================================================
echo   Enhanced Document Converter - Setup and Launch
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/
    echo.
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python --version

REM Verify Python >= 3.11
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11+ is required
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Check/create virtual environment
if not exist "env" (
    echo.
    echo [2/5] Creating virtual environment...
    python -m venv env
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo.
    echo [2/5] Virtual environment already exists.
    env\Scripts\python.exe -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: The existing env folder uses Python older than 3.11 ^(or is broken^).
        echo Delete the env folder and run this script again to rebuild it with Python 3.11+.
        pause
        exit /b 1
    )
)

REM Upgrade pip, setuptools, and wheel
echo.
echo [3/5] Upgrading pip, setuptools, and wheel...
env\Scripts\python.exe -m pip install --upgrade pip setuptools wheel > logs\pip_install.log 2>&1
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip
)

REM Install/upgrade dependencies
echo.
echo [4/5] Installing dependencies (this may take a few minutes)...
echo This process is logged to logs\pip_install.log

echo Installing core dependencies from requirements.txt...
env\Scripts\python.exe -m pip install -r requirements.txt >> logs\pip_install.log 2>&1

echo Installing optional dependencies (audio, YouTube, etc.)...
if exist "requirements-optional.txt" (
    env\Scripts\python.exe -m pip install -r requirements-optional.txt >> logs\pip_install.log 2>&1
    if errorlevel 1 (
        echo WARNING: Some optional dependencies failed to install. See logs\pip_install.log
    )
)

echo.
echo [5/5] Verifying installation...
env\Scripts\python.exe -m pip check > logs\pip_check.log 2>&1
if errorlevel 1 (
    echo WARNING: Some package conflicts detected. See logs\pip_check.log
) else (
    echo All packages installed successfully.
)

REM Verify critical dependency
env\Scripts\python.exe -c "import mistralai" >nul 2>&1
if errorlevel 1 (
    echo WARNING: mistralai package failed to install.
    echo          Check logs\pip_install.log for errors.
    echo          Try: env\Scripts\python.exe -m pip install --no-cache-dir mistralai
)

REM Save installed versions
env\Scripts\python.exe -m pip list > logs\installed_versions.txt 2>&1
echo Installed package versions saved to logs\installed_versions.txt

echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.

REM Check for .env file
if not exist ".env" (
    echo.
    echo ============================================================
    echo   CONFIGURATION REQUIRED
    echo ============================================================
    echo.
    echo WARNING: .env file not found
    echo.
    echo Please create a .env file with your configuration:
    echo   1. Copy .env.example to .env
    echo   2. Add your MISTRAL_API_KEY
    echo   3. See README.md for complete configuration options
    echo.
    echo ============================================================
    echo.
    set /p "create_env=Would you like to create a basic .env file now? (Y/N): "
    echo.

    if /i "!create_env!"=="Y" (
        if exist ".env.example" (
            copy .env.example .env >nul
            echo Copied .env.example to .env
        ) else (
            echo # Enhanced Document Converter Configuration > .env
            echo # Add your API key below: >> .env
            echo MISTRAL_API_KEY="" >> .env
            echo # See .env.example for all configuration options >> .env
            echo Created basic .env file
        )
        echo.
        echo Opening .env in notepad -- add your MISTRAL_API_KEY and save.
        echo.
        notepad .env
        echo.
        echo Configuration saved.
        pause
    ) else (
        echo.
        echo Skipping .env creation.
        echo NOTE: You will need to create a .env file manually before using
        echo       Mistral OCR features. See README.md for details.
        echo.
        pause
    )
)

echo.
echo ============================================================
echo   Launching Document Converter...
echo ============================================================
echo.

REM Launch the application
env\Scripts\python.exe main.py %*

REM Pause if there was an error
if errorlevel 1 (
    echo.
    echo ============================================================
    echo   An error occurred. Press any key to exit...
    echo ============================================================
    pause >nul
)

endlocal
