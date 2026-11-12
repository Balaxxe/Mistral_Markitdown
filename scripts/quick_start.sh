#!/usr/bin/env bash

# Enhanced Document Converter - macOS/Linux Quick Start Script
# This script sets up the virtual environment, installs dependencies,
# and launches the document converter.

if [ -z "$BASH_VERSION" ]; then
    echo "ERROR: This script requires bash"
    echo "Please run with: bash scripts/quick_start.sh"
    exit 1
fi

set -e

# Always run from the project root (parent of scripts/)
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  Enhanced Document Converter - Setup and Launch"
echo "============================================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.11+ from https://www.python.org/"
    exit 1
fi

echo "[1/5] Checking Python version..."
python3 --version

# Verify Python >= 3.11
py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
py_major=$(echo "$py_version" | cut -d. -f1)
py_minor=$(echo "$py_version" | cut -d. -f2)
if [ "$py_major" -lt 3 ] || { [ "$py_major" -eq 3 ] && [ "$py_minor" -lt 11 ]; }; then
    echo "ERROR: Python 3.11+ is required (found $py_version)"
    echo "Please install Python 3.11+ from https://www.python.org/"
    exit 1
fi

mkdir -p logs

if [ ! -d "env" ]; then
    echo ""
    echo "[2/5] Creating virtual environment..."
    python3 -m venv env
    echo "Virtual environment created successfully."
else
    echo ""
    echo "[2/5] Virtual environment already exists."
    if ! env/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        echo "ERROR: the existing env folder uses Python older than 3.11 (or is broken)."
        echo "Delete the env folder and run this script again to rebuild it with Python 3.11+."
        exit 1
    fi
fi

source env/bin/activate

echo ""
echo "[3/5] Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel > logs/pip_install.log 2>&1

echo ""
echo "[4/5] Installing dependencies (this may take a few minutes)..."
echo "This process is logged to logs/pip_install.log"

pip install -r requirements.txt >> logs/pip_install.log 2>&1

if [ -f "requirements-optional.txt" ]; then
    echo "Installing optional dependencies (audio, YouTube, etc.)..."
    pip install -r requirements-optional.txt >> logs/pip_install.log 2>&1 || \
        echo "WARNING: Some optional dependencies failed to install. See logs/pip_install.log"
fi

echo ""
echo "[5/5] Verifying installation..."
pip check > logs/pip_check.log 2>&1 || echo "WARNING: Some package conflicts detected. See logs/pip_check.log"

pip list > logs/installed_versions.txt 2>&1
echo "Installed package versions saved to logs/installed_versions.txt"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""

if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found"
    echo ""
    echo "Please create a .env file with your configuration:"
    echo "  1. Copy .env.example to .env"
    echo "  2. Add your MISTRAL_API_KEY"
    echo "  3. See README.md for complete configuration options"
    echo ""
    read -p "Would you like to copy .env.example to .env now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            echo "Copied .env.example to .env"
        else
            cat > .env << 'EOF'
# Enhanced Document Converter Configuration
# Add your API key below:
MISTRAL_API_KEY=""
# See .env.example for all configuration options
EOF
            echo "Created basic .env file"
        fi
        echo ""
        echo "Opening .env in your editor -- add your MISTRAL_API_KEY and save."
        ${EDITOR:-nano} .env
    fi
fi

echo ""
echo "Running smoke test..."
python3 main.py --test

echo ""
echo "============================================================"
echo "  Smoke test complete!"
echo ""
echo "  To run the converter:"
echo "    source env/bin/activate"
echo "    python3 main.py"
echo ""
echo "  Or simply run: ./scripts/quick_start.sh"
echo "============================================================"
echo ""

exec bash --rcfile <(echo "source env/bin/activate; PS1='(converter-env) \u@\h:\w\$ '")
