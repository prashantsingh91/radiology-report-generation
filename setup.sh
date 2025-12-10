#!/bin/bash

# Setup script for MedGemma App
# This script creates a virtual environment and installs dependencies

set -e

echo "🚀 Setting up MedGemma App..."

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check disk space
AVAILABLE_SPACE=$(df -BG "$SCRIPT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "💾 Available disk space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt 5 ]; then
    echo "⚠️  Low disk space detected. Cleaning pip cache..."
    pip cache purge 2>/dev/null || rm -rf ~/.cache/pip/* 2>/dev/null || true
    echo "✅ Cache cleaned"
fi

# Create virtual environment in the project root
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --no-cache-dir

# Install backend dependencies
echo "📥 Installing backend dependencies..."
cd backend
pip install --no-cache-dir -r requirements.txt
cd ..

# Install frontend dependencies
echo "📥 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start the backend: cd backend && python main.py"
echo "3. In another terminal, start the frontend: cd frontend && npm run dev"
echo ""
echo "Or use the start.sh script to start both services."
