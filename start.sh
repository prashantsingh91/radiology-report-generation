#!/bin/bash
# Master script to start both backend and frontend for MedGemma App

echo "=========================================="
echo "MedGemma Radiology Report App"
echo "=========================================="
echo ""

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from .env file if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Verify Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please check your virtual environment."
    exit 1
fi

# Start backend
echo "[1/2] Starting Backend (FastAPI)..."
cd "$SCRIPT_DIR/backend"
python3 main.py > ../backend.log 2>&1 &
BACKEND_PID=$!

echo "  ✓ Backend started (PID: $BACKEND_PID)"
echo "  → Backend logs: $SCRIPT_DIR/backend.log"
echo ""

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 5

# Start frontend
echo "[2/2] Starting Frontend (React/Vite)..."
cd "$SCRIPT_DIR/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "  → Installing npm dependencies..."
    npm install
fi

npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!

echo "  ✓ Frontend started (PID: $FRONTEND_PID)"
echo "  → Frontend logs: $SCRIPT_DIR/frontend.log"
echo ""

echo "=========================================="
echo "System is ready!"
echo "=========================================="
echo ""
echo "Access the application:"
echo "  • Frontend: http://localhost:3006"
echo "  • Backend API: http://localhost:8004"
echo "  • API Docs: http://localhost:8004/docs"
echo ""

# Save PIDs to file for systemd
echo $BACKEND_PID > /tmp/medgemma-backend.pid
echo $FRONTEND_PID > /tmp/medgemma-frontend.pid

# Verify processes started successfully
sleep 2
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "ERROR: Backend process failed to start!"
    exit 1
fi

if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "ERROR: Frontend process failed to start!"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# If running interactively, wait for processes
if [ -t 0 ]; then
echo "Press Ctrl+C to stop all services"
    echo ""
    wait $BACKEND_PID $FRONTEND_PID
else
    # Running as systemd service - keep running and monitor processes
    echo "Running as systemd service..."
    echo "Service started successfully. Monitoring processes..."
    
    # Monitor processes and restart if needed
    while true; do
        BACKEND_ALIVE=$(kill -0 $BACKEND_PID 2>/dev/null && echo "yes" || echo "no")
        FRONTEND_ALIVE=$(kill -0 $FRONTEND_PID 2>/dev/null && echo "yes" || echo "no")
        
        if [ "$BACKEND_ALIVE" = "no" ] && [ "$FRONTEND_ALIVE" = "no" ]; then
            echo "Both processes stopped, exiting..."
            exit 1
        elif [ "$BACKEND_ALIVE" = "no" ]; then
            echo "Backend process died, exiting..."
            kill $FRONTEND_PID 2>/dev/null
            exit 1
        elif [ "$FRONTEND_ALIVE" = "no" ]; then
            echo "Frontend process died, exiting..."
            kill $BACKEND_PID 2>/dev/null
            exit 1
        fi
        
        sleep 5
    done
fi

