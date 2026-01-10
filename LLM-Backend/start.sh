#!/bin/bash
# LLM-Backend Start Script (for Git Bash)

echo "================================"
echo " LLM-Backend - Quick Start"
echo "================================"
echo

cd "$(dirname "$0")"

# Activate venv if exists
if [ -f "myenv/Scripts/activate" ]; then
    source myenv/Scripts/activate
fi

# Start KoboldCpp in background
echo "[STARTING] Starting KoboldCpp..."
python init.py &
KOBOLD_PID=$!

echo "Waiting 20 seconds for KoboldCpp to load..."
sleep 20

# Cleanup function
cleanup() {
    echo
    echo "[STOPPING] Shutting down..."
    kill $KOBOLD_PID 2>/dev/null
    echo "[OK] Stopped."
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

echo "[STARTING] Starting LLM-Backend on port 8080..."
echo "Press Ctrl+C to stop both servers..."
echo

python main.py

# Cleanup on normal exit too
cleanup
