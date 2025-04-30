#!/bin/bash

# Source common functions
source "$(dirname "$0")/common.sh"

# Setup environment and test connections
setup_environment
test_redis_connection
setup_logs
goto_project_root

# Start workers in the background
echo "🚀 Starting Celery workers..."
./start_workers.sh &
WORKERS_PID=$!

# Give workers a moment to start
sleep 2

# Start Streamlit app
echo "🌟 Starting Streamlit app..."
PYTHONPATH="${SCRIPT_DIR}" python app/run_streamlit.py > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!

# Function to cleanup processes on exit
cleanup() {
    echo "🛑 Stopping services..."
    kill $STREAMLIT_PID
    kill $WORKERS_PID
    rm -f /tmp/test_redis.py
    exit 0
}

# Register cleanup function
trap cleanup SIGINT SIGTERM

# Keep script running and show logs
echo "📝 Viewing logs (Ctrl+C to stop)..."
echo "----------------------------------------"
tail -f logs/streamlit.log

# Wait for processes
wait $STREAMLIT_PID $WORKERS_PID
