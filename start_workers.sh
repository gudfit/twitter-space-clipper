#!/bin/bash

# Source common functions
source "$(dirname "$0")/common.sh"

# Setup environment and test connections
setup_environment
test_redis_connection
setup_logs
goto_project_root

# Create a function to start a worker
start_worker() {
    local queue=$1
    local name=$2
    local logfile="logs/celery_${queue}.log"
    
    echo "🚀 Starting ${name} worker..."
    PYTHONPATH="${SCRIPT_DIR}" celery -A celery_worker.tasks worker \
        -Q "$queue" \
        -n "${queue}@%h" \
        --pidfile="/tmp/celery-${queue}.pid" \
        --loglevel=INFO > "$logfile" 2>&1 &
    
    echo $! > "/tmp/celery-${queue}-main.pid"
    return $!
}

# Start Celery workers for different queues
echo "🚀 Starting Celery workers..."

# Start workers for each queue
start_worker "download" "download"
DOWNLOAD_PID=$!

start_worker "transcribe" "transcribe"
TRANSCRIBE_PID=$!

start_worker "generate" "generate"
GENERATE_PID=$!

# Function to cleanup processes on exit
cleanup() {
    echo "🛑 Stopping workers..."
    
    # Call stop script
    "${SCRIPT_DIR}/stop.sh"
    
    # Remove PID files
    rm -f /tmp/celery-*.pid
    rm -f /tmp/test_redis.py
    exit 0
}

# Register cleanup function
trap cleanup SIGINT SIGTERM EXIT

# Keep script running and show logs
echo "📝 Viewing logs (Ctrl+C to stop)..."
echo "----------------------------------------"
tail -f logs/celery_*.log

# Wait for all background processes
wait $DOWNLOAD_PID $TRANSCRIBE_PID $GENERATE_PID 