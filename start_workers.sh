#!/bin/bash

# Source common functions
source "$(dirname "$0")/common.sh"

# Setup environment and test connections
setup_environment
test_redis_connection
setup_logs
goto_project_root

# Start Celery workers for different queues
echo "🚀 Starting Celery workers..."

# Start worker for download queue
echo "📥 Starting download worker..."
PYTHONPATH="${SCRIPT_DIR}" celery -A celery_worker.tasks worker -Q download -n download@%h --loglevel=INFO > logs/celery_download.log 2>&1 &
DOWNLOAD_PID=$!

# Start worker for transcribe queue
echo "🎯 Starting transcribe worker..."
PYTHONPATH="${SCRIPT_DIR}" celery -A celery_worker.tasks worker -Q transcribe -n transcribe@%h --loglevel=INFO > logs/celery_transcribe.log 2>&1 &
TRANSCRIBE_PID=$!

# Start worker for generate queue
echo "✍️ Starting generate worker..."
PYTHONPATH="${SCRIPT_DIR}" celery -A celery_worker.tasks worker -Q generate -n generate@%h --loglevel=INFO > logs/celery_generate.log 2>&1 &
GENERATE_PID=$!

# Function to cleanup processes on exit
cleanup() {
    echo "🛑 Stopping workers..."
    kill $DOWNLOAD_PID
    kill $TRANSCRIBE_PID
    kill $GENERATE_PID
    rm -f /tmp/test_redis.py
    exit 0
}

# Register cleanup function
trap cleanup SIGINT SIGTERM

# Keep script running and show logs
echo "📝 Viewing logs (Ctrl+C to stop)..."
echo "----------------------------------------"
tail -f logs/celery_download.log logs/celery_transcribe.log logs/celery_generate.log

# Wait for all background processes
wait $DOWNLOAD_PID $TRANSCRIBE_PID $GENERATE_PID 