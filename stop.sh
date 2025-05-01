#!/bin/bash

# Source common functions
source "$(dirname "$0")/common.sh"

echo "🛑 Stopping all services..."

# Kill any running Streamlit processes
echo "Stopping Streamlit..."
pkill -f "streamlit run"

# Kill all Celery workers
echo "Stopping Celery workers..."
pkill -f "celery.*worker"

# Kill any remaining tail processes from start_workers.sh
echo "Cleaning up tail processes..."
pkill -f "tail -f.*celery"

# Give processes a moment to shutdown gracefully
sleep 2

# Force kill any remaining processes (if any)
pkill -9 -f "streamlit run"
pkill -9 -f "celery.*worker"
pkill -9 -f "tail -f.*celery"

echo "✅ All services stopped"
