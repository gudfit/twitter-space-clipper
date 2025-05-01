#!/bin/bash

# Source common functions
source "$(dirname "$0")/common.sh"

echo "🛑 Stopping all services..."

# Function to kill processes by pattern
kill_processes() {
    local pattern=$1
    local signal=${2:-TERM}
    if pgrep -f "$pattern" > /dev/null; then
        echo "Stopping processes matching: $pattern"
        pkill -$signal -f "$pattern"
        sleep 1
    fi
}

# Kill processes gracefully first
kill_processes "streamlit run"
kill_processes "celery.*worker"
kill_processes "tail -f.*celery"

# Clean up processing states
echo "Cleaning up process states..."
python3 - <<EOF
import json
from pathlib import Path
from datetime import datetime

state_dir = Path("storage/state")
if state_dir.exists():
    for state_file in state_dir.glob("*.json"):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            if state.get('status') == 'processing':
                state.update({
                    'status': 'error',
                    'error': 'Process was interrupted by system shutdown',
                    'stage': None,
                    'progress': 0.0,
                    'last_updated': datetime.now().isoformat()
                })
                with open(state_file, 'w') as f:
                    json.dump(state, f)
        except Exception as e:
            print(f"Error cleaning up state file {state_file}: {e}")
EOF

# Give processes a moment to shutdown gracefully
sleep 2

# Force kill any remaining processes
kill_processes "streamlit run" "9"
kill_processes "celery.*worker" "9"
kill_processes "tail -f.*celery" "9"

# Clean up PID files
echo "Cleaning up PID files..."
rm -f /tmp/celery-*.pid

# Clean up Redis state using configured broker URL
echo "Cleaning up Redis state..."
python3 - <<EOF
import os
import socket
from redis import Redis
from urllib.parse import urlparse
from dotenv import load_dotenv

try:
    # Load environment variables
    load_dotenv()
    
    # Get Redis URL from environment
    redis_url = os.getenv('CELERY_BROKER_URL')
    if not redis_url:
        print("No Redis URL configured, skipping Redis cleanup")
        exit(0)
        
    # Parse Redis URL
    parsed = urlparse(redis_url)
    
    # Connect to Redis with proper configuration
    redis = Redis(
        host=parsed.hostname,
        port=parsed.port or 6379,
        password=parsed.password,
        db=int(parsed.path[1:]) if parsed.path else 0,
        ssl=parsed.scheme == 'rediss'
    )
    
    # Get hostname for namespacing
    hostname = socket.gethostname()
    
    # Clean up task metadata for this host only
    task_pattern = f'celery-task-meta-{hostname}:*'
    task_keys = redis.keys(task_pattern)
    if task_keys:
        redis.delete(*task_keys)
        print(f"Cleaned up {len(task_keys)} task metadata keys for host {hostname}")
    
    # Clean up session state for this host
    session_pattern = f'session:{hostname}:*'
    session_keys = redis.keys(session_pattern)
    if session_keys:
        redis.delete(*session_keys)
        print(f"Cleaned up {len(session_keys)} session keys for host {hostname}")
    
except Exception as e:
    print(f"Error cleaning Redis state: {e}")
EOF

echo "✅ All services stopped"
