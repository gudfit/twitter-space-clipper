#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the project root to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Extract Redis host and port from CELERY_BROKER_URL
if [ -z "$CELERY_BROKER_URL" ]; then
    echo "⚠️  CELERY_BROKER_URL not set in .env file"
    exit 1
fi

# Create a temporary Python script to test Redis connection
cat > /tmp/test_redis.py << 'EOL'
import os
import redis
from urllib.parse import urlparse

def test_redis_connection():
    redis_url = os.environ.get('CELERY_BROKER_URL')
    if not redis_url:
        print("CELERY_BROKER_URL not set")
        return False
    
    try:
        # Parse the Redis URL
        parsed = urlparse(redis_url)
        
        # Extract password from netloc
        password = parsed.password
        host = parsed.hostname
        port = parsed.port
        db = int(parsed.path.lstrip('/') or 0)
        
        print(f"Connecting to Redis at {host}:{port}")
        
        # Create Redis client
        r = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            socket_timeout=5
        )
        
        # Test connection
        r.ping()
        return True
    except Exception as e:
        print(f"Redis connection error: {str(e)}")
        return False

if __name__ == '__main__':
    import sys
    sys.exit(0 if test_redis_connection() else 1)
EOL

# Run the Python Redis connection test
python /tmp/test_redis.py
if [ $? -ne 0 ]; then
    echo "⚠️  Failed to connect to Redis using CELERY_BROKER_URL"
    echo "Current CELERY_BROKER_URL: $CELERY_BROKER_URL"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p logs

# Change to the project root directory
cd "${SCRIPT_DIR}"

# Start Celery worker
echo "🚀 Starting Celery worker..."
PYTHONPATH="${SCRIPT_DIR}" celery -A celery_worker.celery_app worker --loglevel=info > logs/celery.log 2>&1 &
CELERY_PID=$!

# Start Streamlit app using the wrapper
echo "🌟 Starting Streamlit app..."
PYTHONPATH="${SCRIPT_DIR}" python app/run_streamlit.py > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!

# Function to cleanup processes on exit
cleanup() {
    echo "🛑 Stopping services..."
    kill $CELERY_PID
    kill $STREAMLIT_PID
    rm -f /tmp/test_redis.py
    exit 0
}

# Register cleanup function
trap cleanup SIGINT SIGTERM

# Keep script running and show logs
echo "📝 Viewing logs (Ctrl+C to stop)..."
echo "----------------------------------------"
tail -f logs/celery.log logs/streamlit.log

# Wait for processes
wait $CELERY_PID $STREAMLIT_PID
