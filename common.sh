#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to setup environment
setup_environment() {
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
}

# Function to test Redis connection
test_redis_connection() {
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
}

# Function to setup logging directory
setup_logs() {
    # Create log directory if it doesn't exist
    mkdir -p logs
}

# Function to change to project root
goto_project_root() {
    cd "${SCRIPT_DIR}"
}

# Export functions
export -f setup_environment
export -f test_redis_connection
export -f setup_logs
export -f goto_project_root 