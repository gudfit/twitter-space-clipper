"""Redis connection manager module."""

import os
from typing import Optional
from redis import Redis
from urllib.parse import urlparse
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class RedisManager:
    """Manages Redis connections."""
    
    _instance: Optional[Redis] = None
    
    @classmethod
    def get_client(cls) -> Redis:
        """Get Redis client instance.
        
        Returns:
            Redis client instance
        """
        if cls._instance is None:
            try:
                # Get Redis configuration from Celery broker URL each time
                redis_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
                
                # Parse Redis URL
                parsed = urlparse(redis_url)
                
                # Extract connection parameters
                host = parsed.hostname or 'localhost'
                port = parsed.port or 6379
                password = parsed.password
                # Handle db number correctly - strip leading slash and convert to int
                db = int(parsed.path.lstrip('/')) if parsed.path else 0
                
                cls._instance = Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True  # Always decode responses to strings
                )
                logger.info(f"Connected to Redis at {host}:{port} db={db}")
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return cls._instance
    
    @classmethod
    def close(cls) -> None:
        """Close Redis connection."""
        if cls._instance is not None:
            try:
                cls._instance.close()
                cls._instance = None
                logger.info("Closed Redis connection")
                
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
                raise 