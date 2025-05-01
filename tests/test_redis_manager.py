"""Tests for Redis manager."""

import os
import pytest  # type: ignore
from redis import Redis
from core.redis_manager import RedisManager
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

def test_default_connection():
    """Test Redis connection with default configuration."""
    client = RedisManager.get_client()
    assert isinstance(client, Redis)
    
    # Test connection works
    test_key = "test:key"
    test_value = "test_value"
    client.set(test_key, test_value)
    assert client.get(test_key) == test_value  # No decode needed, using decode_responses=True
    client.delete(test_key)
    
    # Close connection
    RedisManager.close()

def test_custom_config():
    """Test Redis connection with custom configuration."""
    # Save original config
    original_broker_url = os.getenv('CELERY_BROKER_URL')
    assert original_broker_url is not None, "CELERY_BROKER_URL must be set in .env"
    
    try:
        # Parse original URL to get host and port
        parsed = urlparse(original_broker_url)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 6379
        
        # Create test URL with same host/port but different db
        test_url = f"redis://{host}:{port}/1"
        if parsed.password:
            test_url = f"redis://:{parsed.password}@{host}:{port}/1"
        
        os.environ['CELERY_BROKER_URL'] = test_url
        
        # Close existing connection
        RedisManager.close()
        
        # Get new client with custom config
        client = RedisManager.get_client()
        
        # Test connection works with new db
        test_key = "test:key"
        test_value = "test_value"
        client.set(test_key, test_value)
        assert client.get(test_key) == test_value  # No decode needed, using decode_responses=True
        client.delete(test_key)
        
        # Verify we're using db 1
        assert client.connection_pool.connection_kwargs['db'] == 1
        
    finally:
        # Restore original config
        if original_broker_url:
            os.environ['CELERY_BROKER_URL'] = original_broker_url
        
        # Close connection
        RedisManager.close()

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after each test."""
    yield
    RedisManager.close() 