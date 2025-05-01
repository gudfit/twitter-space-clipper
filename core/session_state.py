"""Session state management with Redis backing."""
import os
import json
import redis
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta

from core.hostname import HOSTNAME, get_namespaced_key, get_hostname_pattern

# Get Redis connection from environment
REDIS_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')

# Redis client
redis_client = redis.from_url(REDIS_URL)

class SessionState:
    """Session state manager with Redis backing."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.hostname = HOSTNAME
        self._redis_key = get_namespaced_key('session', session_id)
        self._expiry = 3600  # 1 hour default expiry
        
    def _get_full_state(self) -> Dict[str, Any]:
        """Get full state from Redis."""
        state_json = redis_client.get(self._redis_key)
        if state_json:
            return json.loads(state_json)
        return {}
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the session state."""
        state = self._get_full_state()
        return state.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """Set a value in the session state."""
        state = self._get_full_state()
        state[key] = value
        redis_client.setex(self._redis_key, self._expiry, json.dumps(state))
        
    def update(self, data: Dict[str, Any]) -> None:
        """Update multiple values in the session state."""
        state = self._get_full_state()
        state.update(data)
        redis_client.setex(self._redis_key, self._expiry, json.dumps(state))
        
    def delete(self, key: str) -> None:
        """Delete a key from the session state."""
        state = self._get_full_state()
        if key in state:
            del state[key]
            redis_client.setex(self._redis_key, self._expiry, json.dumps(state))
            
    def clear(self) -> None:
        """Clear all session state."""
        redis_client.delete(self._redis_key)
        
    def _extend_expiry(self) -> None:
        """Extend the session expiry time."""
        state = self._get_full_state()
        if state:  # Only extend if state exists
            redis_client.setex(self._redis_key, self._expiry, json.dumps(state))
        
    @staticmethod
    def cleanup_stale_sessions(exclude_keys: Optional[List[str]] = None) -> int:
        """Clean up stale sessions for this hostname.
        
        Args:
            exclude_keys: List of Redis keys to exclude from cleanup
            
        Returns:
            Number of sessions cleaned up
        """
        pattern = get_hostname_pattern('session')
        keys = redis_client.keys(pattern)
        
        if not keys:
            return 0
            
        # Filter out excluded keys
        if exclude_keys:
            keys = [k for k in keys if k.decode('utf-8') not in exclude_keys]
            
        if keys:
            return redis_client.delete(*keys)
        return 0 