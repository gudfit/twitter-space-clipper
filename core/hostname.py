"""Centralized hostname utilities for consistent namespacing."""
import socket
from typing import Optional

# Get hostname once at module level
HOSTNAME = socket.gethostname()

def get_namespaced_key(key_type: str, key_name: str) -> str:
    """Get a hostname-namespaced key.
    
    Args:
        key_type: Type of key (e.g. 'task', 'space', 'session')
        key_name: Name/ID of the key
        
    Returns:
        Namespaced key in format: {hostname}:{key_type}:{key_name}
    """
    return f"{HOSTNAME}:{key_type}:{key_name}"

def strip_hostname_prefix(key: str) -> str:
    """Strip hostname prefix from a namespaced key if present.
    
    Args:
        key: Namespaced or raw key
        
    Returns:
        Key without hostname prefix
    """
    if ':' in key:
        parts = key.split(':', 2)
        if len(parts) > 2:
            return ':'.join(parts[2:])
    return key

def get_hostname_pattern(key_type: Optional[str] = None) -> str:
    """Get a pattern for matching hostname-namespaced keys.
    
    Args:
        key_type: Optional key type to match
        
    Returns:
        Pattern string for matching namespaced keys
    """
    if key_type:
        return f"{HOSTNAME}:{key_type}:*"
    return f"{HOSTNAME}:*" 