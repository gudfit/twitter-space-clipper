"""Tests for the state manager module."""

import pytest  # type: ignore
from redis import Redis
from datetime import datetime, timedelta
import json
import zlib
from typing import cast
import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from core.state_manager import StateManager, ProcessState, StateStatus, StateMetadata

# Load environment variables
load_dotenv()

@pytest.fixture
def redis_client():
    """Create a Redis client for testing."""
    # Get Redis URL from environment
    redis_url = os.getenv('CELERY_BROKER_URL')
    assert redis_url is not None, "CELERY_BROKER_URL must be set in .env"
    
    # Parse Redis URL
    parsed = urlparse(redis_url)
    host = parsed.hostname or 'localhost'
    port = parsed.port or 6379
    password = parsed.password
    db = int(parsed.path[1:]) if parsed.path else 0
    
    # Create Redis client with parsed configuration
    return Redis(
        host=host,
        port=port,
        password=password,
        db=db,
        decode_responses=True  # Match RedisManager configuration
    )

@pytest.fixture
def state_manager(redis_client):
    """Create a StateManager instance for testing."""
    return StateManager(redis_client)

def test_set_and_get_state(state_manager):
    """Test setting and getting state."""
    space_id = "test_space"
    metadata: StateMetadata = {
        'url': 'https://example.com',
        'space_id': space_id,
        'task_id': 'task123'
    }
    
    # Set initial state
    state_manager.set_state(
        space_id=space_id,
        status='INIT',
        metadata=metadata
    )
    
    # Get state
    state = state_manager.get_state(space_id)
    assert state is not None
    assert state['status'] == 'INIT'
    assert state['message'] == 'Starting process...'
    assert state['metadata'] == metadata
    assert state['error'] is None
    
    # Update state
    state_manager.set_state(
        space_id=space_id,
        status='DOWNLOADING',
        metadata=metadata
    )
    
    # Get updated state
    state = state_manager.get_state(space_id)
    assert state is not None
    assert state['status'] == 'DOWNLOADING'
    assert state['message'] == 'Downloading media...'

def test_clear_state(state_manager):
    """Test clearing state."""
    space_id = "test_space"
    
    # Set state
    state_manager.set_state(
        space_id=space_id,
        status='INIT'
    )
    
    # Verify state exists
    assert state_manager.get_state(space_id) is not None
    
    # Clear state
    state_manager.clear_state(space_id)
    
    # Verify state is cleared
    assert state_manager.get_state(space_id) is None

def test_validate_state(state_manager):
    """Test state validation."""
    valid_state: ProcessState = {
        'status': 'INIT',
        'message': 'Starting process...',
        'error': None,
        'updated_at': datetime.now().isoformat(),
        'metadata': {
            'url': 'https://example.com',
            'space_id': 'test_space',
            'task_id': 'task123'
        }
    }
    
    assert state_manager.validate_state(valid_state) is True
    
    # Test invalid state
    invalid_state = {
        'status': 'INVALID_STATUS',
        'message': 'Invalid'
    }
    
    assert state_manager.validate_state(invalid_state) is False

def test_cleanup_stale_states(state_manager):
    """Test cleaning up stale states."""
    space_id = "test_space"
    
    # Set state with old timestamp
    old_state: ProcessState = {
        'status': 'ERROR',
        'message': 'Error occurred',
        'error': 'Test error',
        'updated_at': (datetime.now() - timedelta(hours=25)).isoformat(),
        'metadata': {
            'url': None,
            'space_id': None,
            'task_id': None
        }
    }
    
    state_manager.redis.set(
        state_manager._get_key(space_id),
        json.dumps(old_state)
    )
    
    # Run cleanup
    state_manager.cleanup_stale_states(max_age_hours=24)
    
    # Verify stale state is removed
    assert state_manager.get_state(space_id) is None
    
    # Set fresh state
    fresh_state: ProcessState = {
        'status': 'COMPLETE',
        'message': 'Process complete',
        'error': None,
        'updated_at': datetime.now().isoformat(),
        'metadata': {
            'url': None,
            'space_id': None,
            'task_id': None
        }
    }
    
    state_manager.redis.set(
        state_manager._get_key("fresh_space"),
        json.dumps(fresh_state)
    )
    
    # Run cleanup
    state_manager.cleanup_stale_states(max_age_hours=24)
    
    # Verify fresh state remains
    assert state_manager.get_state("fresh_space") is not None

def test_error_state(state_manager):
    """Test error state handling."""
    space_id = "test_space"
    error_msg = "Test error message"
    
    # Set error state
    state_manager.set_state(
        space_id=space_id,
        status='ERROR',
        error=error_msg
    )
    
    # Get state
    state = state_manager.get_state(space_id)
    assert state is not None
    assert state['status'] == 'ERROR'
    assert state['message'] == 'Error occurred'
    assert state['error'] == error_msg

def test_archive_state(state_manager):
    """Test archiving process state."""
    space_id = "test_space"
    metadata: StateMetadata = {
        'url': 'https://example.com',
        'space_id': space_id,
        'task_id': 'task123'
    }
    
    # Set initial state
    state_manager.set_state(
        space_id=space_id,
        status='COMPLETE',
        metadata=metadata
    )
    
    # Get the state before archiving to ensure it's valid
    current_state = state_manager.get_state(space_id)
    assert current_state is not None
    assert current_state['status'] == 'COMPLETE'
    
    # Archive the state
    success = state_manager.archive_state(space_id)
    assert success is True
    
    # Original state should be cleared
    assert state_manager.get_state(space_id) is None
    
    # Get archived state
    archived = state_manager.get_archived_state(space_id)
    assert archived is not None
    assert archived['status'] == 'COMPLETE'
    assert archived['metadata'] == metadata

def test_archive_incomplete_state(state_manager):
    """Test attempting to archive incomplete state."""
    space_id = "test_space"
    
    # Set processing state
    state_manager.set_state(
        space_id=space_id,
        status='DOWNLOADING'
    )
    
    # Try to archive - should fail
    success = state_manager.archive_state(space_id)
    assert success is False
    
    # State should still exist
    state = state_manager.get_state(space_id)
    assert state is not None
    assert state['status'] == 'DOWNLOADING'

def test_list_archived_states(state_manager):
    """Test listing archived states."""
    # Archive multiple states
    states = [
        ("space1", 'COMPLETE', {'url': 'https://example1.com'}),
        ("space2", 'ERROR', {'url': 'https://example2.com'}),
        ("space3", 'COMPLETE', {'url': 'https://example3.com'})
    ]
    
    for space_id, status, url_data in states:
        metadata: StateMetadata = {
            'url': url_data['url'],
            'space_id': space_id,
            'task_id': f'task_{space_id}'
        }
        state_manager.set_state(
            space_id=space_id,
            status=cast(StateStatus, status),
            metadata=metadata
        )
        success = state_manager.archive_state(space_id)
        assert success is True
    
    # List archives
    archives = state_manager.list_archived_states()
    assert len(archives) == 3
    
    # Check sorting (most recent first)
    assert all('archived_at' in archive for archive in archives)
    dates = [datetime.fromisoformat(a['archived_at']) for a in archives]
    assert dates == sorted(dates, reverse=True)

def test_compressed_state_handling(state_manager):
    """Test compression and decompression of state data."""
    space_id = "test_space"
    
    # Create state with large console output
    large_output = "x" * 10000  # 10KB of data
    metadata: StateMetadata = {
        'url': 'https://example.com',
        'space_id': space_id,
        'task_id': 'task123'
    }
    
    # Set initial state with metadata
    state_manager.set_state(
        space_id=space_id,
        status='COMPLETE',
        metadata=metadata,
        error=None
    )
    
    # Update console output
    current_state = state_manager.get_state(space_id)
    assert current_state is not None
    current_state['console_output'] = large_output
    state_manager.redis.set(
        state_manager._get_key(space_id),
        json.dumps(current_state)  # No need to encode, Redis client handles it
    )
    
    # Archive the state
    success = state_manager.archive_state(space_id)
    assert success is True
    
    # Get archived state
    archived = state_manager.get_archived_state(space_id)
    assert archived is not None
    assert archived['status'] == 'COMPLETE'
    assert archived.get('console_output') == large_output

@pytest.fixture(autouse=True)
def cleanup(state_manager):
    """Clean up after each test."""
    yield
    # Delete all test keys
    keys = state_manager.redis.keys(f"{state_manager.state_prefix}*")
    archive_keys = state_manager.redis.keys(f"{state_manager.archive_prefix}*")
    all_keys = list(keys) + list(archive_keys)
    if all_keys:
        state_manager.redis.delete(*all_keys) 