"""Tests for Celery tasks."""

import pytest  # type: ignore
import os
from pathlib import Path
import json
from celery_worker.tasks import (
    download_media,
    transcribe_audio,
    generate_quotes_task,
    generate_summary_task
)
from core.redis_manager import RedisManager
from core.state_manager import StateManager

# Initialize Redis and state manager for tests
redis_client = RedisManager.get_client()
state_manager = StateManager(redis_client)

@pytest.fixture
def test_space_id():
    """Get test space ID."""
    return "test_space_123"

@pytest.fixture
def test_storage_dir(tmp_path):
    """Create temporary storage directory."""
    return tmp_path / "storage"

@pytest.fixture
def test_paths(test_storage_dir, test_space_id):
    """Create test file paths."""
    test_storage_dir.mkdir(exist_ok=True)
    return {
        'audio': test_storage_dir / f"{test_space_id}.mp3",
        'transcript': test_storage_dir / f"{test_space_id}.txt",
        'quotes': test_storage_dir / f"{test_space_id}_quotes.txt",
        'summary': test_storage_dir / f"{test_space_id}_summary.txt"
    }

def test_download_media(test_paths, test_space_id):
    """Test media download task."""
    # Mock URL for testing
    test_url = "https://example.com/test.mp3"
    
    # Start download task
    task = download_media.delay(test_url, str(test_paths['audio']))
    
    # Check initial state
    state = state_manager.get_state(test_space_id)
    assert state is not None
    assert state['status'] == 'DOWNLOADING'
    assert state['metadata']['url'] == test_url
    assert state['metadata']['task_id'] == task.id
    
    # Wait for task to complete
    result = task.get()
    assert result == str(test_paths['audio'])
    
    # Check final state
    state = state_manager.get_state(test_space_id)
    assert state is not None
    assert state['status'] == 'COMPLETE'
    assert state['error'] is None

def test_transcribe_audio(test_paths, test_space_id):
    """Test audio transcription task."""
    # Create dummy audio file
    test_paths['audio'].write_text("Test audio content")
    
    # Start transcription task
    task = transcribe_audio.delay(
        str(test_paths['audio']),
        str(test_paths['transcript'])
    )
    
    # Check initial state
    state = state_manager.get_state(test_space_id)
    assert state is not None
    assert state['status'] == 'TRANSCRIBING'
    assert state['metadata']['task_id'] == task.id
    
    # Wait for task to complete
    result = task.get()
    assert result == str(test_paths['transcript'])
    
    # Check final state
    state = state_manager.get_state(test_space_id)
    assert state is not None
    assert state['status'] == 'COMPLETE'
    assert state['error'] is None

def test_generate_quotes(test_paths, test_space_id):
    """Test quote generation task."""
    # Create test data
    test_data = {
        'space_id': test_space_id,
        'storage_dir': str(test_paths['audio'].parent),
        'transcript': "This is a test transcript.",
        'url': "https://example.com/test"
    }
    
    # Start quotes task
    task = generate_quotes_task.delay(test_data)
    
    # Check initial state
    state = state_manager.get_state(test_space_id)
    assert state is not None
    assert state['status'] == 'GENERATING_QUOTES'
    assert state['metadata']['task_id'] == task.id
    
    # Wait for task to complete
    result = task.get()
    assert result['quotes_path'] == str(test_paths['quotes'])
    assert len(result['quotes']) > 0
    
    # Check final state
    state = state_manager.get_state(test_space_id)
    assert state is not None
    assert state['status'] == 'COMPLETE'
    assert state['error'] is None

def test_generate_summary(test_paths, test_space_id):
    """Test summary generation task."""
    # Create dummy transcript
    test_paths['transcript'].write_text("Test transcript content")
    
    # Start summary task
    task = generate_summary_task.delay(
        str(test_paths['transcript']),
        str(test_paths['summary'])
    )
    
    # Check initial state
    state = state_manager.get_state(test_space_id)
    assert state is not None
    assert state['status'] == 'GENERATING_SUMMARY'
    assert state['metadata']['task_id'] == task.id
    
    # Wait for task to complete
    result = task.get()
    assert result == str(test_paths['summary'])
    
    # Check final state
    state = state_manager.get_state(test_space_id)
    assert state is not None
    assert state['status'] == 'COMPLETE'
    assert state['error'] is None

def test_error_handling(test_paths, test_space_id):
    """Test error handling in tasks."""
    # Try to download from invalid URL
    invalid_url = "invalid://url"
    
    # Start download task
    task = download_media.delay(invalid_url, str(test_paths['audio']))
    
    # Check initial state
    state = state_manager.get_state(test_space_id)
    assert state is not None
    assert state['status'] == 'DOWNLOADING'
    assert state['metadata']['url'] == invalid_url
    assert state['metadata']['task_id'] == task.id
    
    # Wait for task to fail
    with pytest.raises(Exception):
        task.get()
    
    # Check error state
    state = state_manager.get_state(test_space_id)
    assert state is not None
    assert state['status'] == 'ERROR'
    assert state['error'] is not None

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after each test."""
    yield
    # Clear all test states
    pattern = f"{state_manager.state_prefix}*"
    keys = redis_client.keys(pattern)
    if keys:
        redis_client.delete(*keys) 