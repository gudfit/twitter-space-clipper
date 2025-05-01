"""State management module for handling process state in Redis."""

import json
from datetime import datetime
from typing import Dict, Any, Optional, TypedDict, Literal, cast, List, Union
from redis import Redis
import logging
import zlib
import base64
from core.hostname import HOSTNAME
from celery.result import AsyncResult
from celery_worker.celery_app import celery_app

logger = logging.getLogger(__name__)

# Define state types
StateStatus = Literal[
    'INIT',
    'DOWNLOADING',
    'TRANSCRIBING',
    'GENERATING_QUOTES',
    'GENERATING_SUMMARY',
    'ERROR',
    'COMPLETE'
]

class StateMetadata(TypedDict):
    url: Optional[str]
    space_id: Optional[str]
    task_id: Optional[str]

class ProcessStateRequired(TypedDict):
    status: StateStatus
    message: str
    error: Optional[str]
    updated_at: str
    metadata: StateMetadata

class ProcessState(ProcessStateRequired, total=False):
    console_output: str

# State message mapping
STATE_MESSAGES = {
    'INIT': 'Starting process...',
    'DOWNLOADING': 'Downloading media...',
    'TRANSCRIBING': 'Transcribing audio...',
    'GENERATING_QUOTES': 'Generating quotes...',
    'GENERATING_SUMMARY': 'Creating summary...',
    'ERROR': 'Error occurred',
    'COMPLETE': 'Process complete'
}

class StateManager:
    """Manages process state in Redis."""
    
    def __init__(self, redis_client: Redis):
        """Initialize state manager.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        self.state_prefix = f"{HOSTNAME}:process_state:"
        self.archive_prefix = f"{HOSTNAME}:archived_state:"
    
    def _get_key(self, space_id: str) -> str:
        """Get Redis key for a space ID."""
        return f"{self.state_prefix}{space_id}"
    
    def _get_archive_key(self, space_id: str) -> str:
        """Get Redis key for archived state."""
        return f"{self.archive_prefix}{space_id}"
    
    def _compress_state(self, state: ProcessState) -> str:
        """Compress state data for storage.
        
        Args:
            state: State to compress
            
        Returns:
            Base64 encoded compressed state data
        """
        state_json = json.dumps(state)
        compressed = zlib.compress(state_json.encode())
        return base64.b64encode(compressed).decode()
    
    def _decompress_state(self, data: str) -> ProcessState:
        """Decompress state data.
        
        Args:
            data: Base64 encoded compressed state data
            
        Returns:
            Decompressed state
        """
        compressed = base64.b64decode(data.encode())
        decompressed = zlib.decompress(compressed)
        return cast(ProcessState, json.loads(decompressed.decode()))
    
    def set_state(
        self,
        space_id: str,
        status: StateStatus,
        metadata: Optional[StateMetadata] = None,
        error: Optional[str] = None
    ) -> None:
        """Set process state atomically.
        
        Args:
            space_id: Unique space identifier
            status: Current process status
            metadata: Optional metadata about the process
            error: Optional error message
        """
        default_metadata: StateMetadata = {
            'url': None,
            'space_id': None,
            'task_id': None
        }
        
        if metadata:
            default_metadata.update(metadata)
        
        state: ProcessState = {
            'status': status,
            'message': STATE_MESSAGES[status],
            'error': error,
            'updated_at': datetime.now().isoformat(),
            'metadata': default_metadata
        }
        
        # Atomic state update
        self.redis.set(
            self._get_key(space_id),
            json.dumps(state),
            ex=86400  # 24 hour expiry
        )
        
        logger.debug(f"State updated for {space_id}: {state}")
    
    def get_state(self, space_id: str) -> Optional[ProcessState]:
        """Get current process state.
        
        Args:
            space_id: Unique space identifier
            
        Returns:
            Current process state or None if not found
        """
        try:
            state_data = self.redis.get(self._get_key(space_id))
            if not state_data:
                return None
            
            state = json.loads(state_data)
            # Ensure metadata has all required fields
            if 'metadata' in state:
                default_metadata: StateMetadata = {
                    'url': None,
                    'space_id': None,
                    'task_id': None
                }
                default_metadata.update(state['metadata'])
                state['metadata'] = default_metadata
            
            return cast(ProcessState, state)
            
        except Exception as e:
            logger.error(f"Error getting state for {space_id}: {e}")
            return None
    
    def clear_state(self, space_id: str) -> None:
        """Clear process state.
        
        Args:
            space_id: Unique space identifier
        """
        self.redis.delete(self._get_key(space_id))
        logger.debug(f"State cleared for {space_id}")
    
    def validate_state(self, state: ProcessState) -> bool:
        """Validate state data structure.
        
        Args:
            state: State to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = {'status', 'message', 'updated_at', 'metadata'}
            if not all(field in state for field in required_fields):
                return False
            
            # Validate status
            if state['status'] not in STATE_MESSAGES:
                return False
            
            # Validate timestamp
            datetime.fromisoformat(state['updated_at'])
            
            # Validate metadata
            metadata = state['metadata']
            if not isinstance(metadata, dict):
                return False
            
            # Check metadata fields
            required_metadata = {'url', 'space_id', 'task_id'}
            if not all(field in metadata for field in required_metadata):
                return False
            
            return True
            
        except Exception:
            return False
    
    def check_task_status(self, space_id: str) -> Dict[str, Any]:
        """Check status of current task for a space.
        
        Args:
            space_id: Unique space identifier
            
        Returns:
            Dictionary containing task status information
        """
        state = self.get_state(space_id)
        if not state:
            return {
                'status': 'unknown',
                'message': 'No state found',
                'error': None
            }
        
        # If we have a task ID, check its status
        task_id = state['metadata'].get('task_id')
        if task_id:
            result = AsyncResult(task_id, app=celery_app)
            
            if result.ready():
                if result.successful():
                    # Task completed successfully
                    self.set_state(
                        space_id=space_id,
                        status='COMPLETE',
                        metadata=state['metadata']
                    )
                    return {
                        'status': 'complete',
                        'message': STATE_MESSAGES['COMPLETE'],
                        'error': None
                    }
                else:
                    # Task failed
                    error = str(result.get(propagate=False))
                    self.set_state(
                        space_id=space_id,
                        status='ERROR',
                        error=error,
                        metadata=state['metadata']
                    )
                    return {
                        'status': 'error',
                        'message': STATE_MESSAGES['ERROR'],
                        'error': error
                    }
            else:
                # Task still running
                return {
                    'status': 'processing',
                    'message': state['message'],
                    'error': None
                }
        
        # No task ID, return current state
        return {
            'status': state['status'].lower(),
            'message': state['message'],
            'error': state['error']
        }
    
    def cleanup_stale_states(self, max_age_hours: int = 24) -> None:
        """Clean up stale process states.
        
        Args:
            max_age_hours: Maximum age in hours before state is considered stale
        """
        # Get all state keys
        pattern = f"{self.state_prefix}*"
        keys = self.redis.keys(pattern)
        
        now = datetime.now()
        for key in keys:
            try:
                state_data = self.redis.get(key)
                if not state_data:
                    continue
                
                state = cast(ProcessState, json.loads(state_data))
                updated_at = datetime.fromisoformat(state['updated_at'])
                
                # Check if state is stale
                age = now - updated_at
                if age.total_seconds() > (max_age_hours * 3600):
                    # Archive if complete, delete if error/stale
                    if state['status'] == 'COMPLETE':
                        # TODO: Implement archiving
                        pass
                    else:
                        self.redis.delete(key)
                        logger.info(f"Cleaned up stale state: {key}")
                        
            except Exception as e:
                logger.error(f"Error processing state {key}: {e}")
                continue
    
    def archive_state(self, space_id: str) -> bool:
        """Archive a completed process state.
        
        Args:
            space_id: Unique space identifier
            
        Returns:
            True if archived successfully, False otherwise
        """
        try:
            # Get current state
            state = self.get_state(space_id)
            if not state:
                return False
            
            # Only archive complete or error states
            if state['status'] not in ['COMPLETE', 'ERROR']:
                return False
            
            # Compress state data
            compressed = self._compress_state(state)
            
            # Store in archive with timestamp
            archive_data = {
                'state': compressed,  # Now a base64 string
                'archived_at': datetime.now().isoformat(),
                'status': state['status']
            }
            
            # Save to archive
            self.redis.set(
                self._get_archive_key(space_id),
                json.dumps(archive_data),
                ex=2592000  # 30 day expiry for archives
            )
            
            # Remove original state
            self.clear_state(space_id)
            
            logger.info(f"Archived state for {space_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error archiving state for {space_id}: {e}")
            return False
    
    def get_archived_state(self, space_id: str) -> Optional[ProcessState]:
        """Get archived state for a space.
        
        Args:
            space_id: Unique space identifier
            
        Returns:
            Archived state or None if not found
        """
        try:
            archive_data = self.redis.get(self._get_archive_key(space_id))
            if not archive_data:
                logger.debug(f"No archive found for {space_id}")
                return None
            
            # Parse archive data
            data = json.loads(archive_data)
            compressed_state = data.get('state')
            if compressed_state:
                return self._decompress_state(compressed_state)
            
            logger.warning(f"No state data found in archive for {space_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting archived state for {space_id}: {e}")
            return None
    
    def list_archived_states(self) -> List[Dict[str, Any]]:
        """List all archived states.
        
        Returns:
            List of archived states with metadata
        """
        try:
            # Get all archive keys
            pattern = f"{self.archive_prefix}*"
            keys = self.redis.keys(pattern)
            
            archives = []
            for key in keys:
                try:
                    # Key is already decoded due to decode_responses=True
                    space_id = key.split(':')[-1]
                    raw_data = self.redis.get(key)
                    if not raw_data:
                        continue
                        
                    # Data is already decoded, no need to decode again
                    archive_data = json.loads(raw_data)
                    
                    archives.append({
                        'space_id': space_id,
                        'archived_at': archive_data['archived_at'],
                        'status': archive_data['status']
                    })
                except Exception as e:
                    logger.error(f"Error processing archive {key}: {e}")
                    continue
                    
            return sorted(archives, key=lambda x: x['archived_at'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing archived states: {e}")
            return [] 