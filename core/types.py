from typing import Dict, Optional, Any, TypedDict, List, Union
from datetime import datetime

class ProcessState(TypedDict):
    """Type definition for process state."""
    status: str  # 'processing', 'complete', 'error'
    stage: Optional[str]  # 'download', 'transcribe', 'quotes', 'summary'
    progress: float
    stage_status: str
    last_updated: str  # ISO format datetime string
    error: Optional[str]
    files: Dict[str, bool]
    # Chunk tracking for quote generation
    current_chunk: Optional[int]  # Current chunk being processed
    total_chunks: Optional[int]  # Total number of chunks
    completed_chunks: List[int]  # List of completed chunk indices
    # Console output capture
    console_output: Optional[str]  # Captured console output for current stage
    # Task tracking
    task_id: Optional[str]  # Celery task ID for tracking
    # Host tracking
    hostname: Optional[str]  # Hostname for distributed processing

def create_process_state(
    stage: str,
    progress: float = 0.0,
    status: str = 'processing',
    stage_status: str = '',
    error: Optional[str] = None,
    files: Optional[Dict[str, bool]] = None,
    console_output: Optional[str] = None,
    task_id: Optional[str] = None,
    hostname: Optional[str] = None
) -> ProcessState:
    """Create a properly initialized ProcessState dictionary.
    
    Args:
        stage: Current processing stage
        progress: Current progress (0.0 to 1.0)
        status: Process status ('processing', 'complete', 'error')
        stage_status: Current stage status message
        error: Error message if any
        files: Dictionary of file statuses
        console_output: Console output text
        task_id: Celery task ID
        hostname: Hostname for distributed processing
        
    Returns:
        Properly initialized ProcessState dictionary
    """
    return {
        'status': status,
        'stage': stage,
        'progress': progress,
        'stage_status': stage_status,
        'last_updated': datetime.now().isoformat(),
        'error': error,
        'files': files or {},
        'current_chunk': None,
        'total_chunks': None,
        'completed_chunks': [],
        'console_output': console_output or '',
        'task_id': task_id,
        'hostname': hostname
    }

class StoragePaths(TypedDict):
    """Type definition for storage paths."""
    audio_path: str
    transcript_path: str
    quotes_path: str
    summary_path: str

class SummaryData(TypedDict):
    """Type definition for summary data."""
    overview: str
    key_points: List[str] 