from typing import Dict, Optional, Any, TypedDict, List

class ProcessState(TypedDict):
    """Type definition for process state."""
    status: str  # 'not_started', 'processing', 'complete', 'error'
    stage: Optional[str]  # 'download', 'transcribe', 'quotes', 'summary', 'complete'
    progress: float
    last_updated: Optional[str]
    error: Optional[str]
    files: Dict[str, bool]
    # Chunk tracking for quote generation
    current_chunk: Optional[int]  # Current chunk being processed
    total_chunks: Optional[int]  # Total number of chunks
    completed_chunks: List[int]  # List of completed chunk indices
    # Console output capture
    console_output: Optional[str]  # Captured console output for current stage

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