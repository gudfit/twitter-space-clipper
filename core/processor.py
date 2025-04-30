"""Core functionality for processing media files."""
import os
import shutil
import logging
import json
import fcntl
import time
from typing import Dict, Optional, Protocol, Any, Callable, List
from pathlib import Path
from datetime import datetime, timedelta

from core.download import download_twitter_space
from celery_worker.tasks import transcribe_task, generate_quotes_task, generate_summary_task
from core.quotes import create_quote_thread
from core.summary import generate_summary
from utils.file_utils import clean_filename

# Configure module logger
logger = logging.getLogger(__name__)

class ProgressCallback(Protocol):
    """Protocol for progress tracking callbacks."""
    def __call__(self, stage: str, progress: float, status: str) -> None: ...

class ProcessLock:
    """Context manager for process locking using file locks."""
    def __init__(self, storage_root: str, space_id: str, timeout: int = 3600, retry_delay: float = 1.0, max_retries: int = 3):
        self.lock_dir = Path(storage_root) / "locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.lock_dir / f"{space_id}.lock"
        self.lock_file = None
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        
    def __enter__(self):
        try:
            # Create lock file if it doesn't exist
            self.lock_file = open(self.lock_path, 'w')
            
            # Try to acquire lock
            start_time = time.time()
            retries = 0
            
            while retries < self.max_retries:
                try:
                    fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # Write current timestamp
                    self.lock_file.seek(0)
                    self.lock_file.write(str(time.time()))
                    self.lock_file.flush()
                    return self
                except IOError:
                    # Check if we've exceeded timeout
                    if time.time() - start_time > self.timeout:
                        raise TimeoutError("Failed to acquire lock: timeout exceeded")
                    
                    # Check if the existing lock is stale
                    if self._is_lock_stale():
                        logger.warning("Found stale lock, attempting to break it")
                        self._break_stale_lock()
                        retries += 1
                        if retries >= self.max_retries:
                            raise TimeoutError("Failed to acquire lock after breaking stale lock")
                    
                    # Sleep before retry
                    time.sleep(self.retry_delay)
            
            raise TimeoutError("Failed to acquire lock: max retries exceeded")
            
        except Exception as e:
            if self.lock_file:
                try:
                    self.lock_file.close()
                except Exception:
                    pass
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_file:
            try:
                fcntl.flock(self.lock_file, fcntl.LOCK_UN)
                self.lock_file.close()
                if os.path.exists(self.lock_path):
                    os.remove(self.lock_path)
            except Exception as e:
                logger.error(f"Error cleaning up lock file: {e}")

    def _is_lock_stale(self) -> bool:
        """Check if the existing lock is stale (older than timeout)."""
        try:
            if not os.path.exists(self.lock_path):
                return True
            with open(self.lock_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    return True
                timestamp = float(content)
                return time.time() - timestamp > self.timeout
        except (IOError, ValueError):
            return True

    def _break_stale_lock(self):
        """Break a stale lock by removing the lock file."""
        try:
            if os.path.exists(self.lock_path):
                os.remove(self.lock_path)
        except OSError as e:
            logger.error(f"Error breaking stale lock: {e}")
            pass

def get_space_id(url: str) -> str:
    """Extract space ID from URL and create a hash for storage.
    
    Args:
        url: The URL to process
        
    Returns:
        A hash string identifying the space
    """
    import hashlib
    space_id = url.strip('/').split('/')[-1]
    return hashlib.md5(space_id.encode()).hexdigest()

def get_storage_paths(storage_root: str, space_id: str) -> Dict[str, str]:
    """Get paths for storing space data.
    
    Args:
        storage_root: Root directory for storage
        space_id: Unique identifier for the space
        
    Returns:
        Dictionary of paths for different file types
    """
    storage_dir = Path(storage_root)
    downloads_dir = storage_dir / "downloads"
    transcripts_dir = storage_dir / "transcripts"
    quotes_dir = storage_dir / "quotes"
    summaries_dir = storage_dir / "summaries"
    
    # Ensure directories exist
    for dir_path in [downloads_dir, transcripts_dir, quotes_dir, summaries_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {
        'audio_path': str(downloads_dir / f"{space_id}.mp3"),
        'transcript_path': str(transcripts_dir / f"{space_id}.txt"),
        'quotes_path': str(quotes_dir / f"{space_id}.txt"),
        'summary_path': str(summaries_dir / f"{space_id}.json")
    }

def get_process_state(storage_root: str, space_id: str) -> Dict[str, Any]:
    """Get the current processing state for a space.
    
    Args:
        storage_root: Root directory for storage
        space_id: Unique identifier for the space
        
    Returns:
        Dictionary containing process state information
    """
    state_path = Path(storage_root) / "state" / f"{space_id}.json"
    if state_path.exists():
        try:
            with open(state_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {
        'status': 'not_started',
        'stage': None,
        'progress': 0.0,
        'last_updated': None,
        'error': None
    }

def save_process_state(storage_root: str, space_id: str, state: Dict[str, Any]):
    """Save the current processing state.
    
    Args:
        storage_root: Root directory for storage
        space_id: Unique identifier for the space
        state: State dictionary to save
    """
    state_dir = Path(storage_root) / "state"
    state_dir.mkdir(exist_ok=True)
    state_path = state_dir / f"{space_id}.json"
    
    state['last_updated'] = datetime.now().isoformat()
    with open(state_path, 'w') as f:
        json.dump(state, f)

def process_space(
    url: str,
    storage_root: str,
    progress_callback: Optional[ProgressCallback] = None,
    download_callback: Optional[Callable] = None
) -> Optional[Dict[str, str]]:
    """Process media URL and return paths to generated files.
    
    Args:
        url: URL of the media to process
        storage_root: Root directory for storage
        progress_callback: Optional callback for progress updates
        download_callback: Optional callback for download progress
        
    Returns:
        Dictionary of paths to generated files if successful, None otherwise
    """
    try:
        logger.info(f"Space processing started: URL: {url}")
        
        # Extract space ID from URL
        space_id = get_space_id(url)
        if not space_id:
            logger.error(f"Could not extract space ID from URL: {url}")
            return None
            
        # Get storage paths
        paths = get_storage_paths(storage_root, space_id)
        
        # Initialize or get process state
        state = get_process_state(storage_root, space_id)
        
        def update_progress(stage: str, progress: float, status: str):
            """Update progress state and call progress callback if provided."""
            if progress_callback:
                progress_callback(stage, progress, status)
            state.update({
                'stage': stage,
                'progress': progress,
                'status': status,
                'last_update': datetime.now().isoformat()
            })
            save_process_state(storage_root, space_id, state)
        
        # Acquire process lock
        with ProcessLock(storage_root, space_id):
            # Download audio if needed
            if not os.path.exists(paths['audio_path']):
                logger.info("Starting download")
                update_progress("download", 0.0, "Starting download...")
                
                audio_dir = os.path.dirname(paths['audio_path'])
                success = download_twitter_space(url, audio_dir)
                if not success:
                    logger.error("Failed to download audio")
                    raise Exception("Failed to download audio")
                    
                update_progress("download", 1.0, "Download complete")

            # Transcribe audio if needed
            if not os.path.exists(paths['transcript_path']):
                logger.info("Starting transcription")
                update_progress("transcribe", 0.0, "Starting transcription...")
                
                # Submit transcription task
                task_result = transcribe_task.delay(paths['audio_path'], paths['transcript_path'])
                
                # Wait for task completion
                while not task_result.ready():
                    time.sleep(5)  # Poll every 5 seconds
                    update_progress("transcribe", 0.5, "Transcription in progress...")
                
                transcript = task_result.get()  # This will raise an exception if the task failed
                if not transcript:
                    logger.error("Failed to transcribe audio")
                    raise Exception("Failed to transcribe audio")
                    
                update_progress("transcribe", 1.0, "Transcription complete")

            # Generate quotes if needed
            if not os.path.exists(paths['quotes_path']):
                logger.info("Starting quote generation")
                update_progress("quotes", 0.0, "Generating quotes...")
                
                # Read transcript
                with open(paths['transcript_path'], 'r', encoding='utf-8') as f:
                    transcript = f.read()
                
                # Submit quote generation task
                task_result = generate_quotes_task.delay(transcript, {"url": url}, paths['quotes_path'])
                
                # Wait for task completion
                while not task_result.ready():
                    time.sleep(5)  # Poll every 5 seconds
                    update_progress("quotes", 0.5, "Quote generation in progress...")
                
                quotes = task_result.get()  # This will raise an exception if the task failed
                if not quotes:
                    logger.error("Failed to generate quotes")
                    raise Exception("Failed to generate quotes")
                    
                update_progress("quotes", 1.0, "Quote generation complete")

            # Generate summary if needed
            if not os.path.exists(paths['summary_path']):
                logger.info("Starting summary generation")
                update_progress("summary", 0.0, "Generating summary...")
                
                # Read transcript and quotes
                with open(paths['transcript_path'], 'r', encoding='utf-8') as f:
                    transcript = f.read()
                with open(paths['quotes_path'], 'r', encoding='utf-8') as f:
                    quotes_text = f.read()
                    quotes = [q.strip() for q in quotes_text.split('\n\n') if q.strip()]
                
                # Submit summary generation task
                task_result = generate_summary_task.delay(transcript, quotes, paths['summary_path'])
                
                # Wait for task completion
                while not task_result.ready():
                    time.sleep(5)  # Poll every 5 seconds
                    update_progress("summary", 0.5, "Summary generation in progress...")
                
                summary = task_result.get()  # This will raise an exception if the task failed
                if summary:
                    update_progress("summary", 1.0, "Summary generation complete")
                else:
                    logger.error("Failed to generate summary")
                    update_progress("summary", 1.0, "Summary generation failed")

            # Update final state
            state.update({
                'status': 'complete',
                'stage': 'complete',
                'progress': 1.0,
                'error': None
            })
            save_process_state(storage_root, space_id, state)
            
            return paths
            
    except Exception as e:
        logger.exception("Error processing space")
        # Update error state
        state = get_process_state(storage_root, space_id)
        state.update({
            'status': 'error',
            'error': str(e)
        })
        save_process_state(storage_root, space_id, state)
        
        if progress_callback:
            progress_callback("error", 0.0, str(e))
        return None

def regenerate_quotes(transcript_path: str, quotes_path: str, url: str) -> List[str]:
    """Regenerate quotes from transcript.
    
    Args:
        transcript_path: Path to the transcript file
        quotes_path: Path to save the generated quotes
        url: Original URL of the media
        
    Returns:
        List of generated quotes
    """
    logger.info(f"Regenerating quotes for {url}")
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        # Add space info to match initial generation
        space_info = {"url": url}
        
        quotes = create_quote_thread(transcript, space_info)
        if not quotes:
            logger.warning("No quotes were generated")
            return []
            
        # Save new quotes
        with open(quotes_path, 'w', encoding='utf-8') as f:
            if isinstance(quotes, list):
                f.write("\n\n".join(quotes))
            else:
                f.write(quotes)
        
        # Return quotes as list
        if isinstance(quotes, list):
            return quotes
        return [q.strip() for q in quotes.split('\n\n') if q.strip()]
        
    except Exception as e:
        logger.exception("Error regenerating quotes")
        return [] 