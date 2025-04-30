"""Core functionality for processing media files."""
import os
import shutil
import logging
import json
import fcntl
import time
import subprocess
from typing import Dict, Optional, Protocol, Any, Callable, List, TypedDict, Union
from pathlib import Path
from datetime import datetime, timedelta

from core.download import download_twitter_space
from celery_worker.tasks import transcribe_task, generate_quotes_task, generate_summary_task
from core.quotes import create_quote_thread, chunk_transcript
from core.summary import generate_summary
from utils.file_utils import clean_filename
from .types import ProcessState, StoragePaths

try:
    import yt_dlp  # type: ignore
except ImportError:
    yt_dlp = None  # type: ignore

# Configure module logger
logger = logging.getLogger(__name__)

def check_ffmpeg() -> bool:
    """Check if ffmpeg is available in the system."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def get_download_cache_path(storage_root: str) -> Path:
    """Get path to download cache directory."""
    return Path(storage_root) / "download_cache"

def get_cached_download(url: str, storage_root: str) -> Optional[str]:
    """Check if URL has been downloaded before and return cached file path if it exists.
    
    Args:
        url: URL to check
        storage_root: Root storage directory
        
    Returns:
        Path to cached file if it exists, None otherwise
    """
    cache_dir = get_download_cache_path(storage_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Use URL hash as cache key
    url_hash = get_space_id(url)
    cache_file = cache_dir / f"{url_hash}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                cached_path = cache_data.get('file_path')
                if cached_path and os.path.exists(cached_path):
                    return cached_path
        except (json.JSONDecodeError, KeyError):
            pass
    return None

def save_to_download_cache(url: str, file_path: str, storage_root: str):
    """Save downloaded file info to cache.
    
    Args:
        url: Original download URL
        file_path: Path to downloaded file
        storage_root: Root storage directory
    """
    cache_dir = get_download_cache_path(storage_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    url_hash = get_space_id(url)
    cache_file = cache_dir / f"{url_hash}.json"
    
    cache_data = {
        'url': url,
        'file_path': file_path,
        'download_time': datetime.now().isoformat()
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

class ProgressCallback(Protocol):
    """Protocol for progress tracking callbacks."""
    def __call__(self, stage: str, progress: float, status: str) -> None: ...

class ProcessLock:
    """Context manager for process locking using file locks."""
    def __init__(self, storage_root: str, space_id: str, timeout: int = 3600, retry_delay: float = 1.0, max_retries: int = 3):
        self.storage_root = storage_root
        self.space_id = space_id
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
            self.lock_file = open(self.lock_path, 'a+')
            
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
                    
                    # Check process state before breaking lock
                    state = get_process_state(self.storage_root, self.space_id)
                    if state['status'] == 'processing':
                        # If process is still active according to state, don't break lock
                        last_updated = datetime.fromisoformat(state['last_updated']) if state.get('last_updated') else None
                        if last_updated and (datetime.now() - last_updated) <= timedelta(hours=1):
                            logger.info("Process is still active, waiting for lock...")
                            time.sleep(self.retry_delay)
                            continue
                    
                    # If process state is not active or is stale, try to break lock
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

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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

    def _break_stale_lock(self) -> None:
        """Break a stale lock by removing the lock file."""
        try:
            if os.path.exists(self.lock_path):
                os.remove(self.lock_path)
        except OSError as e:
            logger.error(f"Error breaking stale lock: {e}")

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

def get_storage_paths(storage_root: str, space_id: str) -> StoragePaths:
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

def get_process_state(storage_root: str, space_id: str) -> ProcessState:
    """Get the current processing state for a space."""
    state_path = Path(storage_root) / "state" / f"{space_id}.json"
    
    # Get current state from file
    current_state: ProcessState = {
        'status': 'not_started',
        'stage': None,
        'progress': 0.0,
        'last_updated': None,
        'error': None,
        'files': {},
        'current_chunk': None,
        'total_chunks': None,
        'completed_chunks': []
    }
    
    if state_path.exists():
        try:
            with open(state_path, 'r') as f:
                loaded_state = json.load(f)
                # Type-safe update of current state
                if isinstance(loaded_state, dict):
                    for key in current_state.keys():
                        if key in loaded_state:
                            current_state[key] = loaded_state[key]  # type: ignore
        except json.JSONDecodeError:
            pass

    # Get expected file paths
    paths = get_storage_paths(storage_root, space_id)
    
    # Check which files actually exist
    files_status = {
        'audio': os.path.exists(paths['audio_path']),
        'transcript': os.path.exists(paths['transcript_path']),
        'quotes': os.path.exists(paths['quotes_path']),
        'summary': os.path.exists(paths['summary_path'])
    }
    current_state['files'] = files_status
    
    # If state shows complete/processing but files are missing, update state
    if current_state['status'] in ['complete', 'processing']:
        expected_files: Dict[str, List[str]] = {
            'download': ['audio'],
            'transcribe': ['audio', 'transcript'],
            'quotes': ['audio', 'transcript', 'quotes'],
            'summary': ['audio', 'transcript', 'quotes', 'summary']
        }
        
        current_stage = current_state['stage']
        if current_stage and current_stage in expected_files:
            missing_files = [
                f for f in expected_files[current_stage]
                if not files_status.get(f, False)
            ]
            
            if missing_files:
                logger.warning(f"Process state indicates {current_stage} but missing files: {missing_files}")
                current_state.update({
                    'status': 'error',
                    'error': f'Missing required files: {", ".join(missing_files)}',
                    'progress': 0.0
                })
                save_process_state(storage_root, space_id, current_state)
    
    return current_state

def save_process_state(storage_root: str, space_id: str, state: ProcessState) -> None:
    """Save the current processing state.
    
    Args:
        storage_root: Root directory for storage
        space_id: Unique identifier for the space
        state: State dictionary to save
    """
    state_dir = Path(storage_root) / "state"
    state_dir.mkdir(exist_ok=True)
    state_path = state_dir / f"{space_id}.json"
    
    # Update last_updated timestamp
    state['last_updated'] = datetime.now().isoformat()
    
    with open(state_path, 'w') as f:
        json.dump(state, f)

def process_space(
    url: str,
    storage_root: str,
    progress_callback: Optional[ProgressCallback] = None
) -> Optional[StoragePaths]:
    """Process media URL and return paths to generated files.
    
    Args:
        url: URL of the media to process
        storage_root: Root directory for storage
        progress_callback: Optional callback for progress updates
        
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
        
        def update_progress(stage: str, progress: float, status: str = ""):
            """Update progress state and call progress callback if provided."""
            if progress_callback:
                progress_callback(stage, progress, status)
            state.update({
                'stage': stage,
                'progress': progress,
                'status': 'processing' if progress < 1.0 else 'complete',
                'last_updated': datetime.now().isoformat()
            })
            save_process_state(storage_root, space_id, state)
        
        # Acquire process lock
        with ProcessLock(storage_root, space_id) as lock:
            # Check if all files already exist
            if all(os.path.exists(str(p)) for p in paths.values()):
                logger.info("All files already exist")
                state.update({
                    'status': 'complete',
                    'stage': 'complete',
                    'progress': 1.0,
                    'error': None
                })
                save_process_state(storage_root, space_id, state)
                return paths

            try:
                # Download audio if needed
                if not os.path.exists(paths['audio_path']):
                    logger.info("Starting audio download")
                    update_progress("download", 0.0, "Starting download...")
                    
                    def download_progress_hook(d: Dict[str, Any]):
                        """Progress hook for yt-dlp."""
                        if d['status'] == 'downloading':
                            total_bytes = d.get('total_bytes')
                            downloaded_bytes = d.get('downloaded_bytes', 0)
                            if total_bytes:
                                progress = downloaded_bytes / total_bytes
                                speed = d.get('speed', 0)
                                if speed:
                                    speed_mb = speed / (1024 * 1024)
                                    status = f"Downloading media: {progress*100:.1f}% ({speed_mb:.1f} MB/s)"
                                else:
                                    status = f"Downloading media: {progress*100:.1f}%"
                            else:
                                progress = 0
                                status = "Downloading media..."
                            update_progress("download", progress * 0.8, status)
                        elif d['status'] == 'finished':
                            update_progress("download", 0.8, "Download complete, extracting audio...")
                    
                    # Configure download options with progress tracking
                    download_opts = {
                        'format': 'bestaudio/best',
                        'outtmpl': os.path.join(os.path.dirname(paths['audio_path']), '%(id)s.%(ext)s'),
                        'progress_hooks': [download_progress_hook],
                        'keepvideo': True,
                        'postprocessors': []
                    }
                    
                    try:
                        with yt_dlp.YoutubeDL(download_opts) as ydl:
                            update_progress("download", 0.1, "Fetching media information...")
                            info = ydl.extract_info(url, download=True)
                            original_file = os.path.join(os.path.dirname(paths['audio_path']), f"{info['id']}.{info['ext']}")
                            
                            # Convert to MP3 if needed
                            if info['ext'] != 'mp3':
                                if check_ffmpeg():
                                    logger.info("Converting to MP3...")
                                    update_progress("download", 0.9, "Converting to MP3 format...")
                                    
                                    # Run ffmpeg conversion
                                    subprocess.run([
                                        'ffmpeg', '-i', original_file,
                                        '-vn', '-ar', '44100', '-ac', '2',
                                        '-b:a', '192k', paths['audio_path']
                                    ], check=True)
                                    
                                    # Clean up original file
                                    os.remove(original_file)
                                    
                                    update_progress("download", 1.0, "Audio extraction complete")
                                else:
                                    logger.warning("ffmpeg not found - keeping original format")
                                    update_progress("download", 1.0, "Download complete (original format)")
                                    shutil.move(original_file, paths['audio_path'])
                            else:
                                # If already MP3, just move to final location
                                shutil.move(original_file, paths['audio_path'])
                                update_progress("download", 1.0, "Download complete")
                            
                            # Ensure file exists and is readable
                            if not os.path.exists(paths['audio_path']):
                                raise Exception("Audio file not found after download")
                            
                    except Exception as e:
                        logger.error(f"Download failed: {str(e)}")
                        state.update({
                            'status': 'error',
                            'error': f'Download failed: {str(e)}',
                            'progress': 0.0
                        })
                        save_process_state(storage_root, space_id, state)
                        raise

                # Transcribe audio if needed
                if not os.path.exists(paths['transcript_path']):
                    logger.info("Starting transcription")
                    update_progress("transcribe", 0.0, "Starting transcription...")
                    
                    from .transcribe import transcribe_audio
                    
                    # Retry transcription a few times if it fails
                    max_retries = 3
                    retry_delay = 5  # seconds
                    
                    for attempt in range(max_retries):
                        try:
                            update_progress("transcribe", 0.2, f"Transcription attempt {attempt + 1}...")
                            
                            transcript = transcribe_audio(paths['audio_path'])
                            
                            if transcript:
                                # Save transcript
                                with open(paths['transcript_path'], 'w', encoding='utf-8') as f:
                                    f.write(transcript)
                                
                                update_progress("transcribe", 1.0, "Transcription complete")
                                break
                            else:
                                logger.warning(f"Transcription attempt {attempt + 1} failed to produce output")
                                if attempt < max_retries - 1:
                                    time.sleep(retry_delay)
                                    continue
                                else:
                                    raise Exception("Failed to transcribe audio after all retries")
                                
                        except Exception as e:
                            logger.error(f"Transcription attempt {attempt + 1} failed: {str(e)}")
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                continue
                            else:
                                state.update({
                                    'status': 'error',
                                    'error': f'Transcription failed: {str(e)}',
                                    'progress': 0.0
                                })
                                save_process_state(storage_root, space_id, state)
                                raise Exception(f"Failed to transcribe audio after {max_retries} attempts: {str(e)}")

                # Generate quotes if needed
                if not os.path.exists(paths['quotes_path']):
                    logger.info("Starting quote generation")
                    update_progress("quotes", 0.0, "Generating quotes...")
                    
                    try:
                        # Read transcript
                        with open(paths['transcript_path'], 'r', encoding='utf-8') as f:
                            transcript = f.read()
                        
                        # Split transcript into chunks
                        chunks = chunk_transcript(transcript)
                        total_chunks = len(chunks)
                        
                        # Update state with chunk information
                        state.update({
                            'total_chunks': total_chunks,
                            'current_chunk': 0,
                            'completed_chunks': []
                        })
                        save_process_state(storage_root, space_id, state)
                        
                        # Create temporary directory for chunk quotes
                        temp_quotes_dir = Path(storage_root) / "temp" / space_id / "quotes"
                        temp_quotes_dir.mkdir(parents=True, exist_ok=True)
                        
                        all_quotes: List[str] = []
                        completed_chunks = state.get('completed_chunks', [])
                        
                        # Process each chunk
                        for i, chunk in enumerate(chunks):
                            # Skip if chunk already completed
                            if i in completed_chunks:
                                logger.info(f"Skipping completed chunk {i+1}/{total_chunks}")
                                continue
                                
                            logger.info(f"Processing chunk {i+1}/{total_chunks}")
                            state.update({
                                'current_chunk': i,
                                'progress': (i + 0.5) / total_chunks  # +0.5 to show chunk is in progress
                            })
                            save_process_state(storage_root, space_id, state)
                            
                            # Generate quotes for this chunk
                            chunk_quotes = create_quote_thread(chunk, {"url": url})
                            if not chunk_quotes:
                                raise Exception(f"Failed to generate quotes for chunk {i+1}")
                            
                            # Save chunk quotes to temporary file
                            chunk_file = temp_quotes_dir / f"chunk_{i}.txt"
                            with open(chunk_file, 'w', encoding='utf-8') as f:
                                f.write('\n\n'.join(chunk_quotes))
                            
                            # Update state to mark chunk as complete
                            completed_chunks.append(i)
                            state['completed_chunks'] = completed_chunks
                            state['progress'] = (i + 1) / total_chunks
                            save_process_state(storage_root, space_id, state)
                            
                            all_quotes.extend(chunk_quotes)
                            
                        # All chunks complete, save final quotes file
                        if all_quotes:
                            with open(paths['quotes_path'], 'w', encoding='utf-8') as f:
                                f.write('\n\n'.join(all_quotes))
                            
                            # Clean up temporary files
                            if temp_quotes_dir.parent.exists():
                                shutil.rmtree(str(temp_quotes_dir.parent), ignore_errors=True)
                            
                            update_progress("quotes", 1.0, "Quote generation complete")
                        else:
                            raise Exception("No quotes were generated")
                            
                    except Exception as e:
                        logger.error(f"Quote generation failed: {str(e)}")
                        state.update({
                            'status': 'error',
                            'error': f'Quote generation failed: {str(e)}',
                            'progress': state['progress']  # Keep progress for resuming
                        })
                        save_process_state(storage_root, space_id, state)
                        raise

                # Generate summary if needed
                if not os.path.exists(paths['summary_path']):
                    logger.info("Starting summary generation")
                    update_progress("summary", 0.0, "Generating summary...")
                    
                    from .summary import generate_summary
                    
                    try:
                        # Read transcript and quotes
                        with open(paths['transcript_path'], 'r', encoding='utf-8') as f:
                            transcript = f.read()
                        with open(paths['quotes_path'], 'r', encoding='utf-8') as f:
                            quotes_text = f.read()
                            quotes = [q.strip() for q in quotes_text.split('\n\n') if q.strip()]
                        
                        # Generate summary
                        summary = generate_summary(transcript, quotes, paths['summary_path'])
                        if summary:
                            update_progress("summary", 1.0, "Summary generation complete")
                        else:
                            logger.error("Failed to generate summary")
                            update_progress("summary", 1.0, "Summary generation failed")
                            
                    except Exception as e:
                        logger.error(f"Summary generation failed: {str(e)}")
                        state.update({
                            'status': 'error',
                            'error': f'Summary generation failed: {str(e)}',
                            'progress': 0.0
                        })
                        save_process_state(storage_root, space_id, state)
                        raise

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
                logger.error(f"Processing failed: {str(e)}")
                state.update({
                    'status': 'error',
                    'error': str(e),
                    'progress': 0.0
                })
                save_process_state(storage_root, space_id, state)
                raise
                
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
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