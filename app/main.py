import streamlit as st  # type: ignore
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import json
import hashlib
import shutil
import yt_dlp  # type: ignore
import re
from typing import Optional, Dict, Any, List, Tuple, Callable, TextIO, BinaryIO, Protocol, Union, cast, Literal
import time
import logging
from streamlit_extras.stylable_container import stylable_container  # type: ignore
from datetime import datetime, timedelta
from celery.result import AsyncResult  # type: ignore
from celery_worker.tasks import app as celery_app  # type: ignore
from celery import chain  # Add this import
import uuid
from core.session_state import SessionState
from core.hostname import HOSTNAME, get_namespaced_key
from core.types import ProcessState, create_process_state, StoragePaths
from core.processor import get_process_state, save_process_state
from core.redis_manager import RedisManager
from core.state_manager import StateManager, StateStatus, StateMetadata
from celery_worker.tasks import (
    download_media,
    transcribe_media   as celery_transcribe,
    generate_quotes_task   as celery_generate_quotes,
    generate_summary_task  as celery_generate_summary,
)
from app.components.state_display import display_state, display_file_status, display_metadata

# Define storage directory
STORAGE_DIR = os.getenv('STORAGE_DIR', 'storage')
STORAGE_PATH = Path(STORAGE_DIR)

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Add file handler with more detailed format for debug logs
debug_handler = logging.FileHandler('app.debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(debug_handler)

# Configure Streamlit page
st.set_page_config(
    page_title="LinkToQuotes",
    page_icon="🎙️",
    layout="wide"
)

# Disable watchdog for PyTorch modules to prevent custom class errors
import streamlit.watcher.path_watcher  # type: ignore
original_watch_file = streamlit.watcher.path_watcher.watch_file

def patched_watch_file(filepath: str, *args: Any, **kwargs: Any) -> Optional[bool]:
    """Patched watch_file function that ignores PyTorch files."""
    if 'torch' in filepath or '_C' in filepath:
        return None
    return cast(bool, original_watch_file(filepath, *args, **kwargs))

# Type ignore because streamlit's type stubs don't match implementation
streamlit.watcher.path_watcher.watch_file = patched_watch_file  # type: ignore

# Initialize session state for UI components that don't need persistence
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize Redis-backed session
session = SessionState(st.session_state.session_id)

# Initialize state manager after Redis session
redis_client = RedisManager.get_client()
state_manager = StateManager(redis_client)

# Check dependencies
def check_dependencies():
    """Check if all required dependencies are installed."""
    missing_deps = []
    
    # Check system ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        missing_deps.append("System ffmpeg")
    
    # Check Python packages
    try:
        import ffmpeg  # type: ignore
    except ImportError:
        missing_deps.append("ffmpeg-python")
    
    return missing_deps

# Check dependencies before proceeding
missing_dependencies = check_dependencies()
if missing_dependencies:
    st.error("""
    ⚠️ Missing required dependencies:
    {}
    
    Please install the missing dependencies:
    - For system ffmpeg: `sudo apt-get install ffmpeg`
    - For ffmpeg-python: `pip install ffmpeg-python`
    
    After installing, please restart the application.
    """.format("\n".join(f"- {dep}" for dep in missing_dependencies)))
    st.stop()

def cleanup_stale_states(session: SessionState):
    """Clean up stale process states on startup."""
    logger.info("Cleaning up stale process states...")
    state_dir = Path(STORAGE_DIR) / "state"
    if not state_dir.exists():
        return

    try:
        # Only clean up other sessions, preserve current session
        current_session_key = session._redis_key
        SessionState.cleanup_stale_sessions(exclude_keys=[current_session_key])
        
        # Then clean up stale process states
        for state_file in state_dir.glob("*.json"):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # If state shows processing but is old, mark as error
                if state.get('status') == 'processing':
                    last_updated = datetime.fromisoformat(state.get('last_updated', '2000-01-01'))
                    time_since_update = datetime.now() - last_updated
                    
                    if time_since_update > timedelta(minutes=5):
                        state.update({
                            'status': 'error',
                            'error': 'Process was interrupted by system shutdown',
                            'stage': None,
                            'progress': 0.0,
                            'last_updated': datetime.now().isoformat()
                        })
                        with open(state_file, 'w') as f:
                            json.dump(state, f)
                        logger.info(f"Marked interrupted process as error: {state_file.stem}")
            except Exception as e:
                logger.error(f"Error cleaning up state file {state_file}: {e}")
    except Exception as e:
        logger.error(f"Error during state cleanup: {e}")

# Run cleanup
cleanup_stale_states(session)

@st.cache_resource
def initialize_session():
    """Initialize session state and cleanup. Only runs once per session."""
    # Initialize session state for UI components that don't need persistence
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Initialize Redis-backed session
    session = SessionState(st.session_state.session_id)
    
    # Run cleanup
    cleanup_stale_states(session)
    
    return session

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible."""
    logger.debug("Checking ffmpeg installation...")
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.debug("ffmpeg check successful")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("ffmpeg not found")
        return False

# Authentication
def check_password():
    """Returns `True` if the user had the correct password."""
    logger.debug("Checking password authentication...")

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            logger.info("Password authentication successful")
            session.set('password_correct', True)
            # Extend session expiry on successful login
            session._extend_expiry()
        else:
            logger.warning("Incorrect password attempt")
            session.set('password_correct', False)

    # Check if already authenticated in this session
    if not session.get('password_correct', False):
        # First show the password input
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        # Only show error if a password was actually entered and it was wrong
        if "password" in st.session_state and st.session_state["password"]:
            st.error("😕 Password incorrect")
        return False
    else:
        # Extend session expiry on each check if authenticated
        session._extend_expiry()
        return True

# Run cleanup AFTER session initialization
cleanup_stale_states(session)

# Initialize UI state (these don't need Redis persistence)
if 'media_selector_key' not in st.session_state:
    st.session_state.media_selector_key = 0  # For forcing selectbox refresh
if 'show_password' not in st.session_state:
    st.session_state.show_password = False

def init_session_state():
    """Initialize session state with proper separation of concerns."""
    # Get persistent state from Redis
    persistent_state = {
        'processing_complete': session.get('processing_complete', False),
        'current_space_id': session.get('current_space_id'),
        'url_history': session.get('url_history', {}),
        'loaded_space_id': session.get('loaded_space_id'),
        'active_processes': session.get('active_processes', []),
        'current_task_id': session.get('current_task_id')
    }
    
    # Update Redis with initial state if needed
    session.update(persistent_state)
    
    # Sync Redis state to Streamlit UI state
    for key, value in persistent_state.items():
        st.session_state[key] = value
    
    # Initialize UI-only state (these don't need persistence)
    ui_state = {
        'download_progress': 0.0,
        'total_fragments': 0,
        'current_fragment': 0,
        'regenerating_quotes': False,
        'selected_media': None,
        'last_check_time': None,
        'last_sync_time': None
    }
    
    for key, value in ui_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sync_session_state():
    """Sync important state changes back to Redis."""
    persistent_keys = [
        'processing_complete',
        'current_space_id',
        'url_history',
        'loaded_space_id',
        'active_processes',
        'current_task_id'
    ]
    
    updates = {
        key: st.session_state.get(key)
        for key in persistent_keys
        if key in st.session_state
    }
    
    session.update(updates)

# Initialize session state
init_session_state()

# Register state sync on shutdown
import atexit
atexit.register(sync_session_state)

# Only proceed with the main app if password check passes
if check_password():
    logger.info("Password verified, loading core functionality...")
    
    # Import core functionality
    from core.download import download_twitter_space
    from core.transcribe import transcribe_audio
    from core.quotes import chunk_transcript
    from core.summary import generate_summary, save_summary, load_summary
    from core.processor import (
        process_space,
        get_space_id,
        get_storage_paths,
        regenerate_quotes,
        get_process_state,
        save_process_state,
        ProcessLock,
        ProgressCallback
    )
    from utils.api import call_deepseek_api

    # Check for ffmpeg at startup
    if not check_ffmpeg():
        logger.error("ffmpeg not found, stopping application")
        st.error("""
        ⚠️ ffmpeg is not installed or not found in PATH. This is required for processing media.
        
        To install ffmpeg:
        - Ubuntu/Debian: `sudo apt-get install ffmpeg`
        - macOS: `brew install ffmpeg`
        - Windows: Download from https://ffmpeg.org/download.html
        
        After installing, please restart the application.
        """)
        st.stop()

    logger.info("Initializing session state...")
    # Initialize session state for progress tracking
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'current_space_id' not in st.session_state:
        st.session_state.current_space_id = None
    if 'download_progress' not in st.session_state:
        st.session_state.download_progress = 0.0
    if 'total_fragments' not in st.session_state:
        st.session_state.total_fragments = 0
    if 'current_fragment' not in st.session_state:
        st.session_state.current_fragment = 0
    if 'regenerating_quotes' not in st.session_state:
        st.session_state.regenerating_quotes = False
    if 'selected_media' not in st.session_state:
        st.session_state.selected_media = None
    if 'url_history' not in st.session_state:
        st.session_state.url_history = {}
    if 'loaded_space_id' not in st.session_state:
        st.session_state.loaded_space_id = None
    if 'active_processes' not in st.session_state:
        st.session_state.active_processes = []
    if 'current_task_id' not in st.session_state:
        st.session_state.current_task_id = None
    if 'last_check_time' not in st.session_state:
        st.session_state.last_check_time = None

    logger.info("Creating storage directories...")
    # Create persistent storage directories
    STORAGE_PATH.mkdir(exist_ok=True)
    DOWNLOADS_DIR = STORAGE_PATH / "downloads"
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    TRANSCRIPTS_DIR = STORAGE_PATH / "transcripts"
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    QUOTES_DIR = STORAGE_PATH / "quotes"
    QUOTES_DIR.mkdir(exist_ok=True)
    SUMMARIES_DIR = STORAGE_PATH / "summaries"
    SUMMARIES_DIR.mkdir(exist_ok=True)

    # Create URL history file
    URL_HISTORY_FILE = STORAGE_PATH / "url_history.json"
    if not URL_HISTORY_FILE.exists():
        logger.info("Creating new URL history file")
        with open(URL_HISTORY_FILE, 'w') as f:
            json.dump({}, f)

    def log_processing_step(step: str, status: str = "started", details: Optional[str] = None) -> None:
        """Helper function to log processing steps consistently"""
        message = f"{step} {status}"
        if details:
            message += f": {details}"
        logger.info(message)

    def load_url_history():
        """Load URL history from file."""
        try:
            with open(URL_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_url_history(space_id: str, url: str):
        """Save URL to history in both Redis and local file system."""
        # Update Redis
        history = session.get('url_history', {})
        history[space_id] = url
        session.set('url_history', history)
        st.session_state.url_history = history  # Update UI state
        
        # Update local file
        try:
            with open(URL_HISTORY_FILE, 'r') as f:
                file_history = json.load(f)
        except Exception:
            file_history = {}
        
        file_history[space_id] = url
        with open(URL_HISTORY_FILE, 'w') as f:
            json.dump(file_history, f)

    def get_url_from_history(space_id: str) -> str:
        """Get original URL for a space_id from Redis or local file.
        
        Checks both Redis and local file system to ensure we don't lose URL history.
        """
        # Check Redis first
        history = session.get('url_history', {})
        url = history.get(space_id)
        if url:
            return url
        
        # Check local file as backup
        try:
            with open(URL_HISTORY_FILE, 'r') as f:
                file_history = json.load(f)
                url = file_history.get(space_id)
                if url:
                    # Sync back to Redis if found in file
                    history[space_id] = url
                    session.set('url_history', history)
                    st.session_state.url_history = history
                    return url
        except Exception as e:
            logger.error(f"Error reading URL history file: {e}")
        
        return "Unknown URL"

    def read_quotes(quotes_path: str) -> List[str]:
        """Read quotes from a file and return as a list.
        
        Args:
            quotes_path: Path to the quotes file
            
        Returns:
            List of quotes, or empty list if file doesn't exist
        """
        try:
            with open(quotes_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return [q.strip() for q in content.split('\n\n') if q.strip()]
        except Exception as e:
            logger.error(f"Error reading quotes: {e}")
            return []

    def display_quotes(quotes: List[str], container: Any):
        """Display quotes in a Streamlit container with copy buttons.
        
        Args:
            quotes: List of quotes to display
            container: Streamlit container to display in
        """
        for i, quote in enumerate(quotes, 1):
            with container:                
                with stylable_container(
                    key=f"quote_{i}",
                    css_styles="""
                        code {
                            white-space: pre-wrap !important;
                        }
                        """
                ):
                    st.code(quote, language="text")

    class ProgressCallbackImpl(Protocol):
        """Protocol for progress tracking callbacks."""
        def __call__(self, stage: str, progress: float, status: str) -> None: ...

    class StreamlitProgressCallback:
        """Progress callback that updates Streamlit UI."""
        def __init__(self, container: Any):
            self.container = container
            self.progress_bar = None
            
        def __call__(self, stage: str, progress: float, status: str) -> None:
            if not self.progress_bar:
                self.progress_bar = self.container.progress(0)
            
            # Update progress bar with emoji indicators for each stage
            stage_emoji = {
                "download": "⬇️",
                "transcribe": "🎯",
                "quotes": "✍️",
                "summary": "📝",
                "complete": "✅",
                "error": "❌"
            }
            
            emoji = stage_emoji.get(stage, "")
            # Update progress bar with emoji
            if self.progress_bar is not None:
                self.progress_bar.progress(progress, f"{emoji} {stage.title()}: {status}")
            
            # Only show completion/error messages outside progress bar
            if stage == "complete":
                self.container.success("✅ Processing complete!")
            elif stage == "error":
                self.container.error(f"❌ Error: {status}")

    def check_process_state(space_id: Optional[str]) -> ProcessState:
        """Check the current state of processing for a space."""
        if not space_id:
            return create_process_state(
                stage='unknown',
                status='error',
                stage_status='No space ID provided',
                error='No space ID provided',
                hostname=HOSTNAME
            )
            
        try:
            state = get_process_state(str(STORAGE_DIR), space_id)
            
            # For active processes, check more frequently
            if state['status'] == 'processing':
                # Check if all files exist, which would indicate completion
                storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
                files_exist = {
                    'audio': os.path.exists(storage_paths['audio_path']),
                    'transcript': os.path.exists(storage_paths['transcript_path']),
                    'quotes': os.path.exists(storage_paths['quotes_path']),
                    'summary': os.path.exists(storage_paths['summary_path'])
                }
                
                # If all required files exist but state shows processing, update to complete
                if files_exist['transcript'] and files_exist['quotes']:
                    state.update({
                        'status': 'complete',
                        'stage': 'complete',
                        'progress': 1.0,
                        'files': files_exist,
                        'last_updated': datetime.now().isoformat()
                    })
                    save_process_state(str(STORAGE_DIR), space_id, state)
                    return state
                
                # Adjust polling frequency based on stage
                if state.get('stage') == 'download':
                    time.sleep(0.5)  # Poll every 0.5s during download
                else:
                    time.sleep(2)  # Poll every 2s for other stages
            else:
                time.sleep(5)  # Poll every 5s for completed/failed states
            
            return state
        except Exception as e:
            logger.error(f"Error checking process state: {e}")
            return create_process_state(
                stage='unknown',
                progress=0.0,
                status='error',
                stage_status=f"Error checking state: {str(e)}",
                error=str(e),
                hostname=HOSTNAME
            )

    def display_process_state(state: ProcessState, container: Any) -> None:
        """Display the current process state in the UI."""
        status = state.get('status', 'unknown')
        stage = state.get('stage', 'unknown')
        progress = state.get('progress', 0.0)
        stage_status = state.get('stage_status', '')
        error = state.get('error')
        
        # Show error state
        if status == 'error':
            container.error(f"❌ Error: {error}")
        # Show progress bar for active processes
        if status == 'processing':
            # Create two columns for progress
            col1, col2 = container.columns([3, 1])
            
            # Main progress bar
            progress_bar = col1.progress(0.0)
            progress_bar.progress(float(progress))
            
            # Status text
            if stage == 'download':
                col2.markdown(f"⏬ {stage_status}")
            elif stage == 'transcribe':
                col2.markdown(f"🎯 {stage_status}")
            elif stage == 'generate':
                col2.markdown(f"✍️ {stage_status}")
            
            # Show detailed status below
            if stage_status:
                container.text(stage_status)
            
            # Show any console output in a scrollable area
            if state.get('console_output'):
                with stylable_container(
                    "console-output",
                    """
                    pre {
                        max-height: 200px;
                        overflow-y: auto;
                        padding: 10px;
                        background-color: #f0f0f0;
                        border-radius: 5px;
                    }
                    """
                ):
                    container.code(state['console_output'])
        
        # Show error state
        elif status == 'error':
            container.error(f"❌ Error in {stage}: {error}")
        
        # Show completion
        elif status == 'complete':
            container.success("✅ Processing complete!")

    def sync_active_processes():
        """Sync active processes list with current state."""
        try:
            # Get all state files
            state_dir = Path(STORAGE_DIR) / "state"
            if not state_dir.exists():
                return
            
            current_processes = []
            for state_file in state_dir.glob("*.json"):
                try:
                    space_id = state_file.stem
                    state = get_process_state(str(STORAGE_DIR), space_id)
                    
                    # Get original URL
                    url = get_url_from_history(space_id)
                    
                    # Check if process is active or needs attention
                    if state['status'] == 'processing':
                        # Check if process is stale
                        last_updated = datetime.fromisoformat(state.get('last_updated', '2000-01-01'))
                        time_since_update = datetime.now() - last_updated
                        
                        if time_since_update <= timedelta(minutes=5):
                            # Active process
                            current_processes.append((space_id, state, url))
                        else:
                            # Stale process - mark as error
                            state.update({
                                'status': 'error',
                                'error': 'Process became stale',
                                'stage': None,
                                'progress': 0.0
                            })
                            save_process_state(str(STORAGE_DIR), space_id, state)
                            
                    elif state['status'] == 'error':
                        # Keep error states visible for a while
                        last_updated = datetime.fromisoformat(state.get('last_updated', '2000-01-01'))
                        time_since_update = datetime.now() - last_updated
                        
                        if time_since_update <= timedelta(minutes=30):  # Show errors for 30 minutes
                            current_processes.append((space_id, state, url))
                            
                except Exception as e:
                    logger.error(f"Error processing state file {state_file}: {e}")
                    continue
                
            # Update session state
            st.session_state.active_processes = current_processes
            
        except Exception as e:
            logger.error(f"Error in sync_active_processes: {e}")
        
    # Add periodic state sync
    if 'last_sync_time' not in st.session_state:
        st.session_state.last_sync_time = None
    
    current_time = time.time()
    if (st.session_state.last_sync_time is None or 
        current_time - st.session_state.last_sync_time >= 5):  # Sync every 5 seconds
        sync_active_processes()
        st.session_state.last_sync_time = current_time

    def get_space_id(url: str) -> str:
        """Extract space ID from URL and create a hash for storage."""
        # Clean the URL first
        url = url.strip()
        # Extract video ID for YouTube URLs
        if 'youtube.com' in url or 'youtu.be' in url:
            if 'v=' in url:
                video_id = url.split('v=')[1].split('&')[0]
            else:
                video_id = url.split('/')[-1].split('?')[0]
            return hashlib.md5(video_id.encode()).hexdigest()
        # For other URLs, hash the entire URL
        return hashlib.md5(url.encode()).hexdigest()

    def get_storage_paths(storage_root: str, space_id: str) -> StoragePaths:
        """Get paths for storing space data."""
        # Ensure we're using the hashed space_id for storage
        return {
            'audio_path': str(DOWNLOADS_DIR / f"{space_id}.mp3"),
            'transcript_path': str(TRANSCRIPTS_DIR / f"{space_id}.txt"),
            'quotes_path': str(QUOTES_DIR / f"{space_id}.txt"),
            'summary_path': str(SUMMARIES_DIR / f"{space_id}.json")
        }

    def process_space_with_ui(url: str, _progress_container: Any) -> Optional[StoragePaths]:
        """
    End-to-end processing pipeline with Streamlit-friendly status updates.
    Uses the “single-dict hand-off” pattern so every Celery task signature
    stays (self, task_result: Dict[str, Any]).
        """
        try:
            log_processing_step("Space processing", "started", f"URL: {url}")

        # ------------------------------------------------------------------ #
        # 0. Priming: minimal state + deterministic paths                    #
        # ------------------------------------------------------------------ #
            space_id = get_space_id(url)                          # local guess
            storage_dir = str(STORAGE_DIR)                        # root folder

        # Create an initial placeholder in the state store
            state_manager.set_state(
                space_id=space_id,
                status="INIT",
                metadata={"url": url, "space_id": space_id, "task_id": None},
            )

        # ------------------------------------------------------------------ #
        # 1. Download                                                        #
        # ------------------------------------------------------------------ #
            state_manager.set_state(space_id=space_id, status="DOWNLOADING")
            download_task = download_media.delay(url, storage_dir)
            state_manager.set_state(
                space_id=space_id,
                status="DOWNLOADING",
                metadata={"url": url, "space_id": space_id, "task_id": download_task.id},
            )

            download_result: Dict[str, Any] = download_task.get()  # blocks
            space_id = download_result["space_id"]                 # definitive
            storage_paths = get_storage_paths(storage_dir, space_id)
            save_url_history(space_id, url)

        # ------------------------------------------------------------------ #
        # 2. Transcribe                                                      #
        # ------------------------------------------------------------------ #
            state_manager.set_state(space_id=space_id, status="TRANSCRIBING")
            transcribe_task = celery_transcribe.delay(download_result)
            state_manager.set_state(
                space_id=space_id,
                status="TRANSCRIBING",
                metadata={"url": url, "space_id": space_id, "task_id": transcribe_task.id},
            )

            transcribe_result: Dict[str, Any] = transcribe_task.get()

        # ------------------------------------------------------------------ #
        # 3. Generate quotes                                                 #
        # ------------------------------------------------------------------ #
            state_manager.set_state(space_id=space_id, status="GENERATING_QUOTES")
            quotes_task = celery_generate_quotes.delay(transcribe_result)
            state_manager.set_state(
                space_id=space_id,
                status="GENERATING_QUOTES",
                metadata={"url": url, "space_id": space_id, "task_id": quotes_task.id},
            )

            quotes_result: Dict[str, Any] = quotes_task.get()

        # ------------------------------------------------------------------ #
        # 4. Generate summary                                                #
        # ------------------------------------------------------------------ #
            state_manager.set_state(space_id=space_id, status="GENERATING_SUMMARY")
            summary_task = celery_generate_summary.delay(quotes_result)
            state_manager.set_state(
                space_id=space_id,
                status="GENERATING_SUMMARY",
                metadata={"url": url, "space_id": space_id, "task_id": summary_task.id},
            )

            summary_task.get()  # we don’t need its return value here

        # ------------------------------------------------------------------ #
        # 5. Done                                                            #
        # ------------------------------------------------------------------ #
            state_manager.set_state(
                space_id=space_id,
                status="COMPLETE",
                metadata={"url": url, "space_id": space_id, "task_id": None},
            )

            return storage_paths

        except Exception as e:
            logger.exception("Error processing space")
            # ensure we still have a space_id for state recording
            space_id = space_id if "space_id" in locals() else get_space_id(url)
            state_manager.set_state(
               space_id=space_id,
                status="ERROR",
               error=str(e),
                metadata={"url": url, "space_id": space_id, "task_id": None},
            )
            return None

    def process_space_with_ui1(url: str, _progress_container: Any) -> Optional[StoragePaths]:
        """Process media URL with Streamlit UI updates."""
        try:
            log_processing_step("Space processing", "started", f"URL: {url}")
            space_id = get_space_id(url)
            logger.info(f"Generated space_id: {space_id} for URL: {url}")
            
            # Initialize state
            state_manager.set_state(
                space_id=space_id,
                status='INIT',
                metadata={
                    'url': url,
                    'space_id': space_id,
                    'task_id': None
                }
            )
            
            # Get storage paths
            storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
            
            # Start download task
            state_manager.set_state(space_id=space_id, status='DOWNLOADING')
            download_task = download_media.delay(url, storage_paths['audio_path'])
            
            # Update state with task ID
            state_manager.set_state(
                space_id=space_id,
                status='DOWNLOADING',
                metadata={
                    'url': url,
                    'space_id': space_id,
                    'task_id': download_task.id
                }
            )
            
            # Save URL to history
            save_url_history(space_id, url)
            
            # Wait for download to complete
            download_task.get()
            
            # Start transcription
            state_manager.set_state(space_id=space_id, status='TRANSCRIBING')
            transcribe_task = celery_transcribe.delay(storage_paths['audio_path'], storage_paths['transcript_path'])
            
            # Update state with task ID
            state_manager.set_state(
                space_id=space_id,
                status='TRANSCRIBING',
                metadata={
                    'url': url,
                    'space_id': space_id,
                    'task_id': transcribe_task.id
                }
            )
            
            # Wait for transcription
            transcribe_task.get()
            
            # Generate quotes
            state_manager.set_state(space_id=space_id, status='GENERATING_QUOTES')
            quotes_task = celery_generate_quotes.delay(storage_paths['transcript_path'], storage_paths['quotes_path'])
            
            # Update state with task ID
            state_manager.set_state(
                space_id=space_id,
                status='GENERATING_QUOTES',
                metadata={
                    'url': url,
                    'space_id': space_id,
                    'task_id': quotes_task.id
                }
            )
            
            # Wait for quotes
            quotes_task.get()
            
            # Generate summary
            state_manager.set_state(space_id=space_id, status='GENERATING_SUMMARY')
            summary_task = generate_summary.delay(storage_paths['transcript_path'], storage_paths['summary_path'])
            
            # Update state with task ID
            state_manager.set_state(
                space_id=space_id,
                status='GENERATING_SUMMARY',
                metadata={
                    'url': url,
                    'space_id': space_id,
                    'task_id': summary_task.id
                }
            )
            
            # Wait for summary
            summary_task.get()
            
            # Mark as complete
            state_manager.set_state(
                space_id=space_id,
                status='COMPLETE',
                metadata={
                    'url': url,
                    'space_id': space_id,
                    'task_id': None
                }
            )
            
            return storage_paths
            
        except Exception as e:
            logger.error(f"Error processing space: {e}")
            # Update state with error
            state_manager.set_state(
                space_id=space_id,
                status='ERROR',
                error=str(e),
                metadata={
                    'url': url,
                    'space_id': space_id,
                    'task_id': None
                }
            )
            return None

    def check_process_status(space_id: str) -> Dict[str, Any]:
        """Check the status of a process."""
        return state_manager.check_task_status(space_id)

    # Main app
    logger.info("Starting main application UI...")
    st.title("🎙️ LinkToQuote")
    st.write("Generate quotes and clips from any media url -  no listening required!")

    # Create sidebar
    with st.sidebar:
        st.markdown("### 🔄 Active Processes")
        
        # Sync active processes
        sync_active_processes()
        
        # Update sidebar with active processes
        if st.session_state.active_processes:
            for space_id, state, original_url in st.session_state.active_processes:
                with st.container():
                    # Check if process is still active
                    current_state = get_process_state(str(STORAGE_DIR), space_id)
                    if current_state['status'] not in ['processing', 'error']:
                        continue
                        
                    if original_url:
                        st.info(f"🔗 Processing: {original_url[:50]}...")
                    else:
                        st.info(f"🔄 Processing space: {space_id[:8]}")
                        
                    # Show stage and progress
                    stage = current_state.get('stage')
                    progress = current_state.get('progress', 0.0)
                    stage_display = stage if stage else "Unknown"
                    st.progress(progress, f"{stage_display.title()}: {progress*100:.0f}%")
                    
                    # Add view details button
                    if st.button("📂 View Details", key=f"view_{space_id}_sidebar"):
                        st.session_state.current_space_id = space_id
                        st.session_state.processing_complete = current_state['status'] == 'complete'
                        st.session_state.url_history = load_url_history()
                        st.rerun()
                    
                    # Show last update time
                    last_updated_str = current_state.get('last_updated')
                    if last_updated_str:
                        last_updated = datetime.fromisoformat(last_updated_str)
                        time_since = datetime.now() - last_updated
                        if time_since < timedelta(minutes=1):
                            st.caption(f"Last updated: {time_since.seconds} seconds ago")
                        elif time_since < timedelta(hours=1):
                            st.caption(f"Last updated: {time_since.seconds // 60} minutes ago")
                        else:
                            st.caption(f"Last updated: {time_since.seconds // 3600} hours ago")
                
                st.markdown("---")
        else:
            st.info("No active processes")

    # Create tabs for main content
    main_tab, summary_tab, logs_tab, history_tab, help_tab = st.tabs(["🎯 Main", "📝 Summary", "🔍 Logs", "📚 History", "❓ Help"])

    with main_tab:
        selected_option = None
        media_info = {}
        # Add dropdown for previous media
        if DOWNLOADS_DIR.exists():
            media_files: List[Path] = list(DOWNLOADS_DIR.glob("*.mp3"))
            if media_files:
                # Load URL history at the start
                url_history = load_url_history()
                
                # Create two columns for the media selector
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Create options list with better formatting
                    options = ["🆕 New URL"]
                    media_info: Dict[str, Dict[str, Union[str, float]]] = {}  # Store full info for each media
                    
                    media_files_list: List[Path] = media_files
                    for media_path in media_files_list:
                        space_id = media_path.stem
                        url = url_history.get(space_id, "Unknown URL")
                        size_mb = media_path.stat().st_size / (1024 * 1024)
                        
                        # Get the current state to show status
                        state = get_process_state(str(STORAGE_DIR), space_id)
                        status = state.get('status', 'unknown')
                        
                        # Create status indicator
                        status_indicator = ""
                        if status == 'complete':
                            status_indicator = "(✅ Complete)"
                        elif status == 'processing':
                            status_indicator = "(⏳ Processing)"
                        elif status == 'error':
                            status_indicator = "(❌ Error)"
                        
                        # Create a display name that shows the full URL
                        if url != "Unknown URL":
                            display_name = f"🔗 {url} {status_indicator}"
                        else:
                            display_name = f"📁 Space {space_id} {status_indicator}"
                        
                        options.append(display_name)
                        media_info[display_name] = {
                            'space_id': space_id,
                            'url': url,
                            'size': size_mb,
                            'status': status
                        }
                    
                    # Add the dropdown with better labeling
                    selected_option = st.selectbox(
                        "Select previous media or enter new URL:",
                        options,
                        index=0,
                        key=f"media_selector_{st.session_state.get('media_selector_key', 0)}"
                    )
                
                with col2:
                    # Show media details and load button if a previous media is selected
                    if selected_option != "🆕 New URL":
                        info = media_info[selected_option]
                        st.caption(f"Size: {info['size']:.1f} MB")
                        
                        # Show appropriate action button based on status
                        if info['status'] == 'error':
                            if st.button("🔄 Resume Processing", key="resume_processing"):
                                # Resume processing from the last successful stage
                                space_id = str(info['space_id'])  # Ensure string type
                                url = str(info['url'])
                                
                                # Check if we have a valid URL
                                if url == "Unknown URL":
                                    st.error("❌ Cannot resume processing - original URL is unknown. Please provide a new URL.")
                                    st.stop()
                                    
                                st.session_state.current_space_id = space_id
                                st.session_state.loaded_space_id = space_id
                                st.session_state.processing_complete = False
                                # Get storage paths
                                storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
                                # Process the URL again to resume from where it left off
                                with st.spinner("Resuming processing..."):
                                    result_paths = process_space_with_ui(url, st.empty())
                                    if result_paths:
                                        st.session_state.url_history = load_url_history()
                                        st.rerun()
                        else:
                            if st.button("📂 Load", key="load_selected"):
                                # Load the media and update all necessary state
                                space_id = str(info['space_id'])  # Ensure string type
                                st.session_state.current_space_id = space_id
                                st.session_state.loaded_space_id = space_id
                                st.session_state.processing_complete = info['status'] == 'complete'
                                # Get storage paths to check what's available
                                storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
                                # Check which files exist
                                files_exist = {
                                    'audio': os.path.exists(storage_paths['audio_path']),
                                    'transcript': os.path.exists(storage_paths['transcript_path']),
                                    'quotes': os.path.exists(storage_paths['quotes_path']),
                                    'summary': os.path.exists(storage_paths['summary_path'])
                                }
                                # Update state based on available files
                                state = get_process_state(str(STORAGE_DIR), space_id)
                                state.update({
                                    'files': files_exist,
                                    'last_updated': datetime.now().isoformat()
                                })
                                save_process_state(str(STORAGE_DIR), space_id, state)
                                # Load the URL history to ensure it's available
                                st.session_state.url_history = load_url_history()
                                # Force refresh of the UI
                                st.session_state.media_selector_key = st.session_state.get('media_selector_key', 0) + 1
                                st.rerun()

        # Add URL input field - show for both new URL and selected URL
        url_value = ""
        if selected_option and selected_option != "🆕 New URL":
            url = str(media_info[selected_option]['url'])
            # Only show URL if it's known
            if url != "Unknown URL":
                url_value = url
        
        # Create columns for URL input and process button
        col1, col2 = st.columns([4, 1])
        with col1:
            url = st.text_input("Paste Media URL:", value=url_value, key="url_input")
        with col2:
            process_button = st.button("🚀 Process", key="process_button", use_container_width=True)
        
        # Process button handler
        if url and process_button:
            # Create progress container
            progress_container = st.container()
            
            # Process the URL
            try:
                result_paths = process_space_with_ui(url, progress_container)
                if result_paths:
                    # Save URL to history
                    space_id = get_space_id(url)
                    save_url_history(space_id, url)
                    st.session_state.current_space_id = space_id
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing URL: {str(e)}")
        
        # If we have a current_space_id from sidebar or loading, show its details
        if st.session_state.current_space_id:
            space_id = str(st.session_state.current_space_id)  # Ensure string type
            state = get_process_state(str(STORAGE_DIR), space_id)
            space_url = st.session_state.url_history.get(space_id, '')
            storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)

            # Show the URL as a copyable field if available
            if space_url and space_url != "Unknown URL":
                st.markdown("### Currently Viewing/Processing")
                st.text_input("", value=space_url, disabled=True, label_visibility="collapsed")
                st.markdown("---")

            # Show process status if still processing
            if state['status'] == 'processing':
                progress_container = st.container()
                with st.status("Processing media...", expanded=True) as status:
                    try:
                        # Display current state
                        display_process_state(state, progress_container)
                    except Exception as e:
                        st.error(f"Error displaying process: {str(e)}")
                        status.update(label="Error!", state="error")
            # Show error state with resume option
            elif state['status'] == 'error':
                st.error(f"❌ Error: {state.get('error', 'Unknown error')}")
                if space_url and space_url != "Unknown URL":
                    if st.button("🔄 Resume Processing"):
                        # Resume processing from the last successful stage
                        with st.spinner("Resuming processing..."):
                            result_paths = process_space_with_ui(space_url, st.empty())
                            if result_paths:
                                st.rerun()
                else:
                    st.warning("⚠️ Cannot resume processing - original URL is unknown. Please provide a new URL.")
            # Show results if complete or files exist
            elif state['status'] == 'complete' or any(os.path.exists(str(p)) for p in storage_paths.values()):
                if state['status'] == 'complete':
                    st.success("✅ Space processed successfully!")
                
                # Display results tabs
                content_tab1, content_tab2, content_tab3 = st.tabs(["📝 Quotes", "🎵 Audio", "📄 Transcript"])
                
                # Show transcript tab if transcript exists
                with content_tab3:
                    st.subheader("Transcript")
                    if os.path.exists(storage_paths['transcript_path']):
                        with open(storage_paths['transcript_path'], 'r') as f:
                            transcript = f.read()
                        
                        # Split transcript into chunks
                        chunks = chunk_transcript(transcript)
                        total_chunks = len(chunks)
                        
                        # Add chunk navigation
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"Transcript length: {len(transcript):,} characters ({total_chunks} pages)")
                        with col2:
                            chunk_idx = st.number_input("Page", min_value=1, max_value=total_chunks, value=1) - 1
                        
                        # Show current chunk with some context
                        st.text_area(
                            f"Transcript (Page {chunk_idx + 1}/{total_chunks})", 
                            chunks[chunk_idx],
                            height=400
                        )
                        
                        # Navigation buttons
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            if chunk_idx > 0:
                                if st.button("⬅️ Previous Page"):
                                    st.session_state.chunk_idx = max(0, chunk_idx - 1)
                                    st.rerun()
                        with col3:
                            if chunk_idx < total_chunks - 1:
                                if st.button("Next Page ➡️"):
                                    st.session_state.chunk_idx = min(total_chunks - 1, chunk_idx + 1)
                                    st.rerun()
                        
                        # Download button for full transcript
                        st.download_button(
                            "⬇️ Download Full Transcript",
                            transcript,
                            file_name=f"transcript_{space_id[:8]}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.info("Transcript is not available yet.")
                        if space_url and space_url != "Unknown URL":
                            if st.button("🔄 Generate Transcript"):
                                with st.spinner("Generating transcript..."):
                                    result_paths = process_space_with_ui(space_url, st.empty())
                                    if result_paths:
                                        st.rerun()
                        else:
                            st.warning("⚠️ Cannot generate transcript - original URL is unknown. Please provide a new URL.")
                
                # Show quotes tab if quotes exist
                with content_tab1:
                    st.subheader("Generated Quotes")
                    if os.path.exists(storage_paths['quotes_path']):
                        # Add regenerate button at the top
                        col1, col2 = st.columns([4, 1])
                        with col2:
                            if space_url and space_url != "Unknown URL":
                                if st.button("🔄 Regenerate", key="regenerate_quotes"):
                                    st.session_state.regenerating_quotes = True
                                    
                                    # Create containers for progress and logs
                                    status_container = st.empty()
                                    progress_container = st.empty()
                                    log_container = st.empty()
                                    
                                    with st.spinner("Regenerating quotes..."):
                                        # Create progress bar
                                        progress_bar = progress_container.progress(0)
                                        
                                        # Read transcript
                                        status_container.info("Reading transcript...")
                                        progress_bar.progress(0.2)
                                        
                                        try:
                                            # Create a log expander
                                            with log_container.expander("Generation Logs", expanded=True):
                                                st.write("Starting quote generation...")
                                                
                                                # Capture stdout to show logs
                                                import io
                                                import sys
                                                old_stdout = sys.stdout
                                                sys.stdout = mystdout = io.StringIO()
                                                
                                                try:
                                                    quotes = regenerate_quotes(
                                                        storage_paths['transcript_path'],
                                                        storage_paths['quotes_path'],
                                                        space_url
                                                    )
                                                    
                                                    # Get the logs
                                                    logs = mystdout.getvalue()
                                                    with stylable_container(
                                                        key="generation_logs",
                                                        css_styles="""
                                                            code {
                                                                white-space: pre-wrap !important;
                                                            }
                                                            """
                                                    ):
                                                        st.code(logs)
                                                    
                                                    if quotes:
                                                        status_container.success("✨ Quotes regenerated successfully!")
                                                        progress_bar.progress(1.0)
                                                    else:
                                                        status_container.error("❌ Failed to generate quotes. Check the logs for details.")
                                                        progress_bar.progress(1.0)
                                                        
                                                finally:
                                                    sys.stdout = old_stdout
                                        
                                        except Exception as e:
                                            status_container.error(f"❌ Error: {str(e)}")
                                            progress_bar.progress(1.0)
                                        
                                    st.session_state.regenerating_quotes = False
                                    time.sleep(1)  # Give time for status messages to be read
                                    st.rerun()
                            else:
                                st.warning("⚠️ Cannot regenerate quotes - original URL is unknown. Please provide a new URL.")
                        
                        # Read and display quotes in the main container
                        quotes = read_quotes(storage_paths['quotes_path'])
                        if quotes:  # Only try to display if we have quotes
                            display_quotes(quotes, st.container())
                        else:
                            st.warning("No quotes found in the file. Try regenerating the quotes.")
                    else:
                        st.info("Quotes are not available yet.")
                        if space_url and space_url != "Unknown URL":
                            if st.button("🔄 Generate Quotes"):
                                with st.spinner("Generating quotes..."):
                                    result_paths = process_space_with_ui(space_url, st.empty())
                                    if result_paths:
                                        st.rerun()
                        else:
                            st.warning("⚠️ Cannot generate quotes - original URL is unknown. Please provide a new URL.")
                
                # Show audio tab if audio exists
                with content_tab2:
                    st.subheader("Audio")
                    if os.path.exists(storage_paths['audio_path']):
                        audio_size = os.path.getsize(storage_paths['audio_path']) / (1024 * 1024)  # MB
                        st.write(f"Audio file size: {audio_size:.1f} MB")
                        
                        with open(storage_paths['audio_path'], 'rb') as audio_file:
                            audio_data = audio_file.read()
                            st.download_button(
                                "⬇️ Download Full Recording",
                                audio_data,
                                file_name=f"space_{space_id[:8]}.mp3",
                                mime="audio/mpeg"
                            )
                    else:
                        st.info("Audio is not available yet.")
                        if space_url and space_url != "Unknown URL":
                            if st.button("🔄 Download Audio"):
                                with st.spinner("Downloading audio..."):
                                    result_paths = process_space_with_ui(space_url, st.empty())
                                    if result_paths:
                                        st.rerun()
                        else:
                            st.warning("⚠️ Cannot download audio - original URL is unknown. Please provide a new URL.")

    with summary_tab:
        st.subheader("📝 Content Summary")
        
        if st.session_state.processing_complete and st.session_state.current_space_id:
            storage_paths = get_storage_paths(str(STORAGE_DIR), st.session_state.current_space_id)
            
            # Check for required files first
            files_status = {
                'transcript': os.path.exists(storage_paths['transcript_path']),
                'quotes': os.path.exists(storage_paths['quotes_path']),
                'summary': os.path.exists(storage_paths['summary_path'])
            }
            
            if not files_status['transcript'] or not files_status['quotes']:
                st.warning("⚠️ Missing required files. Please process content first to generate a summary.")
                # Show which files are missing
                if not files_status['transcript']:
                    st.error("Missing transcript file")
                if not files_status['quotes']:
                    st.error("Missing quotes file")
            else:
                # Load existing summary if available
                existing_summary = load_summary(storage_paths['summary_path']) if files_status['summary'] else None
                
                # Show summary status
                if existing_summary:
                    st.info("✅ Summary already exists")
                else:
                    st.info("ℹ️ No summary generated yet")
                
                # Add generate/regenerate button
                button_text = "🔄 Regenerate Summary" if existing_summary else "✨ Generate Summary"
                if st.button(button_text):
                    try:
                        with st.spinner("Reading files..."):
                            # Read transcript and quotes
                            with open(storage_paths['transcript_path'], 'r', encoding='utf-8') as f:
                                transcript = f.read()
                            quotes = read_quotes(storage_paths['quotes_path'])
                            
                            if not transcript.strip():
                                st.error("❌ Transcript file is empty")
                                st.stop()
                            if not quotes:
                                st.error("❌ No quotes found")
                                st.stop()
                        
                        with st.spinner("Generating summary..."):
                            summary = generate_summary(transcript, quotes, storage_paths['summary_path'])
                            
                            if isinstance(summary, dict) and summary.get('overview') != "Error generating summary":
                                # Save the summary
                                save_summary(summary, storage_paths['summary_path'])
                                st.success("✨ Summary generated successfully!")
                                existing_summary = summary  # Update the displayed summary
                                st.rerun()  # Refresh to show new summary
                            else:
                                st.error("Failed to generate summary. Please try again.")
                                logger.error("Summary generation failed: Error response from API")
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
                        logger.exception("Error during summary generation")
                
                # Display existing summary if available
                if existing_summary:
                    with st.container():
                        # Display overview without redundant header
                        st.markdown("### Content Overview")
                        st.write(existing_summary['overview'])
                        
                        # Display key points
                        st.markdown("### Key Points")
                        for point in existing_summary['key_points']:
                            st.markdown(f"• {point}")
                        
                        # Add download button for summary
                        summary_text = f"Content Overview:\n{existing_summary['overview']}\n\nKey Points:\n"
                        summary_text += '\n'.join(f"• {point}" for point in existing_summary['key_points'])
                        
                        st.download_button(
                            "⬇️ Download Summary",
                            summary_text,
                            file_name=f"summary_{st.session_state.current_space_id[:8]}.txt",
                            mime="text/plain"
                        )
        else:
            st.info("Process content in the Main tab to generate a summary.")

    with logs_tab:
        st.markdown("### 🔍 Processing Logs")
        
        # Create an expander for detailed logs with word wrapping
        with st.expander("Detailed Logs", expanded=True):
            try:
                with open('app.log', 'r') as f:
                    logs_content = f.readlines()
                    # Get last 50 lines instead of 20 for more context
                    recent_logs = logs_content[-50:]
                    
                    # Process logs to highlight status indicators
                    processed_logs = []
                    for log in recent_logs:
                        # Highlight success indicators
                        log = log.replace("✓", "**✓**")
                        # Highlight progress indicators
                        log = log.replace("⏳", "**⏳**")
                        # Highlight error indicators
                        log = log.replace("❌", "**❌**")
                        processed_logs.append(log)
                    
                    with stylable_container(
                        key="detailed_logs",
                        css_styles="""
                            code {
                                white-space: pre-wrap !important;
                            }
                            """
                    ):
                        st.code(''.join(processed_logs), language='text')
                        
                    # Add auto-refresh for logs
                    if st.session_state.get('current_task_id'):
                        time.sleep(1)  # Brief delay to prevent too frequent updates
                        st.rerun()
                    
            except Exception as e:
                st.warning("No logs available yet")

    with history_tab:
        st.markdown("### 📚 Previously Downloaded Media")
        
        # List all downloaded media
        if DOWNLOADS_DIR.exists():
            history_media_files: List[Path] = list(DOWNLOADS_DIR.glob("*.mp3"))
            if history_media_files:
                # Add search/filter box
                search = st.text_input("🔍 Search media files", 
                                     help="Filter by URL or ID")
                
                # Create a radio selection for media files
                media_options = {}
                for media_file in history_media_files:
                    space_id = media_file.stem
                    original_url = get_url_from_history(space_id)
                    size_mb = media_file.stat().st_size / (1024 * 1024)
                    
                    # Create a formatted label with URL preview
                    url_preview = original_url
                    if len(url_preview) > 60:
                        url_preview = url_preview[:57] + "..."
                    
                    label = f"📁 {space_id}\n💾 {size_mb:.1f} MB\n🔗 {url_preview}"
                    media_options[label] = str(media_file)
                
                # Filter options based on search
                if search:
                    media_options = {k: v for k, v in media_options.items() 
                                   if search.lower() in k.lower() or 
                                   search.lower() in get_url_from_history(Path(v).stem).lower()}
                
                if media_options:
                    selected = st.radio(
                        "Select a media file to manage:",
                        options=list(media_options.keys()),
                        index=None,
                        label_visibility="collapsed"
                    )
                    
                    if selected:
                        selected_path = Path(media_options[selected])
                        space_id = selected_path.stem
                        
                        # Create a nice looking container for file details
                        with st.container():
                            st.markdown("---")
                            st.markdown("#### Selected File Details")
                            
                            # Show full URL in an info box
                            original_url = get_url_from_history(space_id)
                            st.info(f"🔗 Original URL:\n{original_url}")
                            
                            # Show action buttons in a row
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col1:
                                if st.button("📂 Load", key="load_selected",
                                           help="Load this media in the main view"):
                                    st.session_state.current_space_id = space_id
                                    st.session_state.processing_complete = True
                                    st.rerun()
                            
                            with col2:
                                # Download button
                                with open(selected_path, 'rb') as audio_file:
                                    audio_data = audio_file.read()
                                    st.download_button(
                                        "⬇️ Download",
                                        audio_data,
                                        file_name=selected_path.name,
                                        mime="audio/mpeg",
                                        help="Download the audio file"
                                    )
                            
                            with col3:
                                # Delete button with confirmation
                                if st.button("🗑️ Delete", key="delete_media",
                                           help="Delete all files associated with this media"):
                                    if st.button("⚠️ Confirm Delete", key="confirm_delete"):
                                        try:
                                            # Delete all associated files
                                            storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
                                            for path in storage_paths.values():
                                                if os.path.exists(str(path)):
                                                    os.remove(str(path))
                                            st.success(f"Deleted all files for {selected_path.stem}")
                                            # Clear session state if the current media was deleted
                                            if space_id == st.session_state.current_space_id:
                                                st.session_state.current_space_id = None
                                                st.session_state.processing_complete = False
                                            # Remove from URL history
                                            history = load_url_history()
                                            if space_id in history:
                                                del history[space_id]
                                                with open(URL_HISTORY_FILE, 'w') as f:
                                                    json.dump(history, f)
                                            time.sleep(1)  # Give time for the success message
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deleting files: {str(e)}")
                            
                            # Show file status
                            st.markdown("#### File Status")
                            storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
                            
                            # Check which associated files exist
                            details = {
                                "Audio": os.path.exists(storage_paths['audio_path']),
                                "Transcript": os.path.exists(storage_paths['transcript_path']),
                                "Quotes": os.path.exists(storage_paths['quotes_path']),
                                "Summary": os.path.exists(storage_paths['summary_path'])
                            }
                            
                            # Create status indicators
                            status_cols = st.columns([1, 1, 1, 1])
                            for i, (file_type, exists) in enumerate(details.items()):
                                with status_cols[i]:
                                    icon = "✅" if exists else "❌"
                                    st.markdown(f"**{icon} {file_type}**")
                                    if exists:
                                        # Get size based on file type with literal keys
                                        if file_type == "Audio":
                                            size = Path(storage_paths['audio_path']).stat().st_size / 1024
                                        elif file_type == "Transcript":
                                            size = Path(storage_paths['transcript_path']).stat().st_size / 1024
                                        elif file_type == "Quotes":
                                            size = Path(storage_paths['quotes_path']).stat().st_size / 1024
                                        elif file_type == "Summary":
                                            size = Path(storage_paths['summary_path']).stat().st_size / 1024
                                        else:
                                            size = 0
                else:
                    st.warning("No media files match your search.")
            else:
                st.info("No media files downloaded yet.")
        else:
            st.warning("Downloads directory not found.")

    with help_tab:
        st.subheader("📖 Instructions")
        st.write("""
        1. Paste media URL
        2. Wait for processing (this may take a few minutes)
        3. View and download:
            - Generated quotes
            - Audio recording
            - Full transcript
        """)
        
        st.subheader("💡 Tips")
        st.write("""
        - For best results, use recent media
        - Quotes are automatically formatted for social media
        - You can download everything for offline use
        """)
        
        # Show storage info
        st.subheader("📊 Storage Info")
        total_audio = sum(f.stat().st_size for f in DOWNLOADS_DIR.glob("*.mp3")) / (1024 * 1024)  # MB
        st.write(f"Total audio storage: {total_audio:.1f} MB")
        if total_audio > 1000:  # 1 GB warning
            st.warning("⚠️ Storage is getting full. Consider cleaning old files.") 

    # Check and restore state for current process if browser was closed
    if st.session_state.current_space_id:
        current_state = get_process_state(str(STORAGE_DIR), st.session_state.current_space_id)
        if current_state['status'] == 'complete':
            st.session_state.processing_complete = True
        elif current_state['status'] == 'processing':
            st.session_state.processing_complete = False

    # Add function to handle individual file generation
    def handle_file_action(space_id: str, file_type: str, action: str) -> None:
        """Handle file action button click.
        
        Args:
            space_id: Space identifier
            file_type: Type of file
            action: Action to perform (view/generate)
        """
        storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
        path_key = f"{file_type.lower()}_path"
        
        if action == 'view':
            path = storage_paths[path_key]
            if file_type == 'Audio':
                st.audio(str(path))
            else:
                with open(str(path), 'r') as f:
                    content = f.read()
                    st.text_area(
                        f"{file_type} Content",
                        value=content,
                        height=300,
                        key=f"content_{file_type}_{space_id}"
                    )
        elif action == 'generate':
            generate_individual_file(space_id, file_type)
            st.rerun()

    # In the main tab, after starting the Celery chain:
    if st.session_state.current_task_id and st.session_state.current_space_id:
        # Check task status
        task_status = check_process_status(st.session_state.current_space_id)
        
        # Display current state
        display_state(task_status)
        
        # Get storage paths
        storage_paths = get_storage_paths(str(STORAGE_DIR), st.session_state.current_space_id)
        
        # Display file status with callback
        display_file_status(
            st.session_state.current_space_id,
            storage_paths,
            show_buttons=True,
            on_action=handle_file_action
        )
        
        # Display metadata
        state = state_manager.get_state(st.session_state.current_space_id)
        if state:
            display_metadata(state['metadata'])
        
        # Handle state transitions
        if task_status['status'] == 'complete':
            st.session_state.processing_complete = True
            st.session_state.current_task_id = None
            st.rerun()
            
        elif task_status['status'] == 'error':
            st.session_state.current_task_id = None
            
            # Show retry button
            if st.button("Retry"):
                state_manager.clear_state(st.session_state.current_space_id)
                st.session_state.current_task_id = None
                st.session_state.current_space_id = None
                st.rerun()

    # Clean up stale processes from session state
    if st.session_state.active_processes:
        active_processes = []
        for space_id, state, url in st.session_state.active_processes:
            current_state = check_process_state(space_id)
            if current_state['status'] == 'processing':
                # Validate last update time
                last_updated_str = current_state.get('last_updated')
                if last_updated_str:
                    last_updated = datetime.fromisoformat(last_updated_str)
                    time_since_update = datetime.now() - last_updated
                    if time_since_update <= timedelta(minutes=5):
                        active_processes.append((space_id, current_state, url))
                    else:
                        logger.warning(f"Removing stale process {space_id} from active processes")
        st.session_state.active_processes = active_processes

    # Add function to handle individual file generation
    def generate_individual_file(space_id: str, file_type: str) -> None:
        """Generate individual file for a space.
        
        Args:
            space_id: Space identifier
            file_type: Type of file to generate
        """
        storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
        task = None
        
        try:
            if file_type == 'Audio':
                # Get URL from state
                state = state_manager.get_state(space_id)
                if not state or not state['metadata'].get('url'):
                    st.error("No URL found for this space")
                    return
                
                # Start download task
                url = state['metadata']['url']
                task = download_media.delay(url, str(storage_paths['audio_path']))
                
                # Update state
                metadata: StateMetadata = {
                    'url': url,
                    'space_id': space_id,
                    'task_id': task.id
                }
                state_manager.set_state(
                    space_id=space_id,
                    status='DOWNLOADING',
                    metadata=metadata
                )
                
            elif file_type == 'Transcript':
                # Check audio exists
                if not storage_paths['audio_path'].exists():
                    st.error("Audio file not found")
                    return
                
                # Start transcription task
                task = transcribe_audio.delay(
                    str(storage_paths['audio_path']),
                    str(storage_paths['transcript_path'])
                )
                
                # Update state
                metadata: StateMetadata = {
                    'url': None,
                    'space_id': space_id,
                    'task_id': task.id
                }
                state_manager.set_state(
                    space_id=space_id,
                    status='TRANSCRIBING',
                    metadata=metadata
                )
                
            elif file_type == 'Quotes':
                # Check transcript exists
                if not storage_paths['transcript_path'].exists():
                    st.error("Transcript file not found")
                    return
                
                # Start quotes task
                task = celery_generate_quotes.delay(
                    str(storage_paths['transcript_path']),
                    str(storage_paths['quotes_path'])
                )
                
                # Update state
                metadata: StateMetadata = {
                    'url': None,
                    'space_id': space_id,
                    'task_id': task.id
                }
                state_manager.set_state(
                    space_id=space_id,
                    status='GENERATING_QUOTES',
                    metadata=metadata
                )
                
            elif file_type == 'Summary':
                # Check transcript exists
                if not storage_paths['transcript_path'].exists():
                    st.error("Transcript file not found")
                    return
                
                # Start summary task
                task = generate_summary.delay(
                    str(storage_paths['transcript_path']),
                    str(storage_paths['summary_path'])
                )
                
                # Update state
                metadata: StateMetadata = {
                    'url': None,
                    'space_id': space_id,
                    'task_id': task.id
                }
                state_manager.set_state(
                    space_id=space_id,
                    status='GENERATING_SUMMARY',
                    metadata=metadata
                )
            
            if task:
                # Store task ID
                st.session_state.current_task_id = task.id
                st.session_state.current_space_id = space_id
                
        except Exception as e:
            logger.error(f"Error generating {file_type}: {e}")
            st.error(f"Error generating {file_type}: {e}")

else:
    st.stop()  # Do not continue if check_password is not True.
