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
from typing import Optional, Dict, Any, List, Tuple, Callable, TextIO, BinaryIO, Protocol, Union, cast
import time
import logging
from streamlit_extras.stylable_container import stylable_container  # type: ignore
from datetime import datetime, timedelta
from celery.result import AsyncResult
from celery_worker.tasks import app as celery_app

from core.types import ProcessState, StoragePaths

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

logger.info("Starting application initialization...")

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

# Configure Streamlit page
logger.info("Configuring Streamlit page...")
st.set_page_config(
    page_title="LinkToQuotes",
    page_icon="🎙️",
    layout="wide"
)

# Authentication
def check_password():
    """Returns `True` if the user had the correct password."""
    logger.debug("Checking password authentication...")
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            logger.info("Password authentication successful")
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            logger.warning("Incorrect password attempt")
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        logger.debug("First password attempt")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        logger.debug("Previous password attempt failed")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        logger.debug("Password already verified")
        return True

# Only import core functionality after password check
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
    STORAGE_DIR = Path("storage")
    STORAGE_DIR.mkdir(exist_ok=True)
    DOWNLOADS_DIR = STORAGE_DIR / "downloads"
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    TRANSCRIPTS_DIR = STORAGE_DIR / "transcripts"
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    QUOTES_DIR = STORAGE_DIR / "quotes"
    QUOTES_DIR.mkdir(exist_ok=True)
    SUMMARIES_DIR = STORAGE_DIR / "summaries"
    SUMMARIES_DIR.mkdir(exist_ok=True)

    # Create URL history file
    URL_HISTORY_FILE = STORAGE_DIR / "url_history.json"
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
        """Save URL to history."""
        history = load_url_history()
        history[space_id] = url
        with open(URL_HISTORY_FILE, 'w') as f:
            json.dump(history, f)
        st.session_state.url_history = history

    def get_url_from_history(space_id: str) -> str:
        """Get original URL for a space_id."""
        if not hasattr(st.session_state, 'url_history'):
            st.session_state.url_history = load_url_history()
        return st.session_state.url_history.get(space_id, "Unknown URL")

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

    def check_process_state(space_id: str) -> ProcessState:
        """Check the current state of processing for a space."""
        state = get_process_state(str(STORAGE_DIR), space_id)
        
        # Only check for stale process if enough time has passed since last check
        current_time = time.time()
        if (st.session_state.last_check_time is None or 
            current_time - st.session_state.last_check_time >= 5):  # Check every 5 seconds
            
            st.session_state.last_check_time = current_time
            
            # Check Celery task status if we have a task ID
            if 'task_id' in state:
                try:
                    result = AsyncResult(state['task_id'], app=celery_app)
                    if result.ready():
                        if result.successful():
                            # Task completed successfully
                            if state['status'] != 'complete':
                                state.update({
                                    'status': 'complete',
                                    'stage': None,
                                    'progress': 1.0,
                                    'last_updated': datetime.now().isoformat()
                                })
                                save_process_state(str(STORAGE_DIR), space_id, state)
                        else:
                            # Task failed
                            error = str(result.get(propagate=False))
                            state.update({
                                'status': 'error',
                                'error': error,
                                'stage': None,
                                'progress': 0.0,
                                'last_updated': datetime.now().isoformat()
                            })
                            save_process_state(str(STORAGE_DIR), space_id, state)
                    else:
                        # Task is still running, get current task in chain
                        current_task = result.parent if result.parent else result
                        if current_task and current_task.state != 'PENDING':
                            # Update last_updated timestamp to prevent timeout
                            state['last_updated'] = datetime.now().isoformat()
                            save_process_state(str(STORAGE_DIR), space_id, state)
                except Exception as e:
                    logger.error(f"Error checking task status: {e}")
            
            # Check if process is running but stale
            if state['status'] == 'processing':
                last_updated_str = state.get('last_updated')
                if last_updated_str:
                    last_updated = datetime.fromisoformat(last_updated_str)
                    time_since_update = datetime.now() - last_updated
                    logger.debug(f"Process state last updated: {time_since_update} ago")
                    
                    # Only mark as error if significantly stale (> 1 hour)
                    if time_since_update > timedelta(hours=1):
                        logger.warning(f"Process {space_id} appears stale (no updates for {time_since_update})")
                        state.update({
                            'status': 'error',
                            'error': 'Process timed out - no updates for over an hour',
                            'stage': None,
                            'progress': 0.0,
                            'last_updated': datetime.now().isoformat()
                        })
                        save_process_state(str(STORAGE_DIR), space_id, state)
                    else:
                        logger.debug(f"Process {space_id} is still active (last update: {time_since_update} ago)")
        
        return state

    def display_process_state(state: ProcessState, container: Any) -> None:
        """Display the current process state in the UI."""
        status = state['status']
        files = state.get('files', {})
        stage = state.get('stage')
        stage_status = state.get('stage_status', '')
        
        with container:
            if status == 'processing':
                # Show stage-specific progress information
                progress = state['progress']
                
                # Create progress display based on stage
                if stage == 'download':
                    st.info("⬇️ Downloading Media")
                    st.progress(progress)
                    # Show download output in collapsible section
                    with st.expander("Download Details", expanded=False):
                        st.caption(str(stage_status))
                        if state.get('console_output'):
                            with stylable_container(
                                key="download_output",
                                css_styles="""
                                    code {
                                        white-space: pre-wrap !important;
                                        max-height: 200px;
                                        overflow-y: auto;
                                    }
                                    """
                            ):
                                st.code(state['console_output'], language="text")
                
                elif stage == 'transcribe':
                    st.info("🎯 Transcribing Audio")
                    st.progress(progress)
                    
                    # Show transcription details in collapsible section
                    with st.expander("Transcription Details", expanded=False):
                        # Show transcription details
                        last_updated_str = state.get('last_updated')
                        if last_updated_str:
                            last_updated = datetime.fromisoformat(last_updated_str)
                            time_since = datetime.now() - last_updated
                            
                            # Show elapsed time with appropriate units
                            if time_since < timedelta(minutes=1):
                                elapsed = f"{time_since.seconds} seconds"
                            elif time_since < timedelta(hours=1):
                                elapsed = f"{time_since.seconds // 60} minutes"
                            else:
                                elapsed = f"{time_since.seconds // 3600} hours"
                            
                            # Show file being transcribed
                            if files.get('audio', False):
                                paths = get_storage_paths(str(STORAGE_DIR), st.session_state.current_space_id)
                                audio_path = paths['audio_path']
                                file_size = os.path.getsize(audio_path) / (1024 * 1024)  # Convert to MB
                                st.caption(f"🎵 Transcribing: {Path(audio_path).name} ({file_size:.1f} MB)")
                            
                            # Show elapsed time and model info
                            st.caption(f"⏱️ Elapsed time: {elapsed}")
                            st.caption("🤖 Using OpenAI Whisper base model")
                            
                            # Show console output if available
                            if state.get('console_output'):
                                with stylable_container(
                                    key="transcribe_output",
                                    css_styles="""
                                        code {
                                            white-space: pre-wrap !important;
                                            max-height: 200px;
                                            overflow-y: auto;
                                        }
                                        """
                                ):
                                    st.code(state['console_output'], language="text")
                    
                    st.caption(str(stage_status))
                
                elif stage == 'quotes':
                    st.info("✍️ Generating Quotes")
                    st.progress(progress)
                    
                    # Show quote generation details in collapsible section
                    with st.expander("Quote Generation Details", expanded=True):
                        last_updated_str = state.get('last_updated')
                        if last_updated_str:
                            last_updated = datetime.fromisoformat(last_updated_str)
                            time_since = datetime.now() - last_updated
                            
                            # Show elapsed time with appropriate units
                            if time_since < timedelta(minutes=1):
                                elapsed = f"{time_since.seconds} seconds"
                            elif time_since < timedelta(hours=1):
                                elapsed = f"{time_since.seconds // 60} minutes"
                            else:
                                elapsed = f"{time_since.seconds // 3600} hours"
                            
                            # Show transcript details
                            if files.get('transcript', False):
                                paths = get_storage_paths(str(STORAGE_DIR), st.session_state.current_space_id)
                                transcript_path = paths['transcript_path']
                                with open(transcript_path, 'r', encoding='utf-8') as f:
                                    transcript = f.read()
                                
                                # Get chunk information
                                chunks = chunk_transcript(transcript)
                                total_chunks = len(chunks)
                                current_chunk = int(progress * total_chunks) if progress > 0 else 0
                                
                                # Show chunk progress
                                st.caption(f"📊 Processing chunk {current_chunk}/{total_chunks}")
                                st.caption(f"📝 Total transcript length: {len(transcript):,} characters")
                            
                            # Show elapsed time and model info
                            st.caption(f"⏱️ Elapsed time: {elapsed}")
                            st.caption("🤖 Using DeepSeek for quote extraction")
                            
                            # Show console output if available
                            if state.get('console_output'):
                                with stylable_container(
                                    key="quotes_output",
                                    css_styles="""
                                        code {
                                            white-space: pre-wrap !important;
                                            max-height: 400px;
                                            overflow-y: auto;
                                        }
                                        """
                                ):
                                    st.code(state['console_output'], language="text")
                        
                        st.caption(str(stage_status))
                
                elif stage == 'summary':
                    st.info("📝 Creating Summary")
                    st.progress(progress)
                    
                    # Show summary generation details in collapsible section
                    with st.expander("Summary Generation Details", expanded=False):
                        last_updated_str = state.get('last_updated')
                        if last_updated_str:
                            last_updated = datetime.fromisoformat(last_updated_str)
                            time_since = datetime.now() - last_updated
                            
                            # Show elapsed time with appropriate units
                            if time_since < timedelta(minutes=1):
                                elapsed = f"{time_since.seconds} seconds"
                            elif time_since < timedelta(hours=1):
                                elapsed = f"{time_since.seconds // 60} minutes"
                            else:
                                elapsed = f"{time_since.seconds // 3600} hours"
                            
                            # Show input details
                            if files.get('transcript', False) and files.get('quotes', False):
                                paths = get_storage_paths(str(STORAGE_DIR), st.session_state.current_space_id)
                                
                                # Show transcript info
                                with open(paths['transcript_path'], 'r', encoding='utf-8') as f:
                                    transcript = f.read()
                                st.caption(f"📝 Processing transcript: {len(transcript):,} characters")
                                
                                # Show quotes info
                                quotes = read_quotes(paths['quotes_path'])
                                st.caption(f"✍️ Using {len(quotes)} generated quotes")
                            
                            # Show elapsed time and model info
                            st.caption(f"⏱️ Elapsed time: {elapsed}")
                            st.caption("🤖 Using DeepSeek for summarization")
                            
                            # Show console output if available
                            if state.get('console_output'):
                                with stylable_container(
                                    key="summary_output",
                                    css_styles="""
                                        code {
                                            white-space: pre-wrap !important;
                                            max-height: 200px;
                                            overflow-y: auto;
                                        }
                                        """
                                ):
                                    st.code(state['console_output'], language="text")
                            
                            # Show stages of summary generation
                            if progress < 0.3:
                                st.caption("📊 Analyzing transcript...")
                            elif progress < 0.6:
                                st.caption("📝 Extracting key points...")
                            elif progress < 0.9:
                                st.caption("✨ Generating overview...")
                            else:
                                st.caption("💾 Saving summary...")
                        
                        st.caption(str(stage_status))
                else:
                    stage_display = stage if stage else "Unknown"
                    st.info(f"🔄 Processing: {stage_display} ({progress*100:.0f}%)")
                    st.progress(progress)
                
                # Show file status
                if files:
                    st.markdown("#### File Status")
                    cols = st.columns([1, 1, 1, 1])
                    file_types = ['audio', 'transcript', 'quotes', 'summary']
                    file_icons = {
                        'audio': '🎵',
                        'transcript': '📝',
                        'quotes': '✍️',
                        'summary': '📋'
                    }
                    for i, file_type in enumerate(file_types):
                        with cols[i]:
                            exists = files.get(file_type, False)
                            icon = "✅" if exists else "❌"
                            emoji = file_icons.get(file_type, '')
                            
                            # Add processing indicator for current stage
                            if stage and stage.lower() == file_type:
                                st.markdown(f"**{icon} {emoji} {file_type.title()}** 🔄")
                            else:
                                st.markdown(f"**{icon} {emoji} {file_type.title()}**")
                            
                            # Show file size if exists
                            if exists:
                                try:
                                    paths = get_storage_paths(str(STORAGE_DIR), st.session_state.current_space_id)
                                    # Type-safe path access based on file type
                                    if file_type == 'audio':
                                        size = Path(paths['audio_path']).stat().st_size / 1024  # KB
                                    elif file_type == 'transcript':
                                        size = Path(paths['transcript_path']).stat().st_size / 1024  # KB
                                    elif file_type == 'quotes':
                                        size = Path(paths['quotes_path']).stat().st_size / 1024  # KB
                                    elif file_type == 'summary':
                                        size = Path(paths['summary_path']).stat().st_size / 1024  # KB
                                    
                                    if size > 1024:
                                        st.caption(f"{size/1024:.1f} MB")
                                    else:
                                        st.caption(f"{size:.1f} KB")
                                except Exception:
                                    pass
                
                # Show last update time if available
                last_updated_str = state.get('last_updated')
                if last_updated_str and stage not in ['transcribe', 'quotes', 'summary']:  # Don't show twice for these stages
                    last_updated = datetime.fromisoformat(last_updated_str)
                    time_since = datetime.now() - last_updated
                    if time_since < timedelta(minutes=1):
                        st.caption(f"Last updated: {time_since.seconds} seconds ago")
                    elif time_since < timedelta(hours=1):
                        st.caption(f"Last updated: {time_since.seconds // 60} minutes ago")
                    else:
                        st.caption(f"Last updated: {time_since.seconds // 3600} hours ago")
                
            elif status == 'error':
                error_msg = state.get('error', 'Unknown error')
                st.error(f"❌ Error: {error_msg}")
                
                # Show file status if any files exist
                if any(files.values()):
                    st.markdown("#### Existing Files")
                    cols = st.columns([1, 1, 1, 1])
                    file_types = ['audio', 'transcript', 'quotes', 'summary']
                    for i, file_type in enumerate(file_types):
                        with cols[i]:
                            exists = files.get(file_type, False)
                            if exists:
                                st.markdown(f"✅ {file_type.title()}")
                
                if st.button("🔄 Retry Processing"):
                    # Check current file status
                    paths = get_storage_paths(str(STORAGE_DIR), space_id)
                    files_exist = {
                        'audio': os.path.exists(paths['audio_path']),
                        'transcript': os.path.exists(paths['transcript_path']),
                        'quotes': os.path.exists(paths['quotes_path']),
                        'summary': os.path.exists(paths['summary_path'])
                    }
                    # Clear error state but preserve file status
                    state.update({
                        'status': 'processing',
                        'stage': None,
                        'error': None,
                        'progress': 0.0,
                        'files': files_exist
                    })
                    save_process_state(str(STORAGE_DIR), space_id, state)
                    st.rerun()
                
            elif status == 'complete':
                if all(files.values()):
                    st.success("✅ Processing complete!")
                else:
                    # Some files are missing despite complete status
                    missing = [file_type for file_type in ['audio', 'transcript', 'quotes', 'summary'] 
                              if not files.get(file_type, False)]
                    st.warning(f"⚠️ Processing marked complete but missing files: {', '.join(missing)}")
                    if st.button("🔄 Reprocess Missing Files"):
                        # Check current file status
                        paths = get_storage_paths(str(STORAGE_DIR), space_id)
                        files_exist = {
                            'audio': os.path.exists(paths['audio_path']),
                            'transcript': os.path.exists(paths['transcript_path']),
                            'quotes': os.path.exists(paths['quotes_path']),
                            'summary': os.path.exists(paths['summary_path'])
                        }
                        state.update({
                            'status': 'processing',
                            'stage': None,
                            'error': None,
                            'progress': 0.0,
                            'files': files_exist
                        })
                        save_process_state(str(STORAGE_DIR), space_id, state)
                        st.rerun()

    def sync_active_processes():
        """Synchronize active processes in session state with actual process states."""
        if not hasattr(st.session_state, 'active_processes'):
            st.session_state.active_processes = []
        
        # Create new list of actually active processes
        active_processes = []
        state_dir = STORAGE_DIR / "state"
        
        if state_dir.exists():
            for state_file in state_dir.glob("*.json"):
                try:
                    space_id = state_file.stem
                    state = get_process_state(str(STORAGE_DIR), space_id)
                    original_url = get_url_from_history(space_id)
                    
                    # Check if process is actually active
                    if state['status'] == 'processing':
                        # Check Celery task status if we have a task ID
                        task_active = False
                        if 'task_id' in state:
                            try:
                                result = AsyncResult(state['task_id'], app=celery_app)
                                if not result.ready():
                                    # Get current task in chain
                                    current_task = result.parent if result.parent else result
                                    if current_task and current_task.state != 'PENDING':
                                        task_active = True
                                        # Update state with current task info
                                        state['stage'] = current_task.name.split('.')[-1].replace('_task', '')
                                        save_process_state(str(STORAGE_DIR), space_id, state)
                            except Exception as e:
                                logger.error(f"Error checking task status: {e}")
                        
                        # Check last update time
                        last_updated_str = state.get('last_updated')
                        if last_updated_str:
                            last_updated = datetime.fromisoformat(last_updated_str)
                            time_since_update = datetime.now() - last_updated
                            
                            # Only consider active if task is running or recently updated
                            if task_active or time_since_update <= timedelta(minutes=5):
                                active_processes.append((space_id, state, original_url))
                            else:
                                # Mark stale process as error
                                logger.warning(f"Found stale process {space_id}, marking as error")
                                state.update({
                                    'status': 'error',
                                    'error': 'Process state was stale or task was terminated',
                                    'stage': None,
                                    'progress': 0.0,
                                    'last_updated': datetime.now().isoformat()
                                })
                                save_process_state(str(STORAGE_DIR), space_id, state)
                
                except Exception as e:
                    logger.error(f"Error checking state file {state_file}: {e}")
                    continue
        
        # Update session state with actually active processes
        st.session_state.active_processes = active_processes

    def process_space_with_ui(url: str, _progress_container: Any) -> Optional[StoragePaths]:
        """Process media URL with Streamlit UI updates."""
        try:
            log_processing_step("Space processing", "started", f"URL: {url}")
            space_id = get_space_id(url)
            
            # Sync active processes first
            sync_active_processes()
            
            # Get storage paths
            storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
            
            # Check if all files already exist and are valid
            if all(os.path.exists(str(p)) for p in storage_paths.values()):
                st.success("✅ All files already exist for this URL!")
                st.session_state.processing_complete = True
                return storage_paths
            
            # Check current state
            state = check_process_state(space_id)
            is_active_process = any(space_id == p[0] for p in st.session_state.active_processes)
            
            # Handle different states
            if state['status'] == 'processing':
                if is_active_process:
                    # Process is actually active and in sidebar
                    display_process_state(state, _progress_container)
                    st.info("⏳ This URL is already being processed. You can view its progress in the sidebar.")
                    return None
                else:
                    # Process claims to be active but isn't in sidebar - validate state
                    task_active = False
                    if 'task_id' in state:
                        try:
                            result = AsyncResult(state['task_id'], app=celery_app)
                            if not result.ready():
                                # Get current task in chain
                                current_task = result.parent if result.parent else result
                                if current_task and current_task.state != 'PENDING':
                                    task_active = True
                                    # Update state with current task info
                                    state['stage'] = current_task.name.split('.')[-1].replace('_task', '')
                                    save_process_state(str(STORAGE_DIR), space_id, state)
                        except Exception as e:
                            logger.error(f"Error checking task status: {e}")
                    
                    if task_active:
                        # Task is still running, add to active processes
                        st.session_state.active_processes.append((space_id, state, url))
                        display_process_state(state, _progress_container)
                        st.info("⏳ This URL is already being processed. You can view its progress in the sidebar.")
                        st.rerun()
                    else:
                        # Task is not active, reset state
                        logger.warning(f"Process {space_id} has no active task, resetting...")
                        state.update({
                            'status': 'error',
                            'error': 'Process was interrupted',
                            'stage': None,
                            'progress': 0.0,
                            'last_updated': datetime.now().isoformat()
                        })
                        save_process_state(str(STORAGE_DIR), space_id, state)
            
            # Check for partial completion and cleanup failed files
            files_exist = {
                'audio': os.path.exists(storage_paths['audio_path']),
                'transcript': os.path.exists(storage_paths['transcript_path']),
                'quotes': os.path.exists(storage_paths['quotes_path']),
                'summary': os.path.exists(storage_paths['summary_path'])
            }
            
            # If previous attempt failed, clean up partial files
            if state['status'] == 'error':
                logger.info(f"Cleaning up failed process state for {space_id}")
                # Clean up partial downloads or corrupted files
                if files_exist['audio'] and state.get('stage') == 'download':
                    logger.info("Removing partial audio file")
                    try:
                        os.remove(storage_paths['audio_path'])
                        files_exist['audio'] = False
                    except Exception as e:
                        logger.error(f"Error removing audio file: {e}")
                
                # Remove other incomplete files based on stage
                if state.get('stage') in ['transcribe', 'quotes', 'summary']:
                    for file_type, exists in files_exist.items():
                        if exists and file_type != 'audio':  # Keep audio if it was fully downloaded
                            try:
                                os.remove(storage_paths[f'{file_type}_path'])
                                files_exist[file_type] = False
                            except Exception as e:
                                logger.error(f"Error removing {file_type} file: {e}")
            
            # Reset state to start/resume processing
            state.update({
                'status': 'processing',
                'stage': None,
                'error': None,
                'progress': 0.0,
                'files': files_exist,
                'last_updated': datetime.now().isoformat()
            })
            save_process_state(str(STORAGE_DIR), space_id, state)
            
            # Start Celery task chain
            from celery_worker.tasks import process_space_chain
            chain = process_space_chain(url, str(STORAGE_DIR))
            result = chain.apply_async()
            
            # Store task ID in session state and process state
            st.session_state.current_task_id = result.id
            st.session_state.current_space_id = space_id
            state['task_id'] = result.id
            save_process_state(str(STORAGE_DIR), space_id, state)
            
            # Add to active processes and ensure it's shown in sidebar
            if space_id not in [p[0] for p in st.session_state.active_processes]:
                st.session_state.active_processes.append((space_id, state, url))
                st.rerun()  # Rerun to update sidebar immediately
            
            # Return paths for UI updates
            return storage_paths
            
        except Exception as e:
            logger.error(f"Error starting process: {str(e)}")
            st.error(f"Error: {str(e)}")
            # Clean up session state
            if 'current_task_id' in st.session_state:
                st.session_state.current_task_id = None
            return None

    def download_with_progress(url: str, output_dir: str, progress_bar: Any) -> Optional[str]:
        """Download media with progress tracking."""
        space_id = get_space_id(url)
        # Create a lock file path
        lock_file = os.path.join(output_dir, f"{space_id}.lock")
        
        try:
            # Try to acquire a lock for this download
            with ProcessLock(str(STORAGE_DIR), space_id, timeout=3600):
                # First check if MP3 already exists
                final_mp3 = os.path.join(output_dir, f"{space_id}.mp3")
                if os.path.exists(final_mp3):
                    logger.info(f"MP3 file already exists: {final_mp3}")
                    progress_bar.progress(1.0, "File already exists")
                    return final_mp3
                
                # First download without post-processing
                download_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
                    'progress_hooks': [lambda d: progress_bar.progress(d['downloaded_bytes'] / d['total_bytes'] if d['total_bytes'] else 0)],
                    'keepvideo': True,  # Keep the original file
                    'postprocessors': [],  # No post-processing yet
                }
                
                try:
                    with yt_dlp.YoutubeDL(download_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        original_file = os.path.join(output_dir, f"{info['id']}.{info['ext']}")
                        
                        # Convert to MP3 if needed
                        if info['ext'] != 'mp3':
                            if check_ffmpeg():
                                logger.info("Converting to MP3...")
                                progress_bar.progress(0.9, "Converting to MP3...")
                                
                                # Run ffmpeg conversion
                                os.system(f'ffmpeg -i "{original_file}" -vn -ar 44100 -ac 2 -b:a 192k "{final_mp3}"')
                                
                                # Clean up original file
                                os.remove(original_file)
                                
                                progress_bar.progress(1.0, "Download complete")
                                return final_mp3
                            else:
                                logger.warning("ffmpeg not found - keeping original format")
                                progress_bar.progress(1.0, "Download complete (original format)")
                                return original_file
                        
                        # If already MP3, just return the original file
                        return original_file
                        
                except Exception as e:
                    logger.error(f"Download error: {str(e)}")
                    progress_bar.progress(1.0, f"Error: {str(e)}")
                    raise
                
        except TimeoutError:
            logger.error("Timeout waiting for download lock")
            st.error("Another download is in progress. Please wait.")
            return None
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            st.error(f"Download error: {str(e)}")
            return None

    def get_space_id(url: str) -> str:
        """Extract space ID from URL and create a hash for storage."""
        space_id = url.strip('/').split('/')[-1]
        return hashlib.md5(space_id.encode()).hexdigest()

    def get_storage_paths(storage_root: str, space_id: str) -> StoragePaths:
        """Get paths for storing space data."""
        return {
            'audio_path': str(DOWNLOADS_DIR / f"{space_id}.mp3"),
            'transcript_path': str(TRANSCRIPTS_DIR / f"{space_id}.txt"),
            'quotes_path': str(QUOTES_DIR / f"{space_id}.txt"),
            'summary_path': str(SUMMARIES_DIR / f"{space_id}.json")
        }

    def find_processing_spaces() -> List[Tuple[str, ProcessState, str]]:
        """Find all spaces currently being processed."""
        logger.info("Starting find_processing_spaces scan...")
        processing_spaces: List[Tuple[str, ProcessState, str]] = []
        state_dir = STORAGE_DIR / "state"
        
        if not state_dir.exists():
            logger.debug("State directory does not exist")
            return []
        
        try:
            # Look through all state files
            state_files = list(state_dir.glob("*.json"))
            logger.debug(f"Found {len(state_files)} state files")
            
            for state_file in state_files:
                try:
                    logger.debug(f"Checking state file: {state_file}")
                    space_id = state_file.stem
                    state = get_process_state(str(STORAGE_DIR), space_id)
                    
                    # Get original URL if available
                    original_url = get_url_from_history(space_id)
                    
                    # Check if process needs to be resumed
                    if state['status'] == 'processing':
                        # Validate last update time
                        last_updated_str = state.get('last_updated')
                        if last_updated_str:
                            last_updated = datetime.fromisoformat(last_updated_str)
                            time_since_update = datetime.now() - last_updated
                            
                            # If no updates in last 5 minutes, consider it stale
                            if time_since_update > timedelta(minutes=5):
                                logger.warning(f"Found stale process {space_id}, marking as error")
                                state.update({
                                    'status': 'error',
                                    'error': 'Process state was stale',
                                    'stage': None,
                                    'progress': 0.0
                                })
                                save_process_state(str(STORAGE_DIR), space_id, state)
                                continue
                        
                        # Check if files exist but are incomplete
                        storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
                        files_exist = {
                            'audio': os.path.exists(storage_paths['audio_path']),
                            'transcript': os.path.exists(storage_paths['transcript_path']),
                            'quotes': os.path.exists(storage_paths['quotes_path']),
                            'summary': os.path.exists(storage_paths['summary_path'])
                        }
                        
                        # If we have some files but not all, this process needs to be resumed
                        if any(files_exist.values()) and not all(files_exist.values()):
                            logger.info(f"Found incomplete process {space_id} that needs resuming")
                            processing_spaces.append((space_id, state, original_url))
                            
                            # Update session state
                            if space_id not in [p[0] for p in st.session_state.active_processes]:
                                st.session_state.active_processes.append((space_id, state, original_url))
                        else:
                            # Check if recently updated
                            if last_updated_str:
                                last_updated = datetime.fromisoformat(last_updated_str)
                                time_since_update = datetime.now() - last_updated
                                
                                # Consider active if updated in last 5 minutes
                                if time_since_update <= timedelta(minutes=5):
                                    logger.info(f"Found active process {space_id}")
                                    processing_spaces.append((space_id, state, original_url))
                                    
                                    # Update session state
                                    if space_id not in [p[0] for p in st.session_state.active_processes]:
                                        st.session_state.active_processes.append((space_id, state, original_url))
                
                except Exception as e:
                    logger.error(f"Error checking state file {state_file}: {e}")
                    continue
            
            logger.info(f"Found {len(processing_spaces)} active/resumable processes")
            return processing_spaces
            
        except Exception as e:
            logger.error(f"Error in find_processing_spaces: {e}")
            return []

    def load_previous_media(space_id: str) -> Optional[StoragePaths]:
        """Load previously processed media files."""
        storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
        if all(os.path.exists(str(p)) for p in storage_paths.values()):
            st.session_state.loaded_space_id = space_id
            st.session_state.processing_complete = True
            return storage_paths
        return None

    def check_celery_task_status(task_id: str) -> Dict[str, Any]:
        """Check status of a Celery task chain."""
        result = AsyncResult(task_id, app=celery_app)
        
        if result.ready():
            if result.successful():
                return {
                    'status': 'complete',
                    'result': result.get()
                }
            else:
                # If the chain failed, get the error
                error = str(result.get(propagate=False))
                return {
                    'status': 'error',
                    'error': error
                }
        else:
            # Get current task in chain
            current_task = result.parent if result.parent else result
            return {
                'status': 'processing',
                'state': current_task.state,
                'task_id': current_task.id
            }

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
        # Add dropdown for previous media
        if DOWNLOADS_DIR.exists():
            media_files = list(DOWNLOADS_DIR.glob("*.mp3"))
            if media_files:
                # Load URL history at the start
                url_history = load_url_history()
                
                # Create two columns for the media selector
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Create options list with better formatting
                    options = ["🆕 New URL"]
                    media_info = {}  # Store full info for each media
                    
                    for f in media_files:
                        space_id = Path(f).stem
                        url = url_history.get(space_id, "Unknown URL")
                        size_mb = f.stat().st_size / (1024 * 1024)
                        
                        # Create a display name that shows the URL clearly
                        if url != "Unknown URL":
                            # Truncate URL for display in dropdown
                            display_url = url
                            if len(display_url) > 60:
                                display_url = f"{url[:30]}...{url[-25:]}"
                            display_name = f"🔗 {display_url}"
                        else:
                            display_name = f"📁 Space {space_id[:8]}"
                            
                        options.append(display_name)
                        media_info[display_name] = {
                            'space_id': space_id,
                            'url': url,
                            'size': size_mb
                        }
                    
                    # Add the dropdown with better labeling
                    selected_option = st.selectbox(
                        "Select previous media or enter new URL:",
                        options,
                        index=0,
                        key="media_selector"
                    )
                
                with col2:
                    # Show media details and load button if a previous media is selected
                    if selected_option != "🆕 New URL":
                        info = media_info[selected_option]
                        st.caption(f"Size: {info['size']:.1f} MB")
                        if st.button("📂 Load", key="load_selected"):
                            # Load the media and update all necessary state
                            st.session_state.current_space_id = info['space_id']
                            st.session_state.loaded_space_id = info['space_id']
                            st.session_state.processing_complete = True
                            # Load the URL history to ensure it's available
                            st.session_state.url_history = load_url_history()
                            st.rerun()

        # Add URL input field - show for both new URL and selected URL
        url_value = ""
        if selected_option and selected_option != "🆕 New URL":
            url_value = media_info[selected_option]['url']
        url = st.text_input("Paste Media URL:", value=url_value, key="url_input")
        
        # Process button
        if url and st.button("🚀 Process", key="process_button"):
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
            space_id = st.session_state.current_space_id
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
            # Show results if complete
            elif state['status'] == 'complete' or all(os.path.exists(str(p)) for p in storage_paths.values()):
                st.success("✅ Space processed successfully!")
                # Display results tabs
                content_tab1, content_tab2, content_tab3 = st.tabs(["📝 Quotes", "🎵 Audio", "📄 Transcript"])
                
                with content_tab1:
                    st.subheader("Generated Quotes")
                    if os.path.exists(storage_paths['quotes_path']):
                        # Add regenerate button at the top
                        col1, col2 = st.columns([4, 1])
                        with col2:
                            if st.button("Regenerate", key="regenerate_quotes"):
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
                        
                        # Read and display quotes in the main container
                        quotes = read_quotes(storage_paths['quotes_path'])
                        if quotes:  # Only try to display if we have quotes
                            display_quotes(quotes, st.container())
                        else:
                            st.warning("No quotes found in the file. Try regenerating the quotes.")
                
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
                            
                            if summary['overview'] != "Error generating summary":
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
                    recent_logs = logs_content[-20:]  # Get last 20 lines
                    with stylable_container(
                        key="detailed_logs",
                        css_styles="""
                            code {
                                white-space: pre-wrap !important;
                            }
                            """
                    ):
                        st.code(''.join(recent_logs), language='text')
            except Exception as e:
                st.warning("No logs available yet")

    with history_tab:
        st.markdown("### 📚 Previously Downloaded Media")
        
        # List all downloaded media
        if DOWNLOADS_DIR.exists():
            media_files = list(DOWNLOADS_DIR.glob("*.mp3"))
            if media_files:
                # Add search/filter box
                search = st.text_input("🔍 Search media files", 
                                     help="Filter by URL or ID")
                
                # Create a radio selection for media files
                media_options = {}
                for media_file in media_files:
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

    # In the main tab, after starting the Celery chain:
    if st.session_state.current_task_id:
        # Check task status
        task_status = check_celery_task_status(st.session_state.current_task_id)
        
        if task_status['status'] == 'processing':
            # Show progress
            with st.status("Processing media...", expanded=True) as status:
                # Display current state from process state file
                state = check_process_state(st.session_state.current_space_id)
                display_process_state(state, st.container())
                
                # Only rerun if enough time has passed
                if (st.session_state.last_check_time is None or 
                    time.time() - st.session_state.last_check_time >= 5):  # Check every 5 seconds
                    time.sleep(1)  # Small delay to prevent UI flicker
                    st.rerun()
        
        elif task_status['status'] == 'complete':
            st.success("✅ Processing complete!")
            st.session_state.processing_complete = True
            # Clear task ID since it's done
            st.session_state.current_task_id = None
            st.rerun()
        
        elif task_status['status'] == 'error':
            st.error(f"❌ Processing failed: {task_status.get('error', 'Unknown error')}")
            # Clear task ID on error
            st.session_state.current_task_id = None

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

else:
    st.stop()  # Do not continue if check_password is not True.
